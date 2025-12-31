"""
Kubeflow Pipeline for AutoML Platform - Version Legere
Optimise pour environnements avec RAM limitee (3GB)
Sans KServe - Sauvegarde du modele uniquement
"""
import kfp
from kfp import dsl
from kfp.dsl import (
    component,
    InputPath,
    OutputPath,
)
from typing import NamedTuple


@component(
    base_image="python:3.9",
    packages_to_install=[
        "pandas==2.0.3",
        "scikit-learn==1.3.2",
        "numpy==1.24.3",
    ],
)
def load_and_preprocess_data(
    output_data: OutputPath("csv"),
    output_labels: OutputPath("csv"),
) -> NamedTuple("Outputs", [("num_samples", int), ("num_features", int), ("data_path", str)]):
    """Load Iris dataset and preprocess data for training"""
    import pandas as pd
    from sklearn.datasets import load_iris
    from typing import NamedTuple

    # Load Iris dataset
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    # Separate features and labels
    labels = df['target']
    features = df.drop('target', axis=1)
    
    # Save preprocessed data
    features.to_csv(output_data, index=False)
    labels.to_csv(output_labels, index=False)
    
    print(f"Dataset loaded: {len(features)} samples, {len(features.columns)} features")
    
    outputs = NamedTuple("Outputs", [("num_samples", int), ("num_features", int), ("data_path", str)])
    return outputs(
        num_samples=len(features),
        num_features=len(features.columns),
        data_path=output_data,
    )


@component(
    base_image="python:3.9",
    packages_to_install=[
        "kubernetes==28.1.0",
        "pyyaml==6.0.1",
    ],
)
def create_katib_experiment(
    experiment_name: str,
    algorithm: str,
    max_trials: int,
    parallel_trials: int,
    namespace: str,
    training_image: str,
    experiment_config: OutputPath("yaml"),
) -> str:
    """Create Katib experiment configuration"""
    import yaml

    # Experiment template based on algorithm
    experiment_template = {
        "apiVersion": "kubeflow.org/v1beta1",
        "kind": "Experiment",
        "metadata": {
            "name": experiment_name,
            "namespace": namespace,
        },
        "spec": {
            "algorithm": {
                "algorithmName": algorithm,
            },
            "objective": {
                "type": "maximize",
                "goal": 0.99,
                "objectiveMetricName": "accuracy",
                "additionalMetricNames": ["loss"],
            },
            "parameters": [
                {
                    "name": "learning_rate",
                    "parameterType": "double",
                    "feasibleSpace": {"min": "0.001", "max": "0.1"},
                },
                {
                    "name": "num_layers",
                    "parameterType": "int",
                    "feasibleSpace": {"min": "1", "max": "3"},
                },
                {
                    "name": "optimizer",
                    "parameterType": "categorical",
                    "feasibleSpace": {"list": ["sgd", "adam", "rmsprop"]},
                },
            ],
            "parallelTrialCount": parallel_trials,
            "maxTrialCount": max_trials,
            "maxFailedTrialCount": 3,
            "trialTemplate": {
                "primaryContainerName": "training-container",
                "trialParameters": [
                    {
                        "name": "learningRate",
                        "description": "Learning rate for the training model",
                        "reference": "learning_rate",
                    },
                    {
                        "name": "numLayers",
                        "description": "Number of layers for the model",
                        "reference": "num_layers",
                    },
                    {
                        "name": "optimizer",
                        "description": "Optimizer algorithm",
                        "reference": "optimizer",
                    },
                ],
                "trialSpec": {
                    "apiVersion": "batch/v1",
                    "kind": "Job",
                    "spec": {
                        "template": {
                            "metadata": {
                                "annotations": {
                                    "sidecar.istio.io/inject": "false",
                                }
                            },
                            "spec": {
                                "containers": [
                                    {
                                        "name": "training-container",
                                        "image": training_image,
                                        "imagePullPolicy": "IfNotPresent",
                                        "command": [
                                            "python",
                                            "-u",
                                            "/app/src/training/train.py",
                                            "--learning-rate=${trialParameters.learningRate}",
                                            "--num-layers=${trialParameters.numLayers}",
                                            "--optimizer=${trialParameters.optimizer}",
                                        ],
                                        "resources": {
                                            "limits": {"cpu": "500m", "memory": "1Gi"},
                                            "requests": {"cpu": "250m", "memory": "512Mi"},
                                        },
                                    }
                                ],
                                "restartPolicy": "Never",
                            }
                        }
                    },
                },
            },
        },
    }

    # Save experiment config
    with open(experiment_config, "w") as f:
        yaml.dump(experiment_template, f)

    print(f"Katib experiment configuration created: {experiment_name}")
    return experiment_name


@component(
    base_image="python:3.9",
    packages_to_install=[
        "kubernetes==28.1.0",
        "pyyaml==6.0.1",
    ],
)
def submit_katib_experiment(
    experiment_name: str,
    namespace: str,
    experiment_config: InputPath("yaml"),
    max_wait_time: int = 3600,
) -> NamedTuple("Outputs", [("best_trial", str), ("best_accuracy", float), ("best_params", str)]):
    """Submit and monitor Katib experiment"""
    import yaml
    import time
    import json
    from kubernetes import client, config
    from typing import NamedTuple

    # Load kubeconfig
    try:
        config.load_incluster_config()
    except:
        config.load_kube_config()

    # Load experiment config
    with open(experiment_config, "r") as f:
        exp_dict = yaml.safe_load(f)

    # Create Katib API client
    custom_api = client.CustomObjectsApi()

    # Create experiment
    try:
        custom_api.create_namespaced_custom_object(
            group="kubeflow.org",
            version="v1beta1",
            namespace=namespace,
            plural="experiments",
            body=exp_dict,
        )
        print(f"Katib experiment {experiment_name} submitted")
    except client.exceptions.ApiException as e:
        if e.status == 409:
            print(f"Experiment {experiment_name} already exists, monitoring existing experiment")
        else:
            raise e

    # Wait for experiment to complete
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        try:
            exp = custom_api.get_namespaced_custom_object(
                group="kubeflow.org",
                version="v1beta1",
                namespace=namespace,
                plural="experiments",
                name=experiment_name,
            )
            
            status = exp.get("status", {})
            conditions = status.get("conditions", [])
            
            # Check if experiment is completed
            for condition in conditions:
                if condition.get("type") == "Succeeded" and condition.get("status") == "True":
                    # Experiment completed successfully
                    current_optimal = status.get("currentOptimalTrial", {})
                    best_trial_name = current_optimal.get("bestTrialName", "")
                    
                    observation = current_optimal.get("observation", {})
                    metrics = observation.get("metrics", [])
                    
                    best_accuracy = 0.0
                    for metric in metrics:
                        if metric.get("name") == "accuracy":
                            best_accuracy = float(metric.get("latest", "0.0"))
                            break
                    
                    # Get best parameters
                    param_assignments = current_optimal.get("parameterAssignments", [])
                    best_params = {p.get("name"): p.get("value") for p in param_assignments}
                    
                    print(f"Experiment completed successfully!")
                    print(f"Best trial: {best_trial_name}")
                    print(f"Best accuracy: {best_accuracy}")
                    print(f"Best parameters: {best_params}")
                    
                    outputs = NamedTuple("Outputs", [("best_trial", str), ("best_accuracy", float), ("best_params", str)])
                    return outputs(
                        best_trial=best_trial_name,
                        best_accuracy=best_accuracy,
                        best_params=json.dumps(best_params)
                    )
                
                elif condition.get("type") == "Failed" and condition.get("status") == "True":
                    print(f"Experiment failed: {condition.get('message', 'Unknown error')}")
                    outputs = NamedTuple("Outputs", [("best_trial", str), ("best_accuracy", float), ("best_params", str)])
                    return outputs(best_trial="", best_accuracy=0.0, best_params="{}")
            
            # Print progress
            trials_running = status.get("trialsRunning", 0)
            trials_succeeded = status.get("trialsSucceeded", 0)
            trials_failed = status.get("trialsFailed", 0)
            print(f"Progress - Running: {trials_running}, Succeeded: {trials_succeeded}, Failed: {trials_failed}")
            
        except Exception as e:
            print(f"Error checking experiment status: {e}")
        
        time.sleep(30)  # Check every 30 seconds

    # Timeout
    print(f"Experiment timed out after {max_wait_time} seconds")
    outputs = NamedTuple("Outputs", [("best_trial", str), ("best_accuracy", float), ("best_params", str)])
    return outputs(best_trial="", best_accuracy=0.0, best_params="{}")


@component(
    base_image="python:3.9",
    packages_to_install=[
        "pandas==2.0.3",
        "scikit-learn==1.3.2",
        "numpy==1.24.3",
    ],
)
def save_best_model_info(
    best_trial: str,
    best_accuracy: float,
    best_params: str,
    model_report: OutputPath("txt"),
) -> NamedTuple("Outputs", [("report_summary", str)]):
    """Save best model information and metrics"""
    import json
    from typing import NamedTuple
    from datetime import datetime
    
    params = json.loads(best_params) if best_params else {}
    
    report = f"""
{'='*60}
AUTOML PIPELINE - RESULTATS FINAUX
{'='*60}

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

MEILLEUR MODELE TROUVE:
-----------------------
Trial: {best_trial}
Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)

HYPERPARAMETRES OPTIMAUX:
------------------------
"""
    for key, value in params.items():
        report += f"  {key}: {value}\n"
    
    report += f"""
COMMENT UTILISER CE MODELE:
---------------------------
1. Les meilleurs hyperparametres sont ci-dessus
2. Entrainez un modele final avec ces parametres
3. Le modele est sauvegarde dans le container du best trial
4. Recuperez le modele avec: kubectl cp <pod>:/tmp/model/model.h5 ./model.h5

COMMANDE POUR RECUPERER LE MODELE:
----------------------------------
# Trouvez le pod du meilleur trial
kubectl get pods -n kubeflow | grep {best_trial.split('-trial-')[0] if best_trial else 'experiment'}

# Copiez le modele
kubectl cp kubeflow/<pod-name>:/tmp/model/model.h5 ./iris_model.h5

ALTERNATIVE - Entrainer localement avec les meilleurs params:
-------------------------------------------------------------
python train.py \\
    --learning-rate={params.get('learning_rate', 0.01)} \\
    --num-layers={params.get('num_layers', 2)} \\
    --optimizer={params.get('optimizer', 'adam')}

{'='*60}
"""
    
    # Save report
    with open(model_report, 'w') as f:
        f.write(report)
    
    print(report)
    
    summary = f"Best model: {best_trial} | Accuracy: {best_accuracy:.4f}"
    
    outputs = NamedTuple("Outputs", [("report_summary", str)])
    return outputs(report_summary=summary)


@dsl.pipeline(
    name="AutoML Pipeline Leger (sans KServe)",
    description="Pipeline AutoML optimise pour 3GB RAM - Katib seulement",
)
def automl_pipeline(
    algorithm: str = "random",
    max_trials: int = 5,
    parallel_trials: int = 1,
    model_name: str = "iris-automl-model",
    namespace: str = "kubeflow",
    training_image: str = "automl-training:latest",
):
    """Pipeline AutoML leger sans KServe"""
    
    # Step 1: Load and preprocess data
    preprocess_task = load_and_preprocess_data()
    
    # Step 2: Create Katib experiment configuration
    create_exp_task = create_katib_experiment(
        experiment_name=f"{model_name}-experiment",
        algorithm=algorithm,
        max_trials=max_trials,
        parallel_trials=parallel_trials,
        namespace=namespace,
        training_image=training_image,
    )
    create_exp_task.after(preprocess_task)
    
    # Step 3: Submit and monitor Katib experiment
    katib_task = submit_katib_experiment(
        experiment_name=f"{model_name}-experiment",
        namespace=namespace,
        experiment_config=create_exp_task.outputs["experiment_config"],
        max_wait_time=3600,
    )
    katib_task.after(create_exp_task)
    
    # Step 4: Save best model information
    save_model_task = save_best_model_info(
        best_trial=katib_task.outputs["best_trial"],
        best_accuracy=katib_task.outputs["best_accuracy"],
        best_params=katib_task.outputs["best_params"],
    )
    save_model_task.after(katib_task)


if __name__ == "__main__":
    # Compile pipeline
    print("Compiling AutoML pipeline (light version - no KServe)...")
    kfp.compiler.Compiler().compile(
        pipeline_func=automl_pipeline,
        package_path="automl_pipeline.yaml",
    )
    print("[OK] Pipeline compiled successfully: automl_pipeline.yaml")
    
    print("\n" + "="*60)
    print("CONFIGURATION OPTIMISEE POUR 3GB RAM")
    print("="*60)
    print("\nCe pipeline utilise:")
    print("  [+] Katib pour l'optimisation des hyperparametres")
    print("  [-] Pas de KServe (economise ~1-1.5GB RAM)")
    print("\nParametres recommandes:")
    print("  - max_trials: 5-10")
    print("  - parallel_trials: 1 (IMPORTANT pour 3GB RAM!)")
    print("\nRessources par trial:")
    print("  - CPU: 250m-500m")
    print("  - Memory: 512Mi-1Gi")
    print("\nLe modele sera sauvegarde dans le pod du best trial")
    print("Vous pourrez le recuperer avec kubectl cp")
    print("="*60)
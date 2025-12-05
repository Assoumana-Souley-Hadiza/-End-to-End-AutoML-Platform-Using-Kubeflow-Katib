"""
Kubeflow Pipeline for AutoML Platform
Integrates Katib for hyperparameter tuning and NAS
"""
import kfp
from kfp import dsl
from kfp.dsl import (
    component,
    Input,
    Output,
    Artifact,
    Model,
    Metrics,
    Dataset,
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
    data_path: str,
    output_data: OutputPath("csv"),
    output_labels: OutputPath("csv"),
) -> NamedTuple("Outputs", [("num_samples", int), ("num_features", int)]):
    """Load and preprocess data for training"""
    import pandas as pd
    import numpy as np
    from typing import NamedTuple

    # Load data
    df = pd.read_csv(data_path)
    
    # Simple preprocessing example
    # Separate features and labels
    if "target" in df.columns:
        labels = df["target"]
        features = df.drop("target", axis=1)
    else:
        labels = df.iloc[:, -1]
        features = df.iloc[:, :-1]
    
    # Save preprocessed data
    features.to_csv(output_data, index=False)
    labels.to_csv(output_labels, index=False)
    
    outputs = NamedTuple("Outputs", [("num_samples", int), ("num_features", int)])
    return outputs(
        num_samples=len(features),
        num_features=len(features.columns),
    )


@component(
    base_image="python:3.9",
    packages_to_install=[
        "kubeflow-katib==0.15.0",
        "kubernetes==28.1.0",
        "pyyaml==6.0.1",
    ],
)
def create_katib_experiment(
    experiment_name: str,
    algorithm: str,
    max_trials: int,
    parallel_trials: int,
    data_path: str,
    experiment_config: OutputPath("yaml"),
) -> str:
    """Create Katib experiment configuration"""
    import yaml
    from kubernetes import client, config

    # Load kubeconfig
    try:
        config.load_incluster_config()
    except:
        config.load_kube_config()

    # Experiment template based on algorithm
    experiment_template = {
        "apiVersion": "kubeflow.org/v1beta1",
        "kind": "Experiment",
        "metadata": {
            "name": experiment_name,
            "namespace": "kubeflow",
        },
        "spec": {
            "algorithm": {
                "algorithmName": algorithm,
            },
            "objective": {
                "type": "maximize",
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
                    "name": "batch_size",
                    "parameterType": "int",
                    "feasibleSpace": {"min": "16", "max": "128"},
                },
            ],
            "parallelTrialCount": parallel_trials,
            "maxTrialCount": max_trials,
            "trialTemplate": {
                "primaryContainerName": "training-container",
                "trialParameters": [
                    {
                        "name": "learningRate",
                        "reference": "${trialParameters.learningRate}",
                    },
                    {
                        "name": "batchSize",
                        "reference": "${trialParameters.batchSize}",
                    },
                ],
                "trialSpec": {
                    "apiVersion": "batch/v1",
                    "kind": "Job",
                    "spec": {
                        "template": {
                            "spec": {
                                "containers": [
                                    {
                                        "name": "training-container",
                                        "image": "automl-training:latest",
                                        "command": [
                                            "python",
                                            "/app/src/training/train.py",
                                            "--learning-rate=${trialParameters.learningRate}",
                                            "--batch-size=${trialParameters.batchSize}",
                                            "--data-path=" + data_path,
                                        ],
                                        "resources": {
                                            "limits": {"cpu": "2", "memory": "4Gi"},
                                            "requests": {"cpu": "1", "memory": "2Gi"},
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

    return experiment_name


@component(
    base_image="python:3.9",
    packages_to_install=[
        "kubeflow-katib==0.15.0",
        "kubernetes==28.1.0",
    ],
)
def submit_katib_experiment(
    experiment_name: str,
    experiment_config: InputPath("yaml"),
) -> NamedTuple("Outputs", [("best_trial", str), ("best_accuracy", float)]):
    """Submit and monitor Katib experiment"""
    import yaml
    import time
    from kubernetes import client, config
    from kubeflow.katib import ApiClient, V1beta1Experiment
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
    api_client = ApiClient()
    custom_api = client.CustomObjectsApi(api_client)

    # Create experiment
    namespace = "kubeflow"
    custom_api.create_namespaced_custom_object(
        group="kubeflow.org",
        version="v1beta1",
        namespace=namespace,
        plural="experiments",
        body=exp_dict,
    )

    # Wait for experiment to complete
    max_wait_time = 3600  # 1 hour
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        exp = custom_api.get_namespaced_custom_object(
            group="kubeflow.org",
            version="v1beta1",
            namespace=namespace,
            plural="experiments",
            name=experiment_name,
        )
        
        status = exp.get("status", {})
        if status.get("completionTime"):
            # Experiment completed
            best_trial = status.get("currentOptimalTrial", {}).get("bestTrialName", "")
            best_metrics = status.get("currentOptimalTrial", {}).get("observation", {}).get("metrics", [])
            best_accuracy = 0.0
            for metric in best_metrics:
                if metric.get("name") == "accuracy":
                    best_accuracy = float(metric.get("latest", "0.0"))
                    break
            
            outputs = NamedTuple("Outputs", [("best_trial", str), ("best_accuracy", float)])
            return outputs(best_trial=best_trial, best_accuracy=best_accuracy)
        
        time.sleep(30)  # Check every 30 seconds

    # Timeout
    outputs = NamedTuple("Outputs", [("best_trial", str), ("best_accuracy", float)])
    return outputs(best_trial="", best_accuracy=0.0)


@component(
    base_image="python:3.9",
    packages_to_install=[
        "kserve==0.11.0",
        "kubernetes==28.1.0",
        "pyyaml==6.0.1",
    ],
)
def export_to_kfserving(
    model_name: str,
    model_path: str,
    best_trial: str,
    serving_config: OutputPath("yaml"),
) -> str:
    """Export best model to KFServing"""
    import yaml
    from kubernetes import client, config

    # Load kubeconfig
    try:
        config.load_incluster_config()
    except:
        config.load_kube_config()

    # KFServing InferenceService configuration
    inference_service = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": model_name,
            "namespace": "kubeflow",
            "annotations": {
                "serving.kserve.io/deploymentMode": "Serverless",
            },
        },
        "spec": {
            "predictor": {
                "tensorflow": {
                    "storageUri": model_path,
                    "resources": {
                        "requests": {"cpu": "1", "memory": "2Gi"},
                        "limits": {"cpu": "2", "memory": "4Gi"},
                    },
                }
            },
        },
    }

    # Save serving config
    with open(serving_config, "w") as f:
        yaml.dump(inference_service, f)

    return model_name


@component(
    base_image="python:3.9",
    packages_to_install=[
        "kserve==0.11.0",
        "kubernetes==28.1.0",
    ],
)
def deploy_kfserving(
    model_name: str,
    serving_config: InputPath("yaml"),
) -> str:
    """Deploy model to KFServing"""
    import yaml
    from kubernetes import client, config

    # Load kubeconfig
    try:
        config.load_incluster_config()
    except:
        config.load_kube_config()

    # Load serving config
    with open(serving_config, "r") as f:
        service_dict = yaml.safe_load(f)

    # Create KFServing API client
    api_client = client.ApiClient()
    custom_api = client.CustomObjectsApi(api_client)

    # Create InferenceService
    namespace = "kubeflow"
    try:
        custom_api.create_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            body=service_dict,
        )
        return f"Model {model_name} deployed successfully"
    except Exception as e:
        # Update if exists
        custom_api.patch_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            name=model_name,
            body=service_dict,
        )
        return f"Model {model_name} updated successfully"


@dsl.pipeline(
    name="AutoML Pipeline",
    description="End-to-end AutoML pipeline with Katib and KFServing",
)
def automl_pipeline(
    data_path: str = "/data/train.csv",
    algorithm: str = "bayesian-optimization",
    max_trials: int = 30,
    parallel_trials: int = 3,
    model_name: str = "automl-model",
):
    """Main AutoML pipeline"""
    
    # Step 1: Load and preprocess data
    preprocess_task = load_and_preprocess_data(data_path=data_path)
    
    # Step 2: Create Katib experiment
    create_exp_task = create_katib_experiment(
        experiment_name=f"{model_name}-experiment",
        algorithm=algorithm,
        max_trials=max_trials,
        parallel_trials=parallel_trials,
        data_path=preprocess_task.outputs["output_data"],
    )
    
    # Step 3: Submit and monitor Katib experiment
    katib_task = submit_katib_experiment(
        experiment_name=create_exp_task.output,
        experiment_config=create_exp_task.outputs["experiment_config"],
    )
    
    # Step 4: Export best model to KFServing
    export_task = export_to_kfserving(
        model_name=model_name,
        model_path=f"s3://models/{model_name}",
        best_trial=katib_task.outputs["best_trial"],
    )
    
    # Step 5: Deploy to KFServing
    deploy_task = deploy_kfserving(
        model_name=model_name,
        serving_config=export_task.outputs["serving_config"],
    )
    
    # Set dependencies
    create_exp_task.after(preprocess_task)
    katib_task.after(create_exp_task)
    export_task.after(katib_task)
    deploy_task.after(export_task)


if __name__ == "__main__":
    # Compile pipeline
    kfp.compiler.Compiler().compile(
        pipeline_func=automl_pipeline,
        package_path="automl_pipeline.yaml",
    )




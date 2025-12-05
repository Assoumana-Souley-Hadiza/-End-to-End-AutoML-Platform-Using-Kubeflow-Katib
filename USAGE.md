# Guide d'Utilisation - Plateforme AutoML Kubeflow + Katib

## Vue d'ensemble

Cette plateforme AutoML permet de :
- Optimiser automatiquement les hyperparamètres avec Katib
- Effectuer de la recherche d'architecture neuronale (NAS)
- Orchestrer des pipelines ML avec Kubeflow Pipelines
- Déployer des modèles avec KFServing

## Démarrage Rapide

### 1. Installation

```bash
# Installer les dépendances
pip install -r requirements.txt

# Configurer l'accès Kubernetes
kubectl config use-context <votre-context>
```

### 2. Hyperparameter Tuning

#### Bayesian Optimization

```bash
# Soumettre une expérience
kubectl apply -f katib/hyperparameter-tuning/bayesian-optimization.yaml

# Suivre l'expérience
kubectl get experiment bayesian-optimization-experiment -n kubeflow -w

# Voir les résultats
kubectl get trials -n kubeflow
```

#### TPE (Tree-structured Parzen Estimator)

```bash
kubectl apply -f katib/hyperparameter-tuning/tpe-optimization.yaml
```

#### Genetic Algorithm

```bash
kubectl apply -f katib/hyperparameter-tuning/genetic-algorithm.yaml
```

#### Population Based Training (PBT)

```bash
kubectl apply -f katib/hyperparameter-tuning/pbt-optimization.yaml
```

#### Grid Search et Random Search

```bash
# Grid Search
kubectl apply -f katib/hyperparameter-tuning/grid-search.yaml

# Random Search
kubectl apply -f katib/hyperparameter-tuning/random-search.yaml
```

### 3. Neural Architecture Search (NAS)

```bash
# ENAS
kubectl apply -f katib/nas/nas-experiment.yaml

# DARTS
kubectl apply -f katib/nas/darts-nas.yaml
```

### 4. Pipeline End-to-End avec KFP

```bash
# Compiler le pipeline
python scripts/compile_pipeline.py --output automl_pipeline.yaml

# Utiliser le pipeline via Python
python -c "
from pipelines.kfp import automl_pipeline
import kfp

client = kfp.Client(host='http://votre-kubeflow-host')
experiment = client.create_experiment(name='mon-experiment')
run = client.run_pipeline(
    experiment_id=experiment.id,
    job_name='mon-run',
    pipeline_package_path='automl_pipeline.yaml',
    arguments={
        'data_path': '/data/train.csv',
        'algorithm': 'bayesian-optimization',
        'max_trials': 30,
        'model_name': 'mon-modele'
    }
)
"
```

### 5. Déploiement avec KFServing

```bash
# Déployer un modèle TensorFlow
kubectl apply -f serving/kfserving/tensorflow-inference-service.yaml

# Vérifier le statut
kubectl get inferenceservice -n kubeflow

# Obtenir l'URL du service
kubectl get inferenceservice automl-tensorflow-model -n kubeflow \
  -o jsonpath='{.status.url}'
```

## Utilisation des Scripts

### Soumettre une expérience Katib

```bash
python scripts/submit_katib_experiment.py \
  katib/hyperparameter-tuning/bayesian-optimization.yaml \
  --namespace kubeflow
```

### Déployer un modèle

```bash
python scripts/deploy_kfserving.py \
  serving/kfserving/tensorflow-inference-service.yaml \
  --namespace kubeflow
```

## Personnalisation

### Modifier les Hyperparamètres

Éditez les fichiers YAML dans `katib/hyperparameter-tuning/` :

```yaml
parameters:
  - name: learning_rate
    parameterType: double
    feasibleSpace:
      min: "0.001"    # Modifier ici
      max: "0.1"      # Modifier ici
```

### Changer l'Algorithme

Modifiez la section `algorithm` :

```yaml
algorithm:
  algorithmName: tpe  # Changer l'algorithme
  algorithmSettings:
    - name: "n_startup_trials"
      value: "10"
```

### Personnaliser le Training

Modifiez `src/training/train.py` pour adapter le script d'entraînement à vos besoins.

## Monitoring

### Voir les Métriques

```bash
# Logs d'un trial
kubectl logs <trial-name> -n kubeflow

# Détails d'une expérience
kubectl describe experiment <experiment-name> -n kubeflow
```

### Interface Web

Accédez à l'interface Kubeflow pour visualiser :
- Les expériences Katib
- Les pipelines KFP
- Les services KFServing

## Troubleshooting

### Problème : Expérience ne démarre pas

```bash
# Vérifier les pods
kubectl get pods -n kubeflow

# Vérifier les événements
kubectl get events -n kubeflow --sort-by='.lastTimestamp'
```

### Problème : Modèle non déployé

```bash
# Vérifier les logs KFServing
kubectl logs -n kubeflow -l serving.kserve.io/inferenceservice=automl-tensorflow-model
```

## Ressources

- [Documentation Katib](https://www.kubeflow.org/docs/components/katib/)
- [Documentation KFP](https://www.kubeflow.org/docs/components/pipelines/)
- [Documentation KFServing](https://kserve.github.io/website/)




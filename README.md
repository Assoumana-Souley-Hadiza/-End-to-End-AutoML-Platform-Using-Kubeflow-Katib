# End-to-End AutoML Platform Using Kubeflow + Katib

Une plateforme AutoML complète utilisant Kubeflow Pipelines (KFP) et Katib pour l'optimisation automatique de modèles de machine learning.

## Fonctionnalités

- **Hyperparameter Tuning** : Support de plusieurs algorithmes d'optimisation
  - Bayesian Optimization
  - Tree-structured Parzen Estimator (TPE)
  - Genetic Algorithm (GA)
  - Population Based Training (PBT)
  - Grid Search (distribué)
  - Random Search (distribué)

- **Neural Architecture Search (NAS)** : Recherche automatique d'architectures de réseaux de neurones

- **Kubeflow Pipelines Integration** : Pipelines ML orchestrés avec KFP

- **Model Serving** : Export et déploiement des meilleurs modèles via KFServing

## Architecture

```
┌─────────────────┐
│   KFP Pipeline  │
│   (Orchestrator)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Katib       │
│  (HP Tuning/NAS)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Training Jobs  │
│  (Distributed)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   KFServing     │
│  (Model Serving)│
└─────────────────┘
```

## Structure du Projet

```
├── katib/
│   ├── hyperparameter-tuning/    # Configurations HP tuning
│   ├── nas/                      # Configurations NAS
│   └── experiments/              # Exemples d'expériences
├── pipelines/
│   └── kfp/                      # Kubeflow Pipelines
├── src/
│   ├── training/                 # Scripts d'entraînement
│   ├── models/                   # Définitions de modèles
│   └── utils/                    # Utilitaires
├── serving/
│   └── kfserving/                # Configurations KFServing
├── notebooks/                    # Notebooks d'exemple
├── docker/                       # Dockerfiles
└── docs/                         # Documentation
```

## Prérequis

- Kubernetes cluster
- Kubeflow installé
- Katib installé
- KFServing installé
- Python 3.8+

## Installation

1. Installer les dépendances Python:
```bash
pip install -r requirements.txt
```

2. Configurer l'accès à votre cluster Kubernetes:
```bash
kubectl config set-context <your-context>
```

3. Déployer les composants:
```bash
kubectl apply -f katib/
kubectl apply -f serving/kfserving/
```

## Utilisation

### Hyperparameter Tuning

```bash
kubectl apply -f katib/hyperparameter-tuning/bayesian-optimization.yaml
```

### Neural Architecture Search

```bash
kubectl apply -f katib/nas/nas-experiment.yaml
```

### Exécuter un Pipeline KFP

```python
from pipelines.kfp import automl_pipeline

# Compiler et soumettre le pipeline
automl_pipeline.compile()
automl_pipeline.submit()
```

## Documentation

Voir le dossier `docs/` pour plus de détails sur:
- Configuration de Katib
- Création de pipelines KFP
- Déploiement avec KFServing
- Exemples d'utilisation

## License

MIT




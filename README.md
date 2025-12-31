# End-to-End AutoML Platform Using Kubeflow + Katib

Une plateforme AutoML complète utilisant Kubeflow Pipelines (KFP) et Katib pour l'optimisation automatique de modèles de machine learning.

## Fonctionnalités

- **Hyperparameter Tuning** : Support de plusieurs algorithmes d'optimisation
  - Bayesian Optimization
  - Tree-structured Parzen Estimator (TPE)
  - Random Search (distribué)

- **Kubeflow Pipelines Integration** : Pipelines ML orchestrés avec KFP


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
│  (HP Tuning)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Training Jobs  │
│  (Distributed)  │
└────────┬────────┘
         │
```

## Structure du Projet

```
├── katib/
│   ├── hyperparameter-tuning/    # Configurations HP tuning                      # Configurations NAS
│   └── experiments/              # Exemples d'expériences
├── pipelines/
│   └── kfp/                      # Kubeflow Pipelines
├── src/
│   ├── training/                 # Scripts d'entraînement
│   ├── models/                   # Définitions de modèles
│   └── utils/                    # Utilitaires
# Configurations KFServing
├── notebooks/                    # Notebooks d'exemple
├── docker/                       # Dockerfiles
└── docs/                         # Documentation
```

## Prérequis

- Kubernetes cluster
- Kubeflow installé
- Katib installé
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
```
## Utilisation

### Hyperparameter Tuning

```bash
kubectl apply -f katib/hyperparameter-tuning/bayesian-optimization.yaml
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
- Exemples d'utilisation

## License

MIT




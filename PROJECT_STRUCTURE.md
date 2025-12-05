# Structure du Projet

## Vue d'ensemble

```
Plateforme_autoML_end_to_end/
│
├── README.md                    # Documentation principale
├── USAGE.md                     # Guide d'utilisation
├── requirements.txt             # Dépendances Python
├── setup.py                     # Configuration du package
├── .gitignore                   # Fichiers à ignorer par Git
│
├── config/                      # Configuration
│   └── config.yaml              # Configuration principale
│
├── katib/                       # Configurations Katib
│   ├── README.md
│   ├── hyperparameter-tuning/   # Expériences HP tuning
│   │   ├── bayesian-optimization.yaml
│   │   ├── tpe-optimization.yaml
│   │   ├── genetic-algorithm.yaml
│   │   ├── pbt-optimization.yaml
│   │   ├── grid-search.yaml
│   │   └── random-search.yaml
│   ├── nas/                     # Expériences NAS
│   │   ├── nas-experiment.yaml
│   │   └── darts-nas.yaml
│   └── experiments/             # Exemples
│       └── README.md
│
├── pipelines/                   # Kubeflow Pipelines
│   ├── README.md
│   └── kfp/
│       ├── __init__.py
│       └── automl_pipeline.py   # Pipeline principal
│
├── src/                         # Code source
│   ├── training/                # Scripts d'entraînement
│   │   ├── __init__.py
│   │   ├── train.py             # Training standard
│   │   ├── train_pbt.py         # Training PBT
│   │   ├── train_nas.py         # Training NAS
│   │   └── train_darts.py       # Training DARTS
│   ├── models/                  # Gestion des modèles
│   │   ├── __init__.py
│   │   ├── base_model.py        # Classe de base
│   │   └── model_registry.py    # Registre des modèles
│   ├── utils/                   # Utilitaires
│   │   ├── __init__.py
│   │   ├── config.py            # Gestion config
│   │   └── logger.py            # Logging
│   └── serving/                 # Model serving
│       └── __init__.py
│
├── serving/                     # Configurations KFServing
│   └── kfserving/
│       ├── README.md
│       ├── tensorflow-inference-service.yaml
│       ├── pytorch-inference-service.yaml
│       ├── sklearn-inference-service.yaml
│       └── custom-predictor.yaml
│
├── scripts/                     # Scripts utilitaires
│   ├── README.md
│   ├── submit_katib_experiment.py
│   ├── deploy_kfserving.py
│   └── compile_pipeline.py
│
├── docker/                      # Dockerfiles
│   ├── Dockerfile.training
│   ├── Dockerfile.serving
│   └── docker-compose.yml
│
├── notebooks/                   # Notebooks Jupyter
│   └── example_katib_experiment.ipynb
│
└── docs/                        # Documentation
    ├── GETTING_STARTED.md
    ├── ARCHITECTURE.md
    └── EXAMPLES.md
```

## Description des Composants

### Katib Configurations

- **Hyperparameter Tuning** : 6 algorithmes différents (Bayesian, TPE, GA, PBT, Grid, Random)
- **NAS** : 2 approches (ENAS, DARTS)
- Tous les fichiers sont prêts à être déployés sur Kubernetes

### Pipelines KFP

- Pipeline end-to-end complet
- Intègre Katib, training, et KFServing
- Compilable et déployable sur Kubeflow

### Scripts d'Entraînement

- Support multi-frameworks (TensorFlow, PyTorch, Scikit-learn)
- Compatible avec tous les algorithmes Katib
- Logging des métriques au format Katib

### KFServing

- Configurations pour TensorFlow, PyTorch, Scikit-learn
- Support de custom predictors
- Auto-scaling configuré

### Scripts Utilitaires

- Soumission d'expériences Katib
- Déploiement KFServing
- Compilation de pipelines

## Fichiers Clés

1. **README.md** : Documentation principale
2. **USAGE.md** : Guide d'utilisation détaillé
3. **requirements.txt** : Toutes les dépendances
4. **config/config.yaml** : Configuration centralisée
5. **pipelines/kfp/automl_pipeline.py** : Pipeline principal

## Prochaines Étapes

1. Installer les dépendances : `pip install -r requirements.txt`
2. Configurer Kubernetes et Kubeflow
3. Adapter les configurations à votre environnement
4. Lancer votre première expérience !




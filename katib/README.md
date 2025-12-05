# Katib Configuration

This directory contains Katib experiment configurations for hyperparameter tuning and Neural Architecture Search (NAS).

## Directory Structure

```
katib/
├── hyperparameter-tuning/    # HP tuning experiments
│   ├── bayesian-optimization.yaml
│   ├── tpe-optimization.yaml
│   ├── genetic-algorithm.yaml
│   ├── pbt-optimization.yaml
│   ├── grid-search.yaml
│   └── random-search.yaml
├── nas/                      # NAS experiments
│   ├── nas-experiment.yaml
│   └── darts-nas.yaml
└── experiments/              # Example experiments
    └── README.md
```

## Supported Algorithms

### Hyperparameter Tuning

1. **Bayesian Optimization**: Uses Gaussian processes to model the objective function
2. **TPE (Tree-structured Parzen Estimator)**: Sequential model-based optimization
3. **Genetic Algorithm**: Evolutionary approach with mutation and crossover
4. **PBT (Population Based Training)**: Trains and mutates a population of models
5. **Grid Search**: Exhaustive search over parameter grid
6. **Random Search**: Random sampling of parameter space

### Neural Architecture Search

1. **ENAS**: Efficient Neural Architecture Search
2. **DARTS**: Differentiable Architecture Search

## Customization

To customize experiments, edit the YAML files and modify:

- `spec.parameters`: Hyperparameters to tune
- `spec.algorithm.algorithmSettings`: Algorithm-specific settings
- `spec.trialTemplate.trialSpec`: Training job specification
- `spec.objective`: Optimization objective and metrics




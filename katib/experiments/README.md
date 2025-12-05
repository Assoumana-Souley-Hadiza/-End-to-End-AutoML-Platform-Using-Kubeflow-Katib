# Katib Experiments

This directory contains example Katib experiment configurations.

## Hyperparameter Tuning Experiments

- `../hyperparameter-tuning/bayesian-optimization.yaml` - Bayesian optimization
- `../hyperparameter-tuning/tpe-optimization.yaml` - Tree-structured Parzen Estimator
- `../hyperparameter-tuning/genetic-algorithm.yaml` - Genetic Algorithm
- `../hyperparameter-tuning/pbt-optimization.yaml` - Population Based Training
- `../hyperparameter-tuning/grid-search.yaml` - Grid Search
- `../hyperparameter-tuning/random-search.yaml` - Random Search

## Neural Architecture Search

- `../nas/nas-experiment.yaml` - ENAS (Efficient Neural Architecture Search)
- `../nas/darts-nas.yaml` - DARTS (Differentiable Architecture Search)

## Usage

Submit an experiment:

```bash
kubectl apply -f <experiment-file>.yaml
```

Monitor experiment:

```bash
kubectl get experiment <experiment-name> -n kubeflow
kubectl describe experiment <experiment-name> -n kubeflow
```

View trials:

```bash
kubectl get trials -n kubeflow
kubectl logs <trial-name> -n kubeflow
```




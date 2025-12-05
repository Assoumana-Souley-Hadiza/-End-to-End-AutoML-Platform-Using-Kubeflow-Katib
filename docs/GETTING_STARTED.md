# Getting Started Guide

## Prerequisites

Before using the AutoML platform, ensure you have:

1. **Kubernetes Cluster** (v1.20+)
2. **Kubeflow** installed and configured
3. **Katib** component installed
4. **KFServing** (KServe) installed
5. **Python 3.8+** with required packages

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Kubernetes Access

```bash
kubectl config set-context <your-context>
kubectl config use-context <your-context>
```

### 3. Verify Kubeflow Installation

```bash
kubectl get pods -n kubeflow
kubectl get experiments -n kubeflow
```

## Quick Start

### Hyperparameter Tuning

1. **Submit a Bayesian Optimization Experiment:**

```bash
kubectl apply -f katib/hyperparameter-tuning/bayesian-optimization.yaml
```

2. **Monitor the experiment:**

```bash
kubectl get experiment bayesian-optimization-experiment -n kubeflow
kubectl describe experiment bayesian-optimization-experiment -n kubeflow
```

3. **View results:**

```bash
kubectl get trials -n kubeflow
kubectl logs <trial-name> -n kubeflow
```

### Neural Architecture Search

1. **Submit a NAS experiment:**

```bash
kubectl apply -f katib/nas/nas-experiment.yaml
```

2. **Monitor progress:**

```bash
kubectl get experiment nas-experiment -n kubeflow -w
```

### Using Kubeflow Pipelines

1. **Compile the pipeline:**

```bash
python scripts/compile_pipeline.py --output automl_pipeline.yaml
```

2. **Upload to Kubeflow:**

```bash
kfp pipeline upload -p automl_pipeline.yaml
```

3. **Run the pipeline:**

```bash
kfp run submit -e <experiment-name> -r <run-name> -f automl_pipeline.yaml
```

### Deploy Model to KFServing

1. **Deploy inference service:**

```bash
kubectl apply -f serving/kfserving/tensorflow-inference-service.yaml
```

2. **Check service status:**

```bash
kubectl get inferenceservice automl-tensorflow-model -n kubeflow
```

3. **Test inference:**

```bash
# Get service URL
SERVICE_URL=$(kubectl get inferenceservice automl-tensorflow-model -n kubeflow -o jsonpath='{.status.url}')

# Make prediction
curl -X POST ${SERVICE_URL}/v1/models/automl-model:predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [[1, 2, 3, ...]]}'
```

## Next Steps

- See [Configuration Guide](CONFIGURATION.md) for advanced settings
- Check [Examples](EXAMPLES.md) for more use cases
- Read [Architecture Documentation](ARCHITECTURE.md) for system design




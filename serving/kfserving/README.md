# KFServing (KServe) Configuration

This directory contains KFServing InferenceService configurations for deploying trained models.

## Available Configurations

- `tensorflow-inference-service.yaml` - TensorFlow/Keras models
- `pytorch-inference-service.yaml` - PyTorch models
- `sklearn-inference-service.yaml` - Scikit-learn models
- `custom-predictor.yaml` - Custom model server

## Usage

### Deploy a Model

```bash
kubectl apply -f tensorflow-inference-service.yaml
```

### Check Status

```bash
kubectl get inferenceservice -n kubeflow
kubectl describe inferenceservice automl-tensorflow-model -n kubeflow
```

### Get Service URL

```bash
kubectl get inferenceservice automl-tensorflow-model -n kubeflow \
  -o jsonpath='{.status.url}'
```

### Test Inference

```bash
SERVICE_URL=$(kubectl get inferenceservice automl-tensorflow-model -n kubeflow \
  -o jsonpath='{.status.url}')

curl -X POST ${SERVICE_URL}/v1/models/automl-model:predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [[1, 2, 3, ...]]}'
```

## Customization

Edit the YAML files to customize:

- `spec.predictor`: Model framework and configuration
- `metadata.annotations`: Deployment mode and autoscaling
- `spec.predictor.resources`: Resource limits and requests
- `spec.predictor.storageUri`: Model storage location

## Model Storage

Models should be stored in a location accessible by KFServing:

- **S3**: `s3://bucket/path/to/model`
- **GCS**: `gs://bucket/path/to/model`
- **Azure Blob**: `https://account.blob.core.windows.net/container/path`
- **PVC**: `pvc://pvc-name/path/to/model`




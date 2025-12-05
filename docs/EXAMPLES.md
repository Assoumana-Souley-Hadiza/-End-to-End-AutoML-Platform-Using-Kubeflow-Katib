# Examples

This document provides practical examples for using the AutoML platform.

## Example 1: Hyperparameter Tuning with Bayesian Optimization

### Step 1: Prepare Data

```python
import pandas as pd
from sklearn.datasets import make_classification

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
df = pd.DataFrame(X)
df['target'] = y
df.to_csv('data/train.csv', index=False)
```

### Step 2: Submit Katib Experiment

```bash
kubectl apply -f katib/hyperparameter-tuning/bayesian-optimization.yaml
```

### Step 3: Monitor Experiment

```bash
# Check experiment status
kubectl get experiment bayesian-optimization-experiment -n kubeflow

# View trials
kubectl get trials -n kubeflow

# Check logs
kubectl logs <trial-name> -n kubeflow
```

## Example 2: Neural Architecture Search

### Submit NAS Experiment

```bash
kubectl apply -f katib/nas/nas-experiment.yaml
```

### Monitor Progress

```bash
kubectl get experiment nas-experiment -n kubeflow -w
```

## Example 3: End-to-End Pipeline

### Using Python SDK

```python
from pipelines.kfp import automl_pipeline
import kfp

# Compile pipeline
kfp.compiler.Compiler().compile(
    pipeline_func=automl_pipeline,
    package_path="automl_pipeline.yaml",
)

# Create client
client = kfp.Client(host="http://localhost:8080")

# Create experiment
experiment = client.create_experiment(name="automl-demo")

# Run pipeline
run = client.run_pipeline(
    experiment_id=experiment.id,
    job_name="automl-demo-run",
    pipeline_package_path="automl_pipeline.yaml",
    arguments={
        "data_path": "s3://bucket/data/train.csv",
        "algorithm": "tpe",
        "max_trials": 50,
        "parallel_trials": 5,
        "model_name": "demo-model",
    },
)
```

## Example 4: Deploy Model to KFServing

### Step 1: Save Model

Ensure your trained model is saved to a storage location accessible by KFServing:

```python
# After training
model.save("s3://models/my-model/1/")
```

### Step 2: Update InferenceService

Edit `serving/kfserving/tensorflow-inference-service.yaml`:

```yaml
spec:
  predictor:
    tensorflow:
      storageUri: "s3://models/my-model/1/"
```

### Step 3: Deploy

```bash
kubectl apply -f serving/kfserving/tensorflow-inference-service.yaml
```

### Step 4: Test Inference

```python
import requests
import json

# Get service URL
service_url = "http://automl-tensorflow-model.kubeflow.example.com"

# Make prediction
data = {
    "instances": [[1.0, 2.0, 3.0, ...]]  # Your feature vector
}

response = requests.post(
    f"{service_url}/v1/models/automl-model:predict",
    json=data
)

print(response.json())
```

## Example 5: Custom Training Script

### Create Custom Training Script

```python
# src/training/custom_train.py
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    # ... other arguments
    
    args = parser.parse_args()
    
    # Your training code
    # ...
    
    # Log metrics for Katib
    print(f"accuracy={accuracy:.4f}")
    print(f"loss={loss:.4f}")

if __name__ == "__main__":
    main()
```

### Update Katib Experiment

Edit the `trialSpec` in your experiment YAML to use your custom script:

```yaml
trialSpec:
  apiVersion: batch/v1
  kind: Job
  spec:
    template:
      spec:
        containers:
          - name: training-container
            image: your-custom-image:latest
            command:
              - python
              - /app/src/training/custom_train.py
              - --learning-rate=${trialParameters.learningRate}
              - --batch-size=${trialParameters.batchSize}
```

## Example 6: Multi-Objective Optimization

Katib supports optimizing multiple objectives. Example configuration:

```yaml
objective:
  type: maximize
  objectiveMetricName: accuracy
  additionalMetricNames:
    - latency
    - model_size
  goal: 0.95  # Target accuracy
```

## Example 7: Distributed Training

For large-scale training, configure distributed training in your experiment:

```yaml
trialSpec:
  apiVersion: kubeflow.org/v1
  kind: TFJob  # or PyTorchJob
  spec:
    tfReplicaSpecs:
      Worker:
        replicas: 4
        template:
          spec:
            containers:
              - name: tensorflow
                image: tensorflow/tensorflow:2.13.0
                # ... training configuration
```




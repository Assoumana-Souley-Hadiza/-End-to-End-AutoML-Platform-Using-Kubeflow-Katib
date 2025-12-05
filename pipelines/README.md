# Kubeflow Pipelines

This directory contains Kubeflow Pipeline definitions for the AutoML platform.

## Pipeline: `kfp/automl_pipeline.py`

End-to-end AutoML pipeline that:

1. Loads and preprocesses data
2. Creates a Katib experiment
3. Submits and monitors the experiment
4. Exports the best model to KFServing
5. Deploys the model for inference

## Usage

### Compile Pipeline

```bash
python scripts/compile_pipeline.py --output automl_pipeline.yaml
```

### Upload to Kubeflow

```bash
kfp pipeline upload -p automl_pipeline.yaml
```

### Run Pipeline

```python
import kfp

client = kfp.Client(host="http://your-kubeflow-host")

# Create experiment
experiment = client.create_experiment(name="automl-experiment")

# Run pipeline
run = client.run_pipeline(
    experiment_id=experiment.id,
    job_name="automl-run",
    pipeline_package_path="automl_pipeline.yaml",
    arguments={
        "data_path": "/data/train.csv",
        "algorithm": "bayesian-optimization",
        "max_trials": 30,
        "parallel_trials": 3,
        "model_name": "my-model",
    },
)
```

## Customization

Edit `automl_pipeline.py` to customize:

- Data preprocessing steps
- Katib experiment configuration
- Model export format
- KFServing deployment settings




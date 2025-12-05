# Scripts

Utility scripts for the AutoML platform.

## Available Scripts

### `submit_katib_experiment.py`

Submit a Katib experiment from a YAML file.

```bash
python scripts/submit_katib_experiment.py katib/hyperparameter-tuning/bayesian-optimization.yaml
```

Options:
- `--namespace`: Kubernetes namespace (default: kubeflow)

### `deploy_kfserving.py`

Deploy a model to KFServing.

```bash
python scripts/deploy_kfserving.py serving/kfserving/tensorflow-inference-service.yaml
```

Options:
- `--namespace`: Kubernetes namespace (default: kubeflow)

### `compile_pipeline.py`

Compile a Kubeflow Pipeline to YAML.

```bash
python scripts/compile_pipeline.py --output automl_pipeline.yaml
```

Options:
- `--output`: Output file path (default: automl_pipeline.yaml)

## Usage

All scripts require:
- Kubernetes cluster access (kubeconfig)
- Python dependencies installed (`pip install -r requirements.txt`)
- Appropriate permissions in the Kubernetes cluster




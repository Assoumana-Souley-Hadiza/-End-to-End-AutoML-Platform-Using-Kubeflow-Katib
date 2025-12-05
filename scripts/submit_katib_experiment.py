#!/usr/bin/env python3
"""
Script to submit Katib experiments
"""
import argparse
import yaml
import sys
from pathlib import Path
from kubernetes import client, config
from kubeflow.katib import ApiClient


def submit_experiment(experiment_file: str, namespace: str = "kubeflow"):
    """Submit a Katib experiment"""
    # Load kubeconfig
    try:
        config.load_incluster_config()
        print("Using in-cluster config")
    except:
        config.load_kube_config()
        print("Using kubeconfig")
    
    # Load experiment YAML
    with open(experiment_file, "r") as f:
        experiment_dict = yaml.safe_load(f)
    
    # Create API client
    api_client = ApiClient()
    custom_api = client.CustomObjectsApi(api_client)
    
    # Submit experiment
    try:
        response = custom_api.create_namespaced_custom_object(
            group="kubeflow.org",
            version="v1beta1",
            namespace=namespace,
            plural="experiments",
            body=experiment_dict,
        )
        experiment_name = response["metadata"]["name"]
        print(f"✓ Experiment '{experiment_name}' submitted successfully")
        print(f"  Namespace: {namespace}")
        print(f"  Status: {response.get('status', {}).get('condition', 'Pending')}")
        return experiment_name
    except Exception as e:
        print(f"✗ Error submitting experiment: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Submit Katib experiment")
    parser.add_argument(
        "experiment_file",
        type=str,
        help="Path to Katib experiment YAML file",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="kubeflow",
        help="Kubernetes namespace (default: kubeflow)",
    )
    
    args = parser.parse_args()
    
    if not Path(args.experiment_file).exists():
        print(f"✗ Experiment file not found: {args.experiment_file}")
        sys.exit(1)
    
    submit_experiment(args.experiment_file, args.namespace)


if __name__ == "__main__":
    main()




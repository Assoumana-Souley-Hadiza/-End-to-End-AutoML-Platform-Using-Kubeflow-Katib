#!/usr/bin/env python3
"""
Script to deploy model to KFServing
"""
import argparse
import yaml
import sys
from pathlib import Path
from kubernetes import client, config


def deploy_inference_service(service_file: str, namespace: str = "kubeflow"):
    """Deploy KFServing InferenceService"""
    # Load kubeconfig
    try:
        config.load_incluster_config()
        print("Using in-cluster config")
    except:
        config.load_kube_config()
        print("Using kubeconfig")
    
    # Load service YAML
    with open(service_file, "r") as f:
        service_dict = yaml.safe_load(f)
    
    # Create API client
    api_client = client.ApiClient()
    custom_api = client.CustomObjectsApi(api_client)
    
    service_name = service_dict["metadata"]["name"]
    
    # Deploy service
    try:
        # Try to create
        response = custom_api.create_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            body=service_dict,
        )
        print(f"✓ InferenceService '{service_name}' created successfully")
    except client.exceptions.ApiException as e:
        if e.status == 409:
            # Service exists, update it
            print(f"⚠ InferenceService '{service_name}' already exists, updating...")
            response = custom_api.patch_namespaced_custom_object(
                group="serving.kserve.io",
                version="v1beta1",
                namespace=namespace,
                plural="inferenceservices",
                name=service_name,
                body=service_dict,
            )
            print(f"✓ InferenceService '{service_name}' updated successfully")
        else:
            print(f"✗ Error deploying service: {e}")
            sys.exit(1)
    
    print(f"  Namespace: {namespace}")
    print(f"  Status: {response.get('status', {}).get('url', 'Pending')}")
    return service_name


def main():
    parser = argparse.ArgumentParser(description="Deploy model to KFServing")
    parser.add_argument(
        "service_file",
        type=str,
        help="Path to KFServing InferenceService YAML file",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="kubeflow",
        help="Kubernetes namespace (default: kubeflow)",
    )
    
    args = parser.parse_args()
    
    if not Path(args.service_file).exists():
        print(f"✗ Service file not found: {args.service_file}")
        sys.exit(1)
    
    deploy_inference_service(args.service_file, args.namespace)


if __name__ == "__main__":
    main()




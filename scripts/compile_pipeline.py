#!/usr/bin/env python3
"""
Script to compile Kubeflow Pipeline
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.kfp.automl_pipeline import automl_pipeline
import kfp


def compile_pipeline(output_file: str = "automl_pipeline.yaml"):
    """Compile the AutoML pipeline"""
    try:
        kfp.compiler.Compiler().compile(
            pipeline_func=automl_pipeline,
            package_path=output_file,
        )
        print(f"✓ Pipeline compiled successfully: {output_file}")
        return True
    except Exception as e:
        print(f"✗ Error compiling pipeline: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Compile Kubeflow Pipeline")
    parser.add_argument(
        "--output",
        type=str,
        default="automl_pipeline.yaml",
        help="Output file path (default: automl_pipeline.yaml)",
    )
    
    args = parser.parse_args()
    
    success = compile_pipeline(args.output)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()




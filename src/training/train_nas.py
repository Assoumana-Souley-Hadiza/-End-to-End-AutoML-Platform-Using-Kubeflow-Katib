"""
Training script for Neural Architecture Search (NAS)
"""
import argparse
import os
import json
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_architecture(architecture_str):
    """Parse architecture string from Katib"""
    # Example: "conv3x3_64,conv5x5_128,pool2x2"
    if not architecture_str:
        return None
    
    ops = architecture_str.split(",")
    parsed = []
    for op in ops:
        if op.startswith("conv"):
            parts = op.replace("conv", "").split("_")
            filter_size = int(parts[0].replace("x", ""))
            num_filters = int(parts[1]) if len(parts) > 1 else 32
            parsed.append(("conv", filter_size, num_filters))
        elif op.startswith("pool"):
            size = int(op.replace("pool", "").replace("x", ""))
            parsed.append(("pool", size))
        elif op == "identity":
            parsed.append(("identity",))
    
    return parsed


def build_nas_model(input_shape, num_classes, architecture):
    """Build model from NAS architecture"""
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))
    
    if architecture:
        for op in architecture:
            if op[0] == "conv":
                _, filter_size, num_filters = op
                model.add(layers.Conv2D(
                    num_filters,
                    (filter_size, filter_size),
                    activation="relu",
                    padding="same",
                ))
            elif op[0] == "pool":
                _, size = op
                model.add(layers.MaxPooling2D((size, size)))
            elif op[0] == "identity":
                pass  # Skip connection handled separately
    else:
        # Default architecture
        model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation="relu"))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation="relu"))
    
    if len(model.layers) > 0 and not isinstance(model.layers[-1], layers.Flatten):
        model.add(layers.Flatten())
    
    model.add(layers.Dense(num_classes if num_classes > 2 else 1,
                          activation="softmax" if num_classes > 2 else "sigmoid"))
    
    loss = "sparse_categorical_crossentropy" if num_classes > 2 else "binary_crossentropy"
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    
    return model


def load_data(data_path):
    """Load data"""
    if not os.path.exists(data_path):
        # Generate synthetic image data
        X = np.random.rand(1000, 32, 32, 3)
        y = np.random.randint(0, 10, 1000)
        return X, y
    
    df = pd.read_csv(data_path)
    # Assume data is flattened images
    # Reshape if needed
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Reshape to image format if needed
    if len(X.shape) == 2 and X.shape[1] == 3072:  # 32x32x3
        X = X.reshape(-1, 32, 32, 3)
    
    return X, y


def train_model(data_path, architecture_str=None):
    """Train NAS model"""
    X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    
    num_classes = len(np.unique(y))
    input_shape = X_train.shape[1:]
    
    # Parse architecture
    architecture = parse_architecture(architecture_str)
    
    # Build model
    model = build_nas_model(input_shape, num_classes, architecture)
    
    # Train
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=10,
        validation_split=0.2,
        verbose=1,
    )
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Calculate model size
    model_size = model.count_params()
    
    # Log metrics
    print(f"accuracy={test_accuracy:.4f}")
    print(f"loss={test_loss:.4f}")
    print(f"model_size={model_size}")
    
    # Save model
    model_path = os.getenv("MODEL_PATH", "/tmp/model")
    os.makedirs(model_path, exist_ok=True)
    model.save(os.path.join(model_path, "model.h5"))
    
    return test_accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="/data/train.csv")
    parser.add_argument("--architecture", type=str, default=None)
    
    args = parser.parse_args()
    
    train_model(
        data_path=args.data_path,
        architecture_str=args.architecture,
    )


if __name__ == "__main__":
    main()




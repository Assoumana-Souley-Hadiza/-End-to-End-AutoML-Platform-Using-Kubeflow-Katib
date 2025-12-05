"""
Training script for Population Based Training (PBT)
Supports checkpointing and model mutation
"""
import argparse
import os
import json
import logging
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_model(input_shape, num_classes, learning_rate=0.01, dropout_rate=0.3, momentum=0.9):
    """Build model for PBT"""
    model = keras.Sequential([
        layers.Dense(128, activation="relu", input_shape=(input_shape,)),
        layers.Dropout(dropout_rate),
        layers.Dense(64, activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes if num_classes > 2 else 1, 
                    activation="softmax" if num_classes > 2 else "sigmoid"),
    ])
    
    loss = "sparse_categorical_crossentropy" if num_classes > 2 else "binary_crossentropy"
    opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    
    model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
    return model


def load_data(data_path):
    """Load data"""
    if not os.path.exists(data_path):
        X = np.random.rand(1000, 20)
        y = np.random.randint(0, 2, 1000)
        return X, y
    
    df = pd.read_csv(data_path)
    if "target" in df.columns:
        X = df.drop("target", axis=1).values
        y = df["target"].values
    else:
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
    
    return X, y


def train_model(
    data_path,
    learning_rate=0.01,
    batch_size=32,
    dropout_rate=0.3,
    momentum=0.9,
    checkpoint_dir="/tmp/checkpoints",
):
    """Train model with PBT support"""
    
    # Load data
    X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train = (X_train - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)
    
    num_classes = len(np.unique(y))
    input_shape = X_train.shape[1]
    
    # Build or load model
    checkpoint_path = os.path.join(checkpoint_dir, "model.h5")
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        model = keras.models.load_model(checkpoint_path)
        # Update optimizer with new hyperparameters
        model.compile(
            optimizer=keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum),
            loss=model.loss,
            metrics=model.metrics_names,
        )
    else:
        model = build_model(input_shape, num_classes, learning_rate, dropout_rate, momentum)
    
    # Train
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=5,  # Short epochs for PBT
        validation_split=0.2,
        verbose=1,
    )
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Save checkpoint
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.save(checkpoint_path)
    
    # Log metrics
    print(f"accuracy={test_accuracy:.4f}")
    print(f"loss={test_loss:.4f}")
    
    return test_accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="/data/train.csv")
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dropout-rate", type=float, default=0.3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--checkpoint-dir", type=str, default="/tmp/checkpoints")
    
    args = parser.parse_args()
    
    train_model(
        data_path=args.data_path,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        dropout_rate=args.dropout_rate,
        momentum=args.momentum,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()




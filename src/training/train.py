"""
Training script for Katib experiments
Supports hyperparameter tuning with various algorithms
"""
import argparse
import os
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build_model(
    input_shape,
    num_classes,
    learning_rate=0.01,
    dropout_rate=0.3,
    num_layers=3,
    hidden_units=128,
    activation="relu",
    optimizer="adam",
):
    """Build a neural network model"""
    model = keras.Sequential()
    
    # Input layer
    model.add(layers.Dense(hidden_units, activation=activation, input_shape=(input_shape,)))
    model.add(layers.Dropout(dropout_rate))
    
    # Hidden layers
    for _ in range(num_layers - 1):
        model.add(layers.Dense(hidden_units, activation=activation))
        model.add(layers.Dropout(dropout_rate))
    
    # Output layer
    if num_classes == 2:
        model.add(layers.Dense(1, activation="sigmoid"))
        loss = "binary_crossentropy"
    else:
        model.add(layers.Dense(num_classes, activation="softmax"))
        loss = "sparse_categorical_crossentropy"
    
    # Compile model
    if optimizer == "adam":
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == "sgd":
        opt = keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == "rmsprop":
        opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=["accuracy"],
    )
    
    return model


def load_data(data_path):
    """Load and prepare data"""
    logger.info(f"Loading data from {data_path}")
    
    if not os.path.exists(data_path):
        # Generate synthetic data for testing
        logger.warning("Data file not found, generating synthetic data")
        X = np.random.rand(1000, 20)
        y = np.random.randint(0, 2, 1000)
        return X, y
    
    df = pd.read_csv(data_path)
    
    # Separate features and target
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
    num_epochs=10,
    dropout_rate=0.3,
    num_layers=3,
    hidden_units=128,
    activation="relu",
    optimizer="adam",
    weight_decay=0.0,
    architecture=None,
):
    """Train model with given hyperparameters"""
    
    # Load data
    X, y = load_data(data_path)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Normalize features
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train = (X_train - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)
    
    # Determine number of classes
    num_classes = len(np.unique(y))
    input_shape = X_train.shape[1]
    
    # Build model
    model = build_model(
        input_shape=input_shape,
        num_classes=num_classes,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        num_layers=num_layers,
        hidden_units=hidden_units,
        activation=activation,
        optimizer=optimizer,
    )
    
    # Train model
    logger.info("Starting training...")
    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.2,
        verbose=1,
    )
    
    # Evaluate model
    logger.info("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Predictions for additional metrics
    y_pred = model.predict(X_test, verbose=0)
    if num_classes == 2:
        y_pred_classes = (y_pred > 0.5).astype(int).flatten()
    else:
        y_pred_classes = np.argmax(y_pred, axis=1)
    
    f1 = f1_score(y_test, y_pred_classes, average="weighted")
    precision = precision_score(y_test, y_pred_classes, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred_classes, average="weighted", zero_division=0)
    
    # Log metrics for Katib
    metrics = {
        "accuracy": float(test_accuracy),
        "loss": float(test_loss),
        "f1_score": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "validation_accuracy": float(history.history.get("val_accuracy", [0])[-1]),
        "training_accuracy": float(history.history.get("accuracy", [0])[-1]),
    }
    
    # Save metrics to file for Katib
    metrics_file = os.getenv("METRICS_PATH", "/tmp/metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f)
    
    # Print metrics (Katib will parse these from stdout)
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    
    # Katib metric format
    print(f"accuracy={test_accuracy:.4f}")
    print(f"loss={test_loss:.4f}")
    print(f"f1_score={f1:.4f}")
    print(f"precision={precision:.4f}")
    print(f"recall={recall:.4f}")
    
    # Save model
    model_path = os.getenv("MODEL_PATH", "/tmp/model")
    os.makedirs(model_path, exist_ok=True)
    model.save(os.path.join(model_path, "model.h5"))
    logger.info(f"Model saved to {model_path}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train model for Katib experiment")
    parser.add_argument("--data-path", type=str, default="/data/train.csv", help="Path to training data")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--dropout-rate", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of layers")
    parser.add_argument("--hidden-units", type=int, default=128, help="Hidden units per layer")
    parser.add_argument("--activation", type=str, default="relu", help="Activation function")
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--architecture", type=str, default=None, help="Model architecture")
    
    args = parser.parse_args()
    
    # Train model
    metrics = train_model(
        data_path=args.data_path,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        dropout_rate=args.dropout_rate,
        num_layers=args.num_layers,
        hidden_units=args.hidden_units,
        activation=args.activation,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        architecture=args.architecture,
    )
    
    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()




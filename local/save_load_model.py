import os
import time
import json
import shutil
from tensorflow.keras.models import model_from_json
import tensorflow as tf


def save_model(model_conv, hyper_parameters, train_data, test_data):
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    save_dir = "model_cache"

    os.makedirs(save_dir, exist_ok=True)

    shutil.copy("./train.py", os.path.join(save_dir, "./train.py"))

    # Save model parameters
    model_conv.save(os.path.join(save_dir, "parameters.h5"))

    # Save hyperparameters, model architecture, and training history
    metadata = {
        "hyperparameters": hyper_parameters,
        "model_architecture": model_conv.to_json(),
        "train_data": train_data,
        "test_data": test_data,
        "timestamp": timestamp
    }

    with open(os.path.join(save_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Model and metadata saved in {save_dir}")

    return save_dir


def load_model(cache_dir):
  
    # Load model metadata
    with open(os.path.join(cache_dir, "model_metadata.json"), "r") as f:
        metadata = json.load(f)
    
    # Reconstruct model architecture
    model_architecture = metadata["model_architecture"]
    model = model_from_json(model_architecture)
    
    # Load saved model parameters
    model.load_weights(os.path.join(cache_dir, "parameters.h5"))
    
    # Compile the model with loaded hyperparameters
    hyper_parameters = metadata["hyperparameters"]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hyper_parameters["learning_rate"]),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model loaded successfully.")
    return model

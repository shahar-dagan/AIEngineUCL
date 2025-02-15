import os
import time
import json
import shutil
from tensorflow.keras.models import model_from_json
import tensorflow as tf




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


def load_and_test(cache_dir = "./model_cache"):
    model = load_model(cache_dir)
    test_data = test_model(model)
    return test_data

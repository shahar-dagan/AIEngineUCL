import os
import time
import json
import shutil


def save_model(model_conv, hyper_parameters, performance_data):
    # Create folder with timestamp
    # timestamp = time.strftime("%Y%m%d-%H%M%S")
    # save_dir = f"model_{timestamp}"
    save_dir = "model_cache"

    os.makedirs(save_dir, exist_ok=True)

    shutil.copy("./train.py", os.path.join(save_dir, "./train.py"))

    # Save model parameters
    model_conv.save(os.path.join(save_dir, "parameters.h5"))

    # Save hyperparameters, model architecture, and training history
    metadata = {
        "hyperparameters": hyper_parameters,
        "model_architecture": model_conv.to_json(),
        "performance": performance_data
    }

    with open(os.path.join(save_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Model and metadata saved in {save_dir}")

    return save_dir
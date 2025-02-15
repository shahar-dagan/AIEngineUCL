import os
import time
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam


# Load dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = (train_images - 127.5) / 127.7
test_images = (test_images - 127.5) / 127.7
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


def build_model(hyper_parameters):
    # Build model
    model_conv = Sequential([
        Input((28, 28, 1)),
        Conv2D(hyper_parameters["conv_filters"], kernel_size=hyper_parameters["kernel_size"], activation=hyper_parameters["activation"]),
        Flatten(),
        Dense(hyper_parameters["dense_layer_neurons"], activation=hyper_parameters["activation"]),
        Dense(10, activation='softmax'),
    ])

        
    model_conv.compile(optimizer=Adam(learning_rate=hyper_parameters["learning_rate"]),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model_conv

def train_model(hyper_parameters, model_conv):

    # Train model
    history = model_conv.fit(train_images, train_labels, epochs=hyper_parameters["epochs"], batch_size=hyper_parameters["batch_size"], validation_data=(test_images, test_labels))

    test_loss, test_accuracy = model_conv.evaluate(test_images, test_labels)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    return {
        "train_loss": history.history["loss"],
        "train_accuracy": history.history["accuracy"],
        "val_loss": history.history["val_loss"],
        "val_accuracy": history.history["val_accuracy"],
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    }


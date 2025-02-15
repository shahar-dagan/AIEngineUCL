from train import build_model, train_model
from save_load_model import save_model




def example_create_store():
    hyper_parameters = dict(
        learning_rate=0.01,
        batch_size=20,
        epochs=1,
        kernel_size=(5, 5),
        conv_filters=20,
        dense_layer_neurons=20,
        activation="swish",
    )

    model_conv = build_model(hyper_parameters)
    performance_data = train_model(hyper_parameters, model_conv)

    save_dir = save_model(model_conv, hyper_parameters, performance_data)

if __name__ == "__main__":
    example_create_store()
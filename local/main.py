from train import build_model, train_model, test_model
from save_load_model import save_model, load_model



def train_and_save(hyper_parameters):

    model_conv = build_model(hyper_parameters)
    train_data = train_model(hyper_parameters, model_conv)
    test_data = test_model(model_conv)


    save_dir = save_model(model_conv, hyper_parameters, train_data, test_data)


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
    train_and_save(hyper_parameters)



def load_and_test(cache_dir = "./model_cache"):
    model = load_model(cache_dir)
    test_data = test_model(model)
    return test_data



if __name__ == "__main__":
    print("TRAINING TESTING AND SAVING MODEL")
    example_create_store()
    print("LOADING MODEL FROM FILE AND TESTING")
    load_and_test()
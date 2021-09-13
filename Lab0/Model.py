class Model:
    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size

    def train(self, x_train, y_train, epochs=100000, use_mini_batch=True, mini_batch_size=100):
        return self

    def predict(self, x):
        pass
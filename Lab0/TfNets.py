from Model import Model
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


class TfModel(Model):
    def __init__(self, input_size: int, output_size: int):
        super(TfModel, self).__init__(input_size, output_size)
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(output_size, activation='softmax')
        ])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                           loss=tf.keras.losses.CategoricalCrossentropy(),
                           metrics=['accuracy'])

    def train(self, x_train, y_train, epochs=1, use_mini_batch=False, mbs=100) -> Model:
        self.model.fit(x_train, y_train, epochs=epochs, verbose=1, batch_size=mbs if use_mini_batch else None)
        return self

    def predict(self, x):
        return self.model.predict(x)


class CNNModel(Model):
    def __init__(self, input_size: int, output_size: int):
        super(CNNModel, self).__init__(input_size, output_size)
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(64, 3, activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(128, 3, activation='relu'),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(256, 3, activation='relu'),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(output_size, activation='softmax')
        ])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                           loss=tf.keras.losses.CategoricalCrossentropy(),
                           metrics=['accuracy'])

    def train(self, x_train, y_train, epochs=10, use_mini_batch=False, mbs=100) -> Model:
        self.model.fit(x_train, y_train, epochs=epochs, verbose=1, batch_size=mbs if use_mini_batch else None)
        return self

    def predict(self, x):
        return self.model.predict(x)
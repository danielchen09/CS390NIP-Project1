from Model import Model
import tensorflow as tf

class TfModel(Model):
    def __init__(self, input_size: int, output_size: int):
        super(TfModel, self).__init__(input_size, output_size)
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10)
        ])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

    def train(self, x_train, y_train, epochs=1, use_mini_batch=False, mbs=100) -> Model:
        self.model.fit(x_train, y_train, epochs=epochs, verbose=1, batch_size=mbs if use_mini_batch else None)
        return self

    def predict(self, x):
        return self.model.predict(x)

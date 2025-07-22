from dataclasses import dataclass
import tensorflow as tf
from keras.layers import (
    Conv2D,
    Conv2DTranspose,
    Dense,
    MaxPool2D,
    BatchNormalization,
    Dropout,
    concatenate,
)
import keras.backend as keras_backend


SIGMOID = "sigmoid"
BINARY_IOU = "BinaryIoU"
LOGS_PATH = "logs/"


def load_model(model_path: str, custom_objects: dict) -> tf.keras.Model | None:
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    return model


@dataclass
class Model:
    input_shape: tuple
    output_shape: tuple
    activation: str
    kernel_init: str
    loss: str
    optimizer: str

    @staticmethod
    def _dice_coef(y_true, y_pred, smooth=100):
        y_true_f = keras_backend.flatten(y_true)
        y_pred_f = keras_backend.flatten(y_pred)
        intersection = keras_backend.sum(y_true_f * y_pred_f)
        dice = (2.0 * intersection + smooth) / (keras_backend.sum(y_true_f) + keras_backend.sum(y_pred_f) + smooth)
        return dice

    def _dice_coef_loss(self, y_true, y_pred):
        return 1 - self._dice_coef(y_true, y_pred)

    def _get_input_layer(self):
        return tf.keras.layers.Input(self.input_shape)

    def _get_output_layer(self):
        if self.output_shape == self.input_shape:
            output_layer = Conv2D(1, (1, 1), activation=SIGMOID)
        else:
            output_layer = Dense(1)

        return output_layer


@dataclass
class UNet(Model):
    def build_model(self):
        input_layer = self._get_input_layer()
        output_layer = self._get_output_layer()

        encoding1, pool1 = self._get_encoding(input_layer, 16, (3, 3), (2, 2))
        encoding2, pool2 = self._get_encoding(pool1, 32, (3, 3), (2, 2))
        encoding3, pool3 = self._get_encoding(pool2, 64, (3, 3), (2, 2))
        encoding4, pool4 = self._get_encoding(pool3, 128, (3, 3), (2, 2))
        encoding5, _ = self._get_encoding(pool4, 256, (3, 3), (2, 2))

        decoding1 = self._get_decoding(encoding5, encoding4, 128, (2, 2), (3, 3))
        decoding2 = self._get_decoding(decoding1, encoding3, 64, (2, 2), (3, 3))
        decoding3 = self._get_decoding(decoding2, encoding2, 32, (2, 2), (3, 3))
        decoding4 = self._get_decoding(decoding3, encoding1, 16, (2, 2), (3, 3))

        outputs = output_layer(decoding4)

        self.model = tf.keras.Model(inputs=[input_layer], outputs=[outputs])
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=[
                BINARY_IOU,
                self._dice_coef,
            ],
        )

    def fit(self, train_dataset, val_dataset, epochs: int, metric: str, file_name: str):
        checkpointer = tf.keras.callbacks.ModelCheckpoint(file_name, verbose=1, metric=metric, save_best_only=True)
        callbacks = [
            checkpointer,
            tf.keras.callbacks.EarlyStopping(patience=100, monitor=metric),
            tf.keras.callbacks.TensorBoard(log_dir=f"{LOGS_PATH}{file_name}"),
        ]

        self.model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=callbacks)

    def _get_encoding(self, input_layer, filters: int, kernel_size: tuple, pool_size: tuple):
        conv_enc = Conv2D(
            filters, kernel_size, activation=self.activation, kernel_initializer=self.kernel_init, padding="same"
        )(input_layer)
        conv_enc = BatchNormalization()(conv_enc)
        conv_enc = Dropout(0.1)(conv_enc)
        conv_enc = Conv2D(
            filters, kernel_size, activation=self.activation, kernel_initializer=self.kernel_init, padding="same"
        )(conv_enc)
        conv_enc = BatchNormalization()(conv_enc)
        max_pool = MaxPool2D(pool_size)(conv_enc)

        return conv_enc, max_pool

    def _get_decoding(
        self,
        input_layer,
        concatenate_layer: tf.keras.Sequential,
        filters: int,
        kernel_size_transpose: tuple,
        kernel_size: tuple,
    ):
        conv_transpose = Conv2DTranspose(
            filters, kernel_size_transpose, activation=self.activation, strides=(2, 2), padding="same"
        )(input_layer)
        concat = concatenate([conv_transpose, concatenate_layer])
        conv_dec = Conv2D(
            filters, kernel_size, activation=self.activation, kernel_initializer=self.kernel_init, padding="same"
        )(concat)
        conv_dec = BatchNormalization()(conv_dec)
        conv_dec = Dropout(0.1)(conv_dec)
        conv_dec = Conv2D(
            filters, kernel_size, activation=self.activation, kernel_initializer=self.kernel_init, padding="same"
        )(conv_dec)
        conv_dec = BatchNormalization()(conv_dec)

        return conv_dec

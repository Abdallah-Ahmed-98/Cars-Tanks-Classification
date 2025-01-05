import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig
                                                

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig, random_seed: int = 42):
        self.config = config
        self.random_seed = random_seed
        self._set_reproducibility()

    def _set_reproducibility(self):
        """Set global random seed for reproducibility."""
        tf.random.set_seed(self.random_seed)

    def get_base_model(self):
        self.model = tf.keras.applications.ResNet50V2(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False

        flatten = tf.keras.layers.Flatten()(model.output)
        dropout = tf.keras.layers.Dropout(rate=0.3)(flatten)
        dense = tf.keras.layers.Dense(128, activation='relu')(dropout)
        dropout = tf.keras.layers.Dropout(rate=0.25)(dense)
        prediction = tf.keras.layers.Dense(units=classes, activation="softmax")(dropout)

        full_model = tf.keras.models.Model(inputs=model.input, outputs=prediction)

        full_model.compile(
            optimizer=tf.keras.optimizers.Adamax(learning_rate=learning_rate, decay=1e-5),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        full_model.summary()
        return full_model

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)



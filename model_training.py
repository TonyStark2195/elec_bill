import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class ClassifierModel:

    def __init__(self, img_dir="landloss_model_training_images", image_size=(500, 500), batch_size=32, num_classes=2):
        self.train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            img_dir,
            validation_split=0.2,
            subset="training",
            seed=1337,
            image_size=image_size,
            batch_size=batch_size,
            label_mode='categorical'
        )
        self.val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            img_dir,
            validation_split=0.2,
            subset="validation",
            seed=1337,
            image_size=image_size,
            batch_size=batch_size,
            label_mode='categorical'
        )

        self.train_ds = self.train_ds.prefetch(buffer_size=32)
        self.val_ds = self.val_ds.prefetch(buffer_size=32)

        data_augmentation = keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.1),
            ]
        )

        input_shape = image_size + (3,)

        inputs = keras.Input(shape=input_shape)
        # Image augmentation block
        x = data_augmentation(inputs)
        # Entry block
        x = layers.Rescaling(1.0 / 255)(x)
        x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.Conv2D(64, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        previous_block_activation = x  # Set aside residual

        for size in [128, 256, 512, 728]:
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(size, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(size, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = layers.Conv2D(size, 1, strides=2, padding="same")(
                previous_block_activation
            )
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        x = layers.SeparableConv2D(1024, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.GlobalAveragePooling2D()(x)
        if num_classes == 2:
            activation = "sigmoid"
            units = 1
        else:
            activation = "softmax"
            units = num_classes

        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(units, activation=activation)(x)

        self.model = keras.Model(inputs, outputs)

    def train(self, epochs=50, chk_path="checkpoint/", loss="categorical_crossentropy", metrics=["accuracy"]):

        callbacks = [
            keras.callbacks.ModelCheckpoint(chk_path + "save_at_{epoch}.h5"),
        ]
        self.model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss=loss,
            metrics=metrics,
        )
        self.model.fit(
            self.train_ds, epochs=epochs, callbacks=callbacks, validation_data=self.val_ds,
        )

    def save(self):
        pass

    def plot(self):
        keras.utils.plot_model(self.model, show_shapes=True)

    def stats(self):
        pass


if __name__ == '__main__':
    model_obj = ClassifierModel(num_classes=3)
    model_obj.train()

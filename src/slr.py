import tensorflow as tf
from preprocess import get_data

# Sign Language Recognition


def create_model():
    num_classes = 29
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])
    return model


def train(model, train_generator, validation_generator, epochs, load_weights=False):
    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VALID = validation_generator.n // validation_generator.batch_size

    checkpoint_path = "checkpoints/cp.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    if load_weights:
        model.load_weights(checkpoint_path)
    if epochs > 0:
        model.fit(train_generator,
                  steps_per_epoch=STEP_SIZE_TRAIN,
                  epochs=epochs,
                  validation_data=validation_generator,
                  validation_steps=STEP_SIZE_VALID,
                  callbacks=[cp_callback])


def test(model, validation_generator):
    STEP_SIZE_VALID = validation_generator.n // validation_generator.batch_size
    model.evaluate(validation_generator, steps=STEP_SIZE_VALID)


def main():
    model = create_model()
    train_generator, validation_generator = get_data()
    train(model, train_generator, validation_generator,
          epochs=0, load_weights=True)
    test(model, validation_generator)

    tf.saved_model.save(model, "./saved_model")


if __name__ == '__main__':
    main()

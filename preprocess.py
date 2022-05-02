from keras.preprocessing.image import ImageDataGenerator

def get_data():
    base = 'data/'
    prefix = 'asl_alphabet_'
    batch_size = 100
    pic_size = 48

    datagen_train = ImageDataGenerator()
    datagen_test = ImageDataGenerator()

    train_generator = datagen_train.flow_from_directory(base + prefix + "train",
                                                        target_size=(pic_size,pic_size),
                                                        color_mode="grayscale",
                                                        batch_size=batch_size,
                                                        class_mode='categorical',
                                                        shuffle=True)

    validation_generator = datagen_test.flow_from_directory(base + prefix + "test",
                                                        target_size=(pic_size,pic_size),
                                                        color_mode="grayscale",
                                                        batch_size=batch_size,
                                                        class_mode='categorical',
                                                        shuffle=False)

    return train_generator, validation_generator
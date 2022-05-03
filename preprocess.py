from keras.preprocessing.image import ImageDataGenerator


def get_data():
    batch_size = 100
    pic_size = 48

    image_flow = ImageDataGenerator(validation_split=0.1)

    train_generator = image_flow.flow_from_directory("data/asl_alphabet_train/asl_alphabet_train/",
                                                     target_size=(
                                                         pic_size, pic_size),
                                                     color_mode="grayscale",
                                                     batch_size=batch_size,
                                                     class_mode='sparse',
                                                     subset="training")

    validation_generator = image_flow.flow_from_directory("data/asl_alphabet_train/asl_alphabet_train/",
                                                          target_size=(
                                                              pic_size, pic_size),
                                                          color_mode="grayscale",
                                                          batch_size=batch_size,
                                                          class_mode='sparse',
                                                          subset="validation")

    return train_generator, validation_generator

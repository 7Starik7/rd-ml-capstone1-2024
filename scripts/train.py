import tensorflow as tf

# noinspection PyUnresolvedReferences
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.src.optimizers import Adam

train_dir = '../dataset/extracted_images/train'
val_dir = '../dataset/extracted_images/validation'
test_dir = '../dataset/extracted_images/test'


def create_generators(train_dir, val_dir, test_dir):
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=50,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       shear_range=0.1,
                                       zoom_range=0.1,
                                       horizontal_flip=True,
                                       fill_mode='nearest')

    val_test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=(32, 32),
                                                        batch_size=32,
                                                        shuffle=True,
                                                        class_mode='sparse')

    val_generator = val_test_datagen.flow_from_directory(val_dir,
                                                         target_size=(32, 32),
                                                         batch_size=32,
                                                         class_mode='sparse',
                                                         shuffle=False)

    test_generator = val_test_datagen.flow_from_directory(test_dir,
                                                          target_size=(32, 32),
                                                          batch_size=32,
                                                          class_mode='sparse',
                                                          shuffle=False)
    return train_generator, val_generator, test_generator


def make_model(learning_rate=0.0001, drop_rate=0.5):
    model = Sequential()
    model.add(Input(shape=(32, 32, 3)))

    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(units=128, activation='relu'))

    model.add(Dropout(drop_rate))

    model.add(Dense(units=43, activation='softmax'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model(model, train, epochs, test):
    return model.fit(train, steps_per_epoch=train.batch_size, epochs=epochs, validation_data=test,
                     validation_steps=test.batch_size)


def val_model(model, train_generator, val_generator, test_generator):
    train_loss, train_accuracy = model.evaluate(train_generator, steps=train_generator.batch_size)
    print(f"Training Loss: {train_loss}")
    print(f"Training Accuracy: {train_accuracy}")
    print()
    val_loss, val_accuracy = model.evaluate(val_generator, steps=val_generator.batch_size)
    print(f"Validation Loss: {val_loss}")
    print(f"Validation Accuracy: {val_accuracy}")
    print()
    test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.batch_size)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")
    print()


def save_model(model):
    model.save('../model.h5')
    model.save('../model.keras')
    tf.saved_model.save(model, '../traffic-sign-recognition-model')


if __name__ == '__main__':
    train_generator, val_generator, test_generator = create_generators(train_dir, val_dir, test_dir)
    model = make_model(learning_rate=0.001, drop_rate=0.2)
    history = train_model(model, train_generator, 400, val_generator)
    val_model(model, train_generator, val_generator, test_generator)
    save_model(model)

from keras.datasets import mnist
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import extra_keras_datasets as ekd


class UtilityAPIs:

    def __init__(self):
        pass

    def get_mnist_data(self):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        X_train /= 255
        X_test /= 255

        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

        return X_train, y_train, X_test, y_test

    def get_emnist_data(self):
        (X_train, y_train), (X_test, y_test) = ekd.emnist.load_data(type='digits')

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        X_train /= 255
        X_test /= 255

        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

        return X_train, y_train, X_test, y_test

    def augmentation(self):
        aug = ImageDataGenerator(
            rotation_range=15,
            zoom_range=0.2,
            width_shift_range=0.2,
            height_shift_range=0.2
        )
        return aug

    def training_augmentation(self):
        aug = ImageDataGenerator(
            rotation_range=15,
            zoom_range=0.2,
            width_shift_range=0.2,
            height_shift_range=0.2,
            validation_split=0.2
        )
        return aug

    def testing_augmentation(self):
        aug = ImageDataGenerator(
            rotation_range=15,
            zoom_range=0.2,
            width_shift_range=0.2,
            height_shift_range=0.2
        )
        return aug

    def to_cnn_shape(self, data):
        data = data.reshape(data.shape[0], 28, 28, 1)
        return data

    def to_mlp_shape(self, data):
        data = data.reshape(data.shape[0], 784)
        return data

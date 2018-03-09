from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Reshape
from keras.layers.convolutional import Convolution2D as Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import UpSampling2D
from keras.layers.normalization import BatchNormalization


def get_generator():
    """
    Create a model that takes in a matrix of
    random values as input and outputs images
    :return: Generator model
    """
    model = Sequential()
    model.add(Dense(input_dim=100, units=1024))
    model.add(Activation('tanh'))
    model.add(Dense(128 * 8 * 8))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((8, 8, 128), input_shape=(128 * 8 * 8,)))  # 8x8 image
    model.add(UpSampling2D(size=(2, 2)))  # 16x16 image
    model.add(Conv2D(128, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))  # 32x32 image
    model.add(Conv2D(128, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))  # 64x64 image
    model.add(Conv2D(3, (5, 5), padding='same'))
    model.add(Activation('tanh'))

    return model


def get_discriminator():
    """
    Create a model that takes in an image
    and outputs whether it contains our desired subject
    :return: Discriminator model
    """
    model = Sequential()
    model.add(
        Conv2D(
            64,
            (5, 5),
            padding='same',
            input_shape=(64, 64, 3)
        )
    )
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


def get_generative_adversarial_network(generator, discriminator):
    """
    A network composed of a generator and discriminator network

    The flow of data is as follows:
        Input -> Generator -> Discriminator -> Output
    """
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model

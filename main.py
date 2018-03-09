import argparse
import numpy
from keras.optimizers import Adam
from PIL import Image
import os

from models import get_generator, get_discriminator, get_generative_adversarial_network
from utils import load_data
import constants


def train(batch_size, learning_rate, beta_1, epochs, data_path):
    """
    Train the generator and discriminator
    :param batch_size: Batch size
    :param learning_rate: Learning rate
    :param beta_1: beta_1 for Adam optimizer
    :param epochs: Number of epochs
    :param data_path: Path of directory
    """
    input_data = load_data(data_path, constants.IMAGE_SIZE)

    # normalize data between (-1, 1) which is the same output scale as tanh
    input_data = (input_data.astype(numpy.float32) - 127.5) / 127.5

    # Get generator, discriminator and composed network
    generator = get_generator()
    discriminator = get_discriminator()
    generative_adversarial_network = get_generative_adversarial_network(generator, discriminator)

    generator_optimizer = Adam(lr=learning_rate, beta_1=beta_1)
    discriminator_optimizer = Adam(lr=learning_rate, beta_1=beta_1)

    # Compile all networks
    generator.compile(loss='binary_crossentropy', optimizer=generator_optimizer)
    generative_adversarial_network.compile(loss='binary_crossentropy', optimizer=generator_optimizer)
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=discriminator_optimizer)

    for epoch in range(epochs):
        print("Epoch:%d" % epoch)
        for batch_number in range(int(input_data.shape[0] / batch_size)):
            input_batch = input_data[batch_number * batch_size: (batch_number + 1) * batch_size]

            noise = numpy.random.uniform(-1, 1, size=(batch_size, 100))
            generated_images = generator.predict(noise, verbose=0)

            input_batch = numpy.concatenate((input_batch, generated_images))

            output_batch = [1] * batch_size + [0] * batch_size

            # train the discriminator to reject the generated images
            discriminator_loss = discriminator.train_on_batch(input_batch, output_batch)

            noise = numpy.random.uniform(-1, 1, (batch_size, 100))

            # we disable training the discriminator when training the generator since the
            # discriminator is being used to judge, we don't want to train it on false data
            discriminator.trainable = False

            # train the generator with the objective of getting the generated images approved
            generator_loss = generative_adversarial_network.train_on_batch(noise, [1] * batch_size)
            discriminator.trainable = True

            print("Batch=%d, Discriminator Loss=%f" % (batch_number, discriminator_loss))
            print("Batch=%d, Generator Loss=%f" % (batch_number, generator_loss))

        if epoch % 10 == 9:
            generator.save_weights('generator_weights.h5', True)
            discriminator.save_weights('discriminator_weights.h5', True)


def generate(batch_size, learning_rate, beta_1):
    """
    Generate images using the trained generator
    :param batch_size: Batch size - Number of images to generate
    :param learning_rate: Learning rate
    :param beta_1: beta_1 for Adam optimizer
    """
    generator = get_generator()
    generator.compile(loss='binary_crossentropy', optimizer=Adam(lr=learning_rate, beta_1=beta_1))
    generator.load_weights('generator_weights.h5')

    noise = numpy.random.uniform(-1, 1, (batch_size, 100))
    generated_images = generator.predict(noise, verbose=1)

    if not os.path.exists("generated_images"):
        os.makedirs("generated_images")

    for i in range(batch_size):
        image = generated_images[i]
        image = image * 127.5 + 127.5

        # Save generated image
        Image.fromarray(image.astype(numpy.uint8)).save("generated_images/image-{}.png".format(i))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse values passed through command line')
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--batch_size', default=constants.BATCH_SIZE, type=int)
    parser.add_argument('--learning_rate', default=constants.LEARNING_RATE, type=float)
    parser.add_argument('--beta_1', default=constants.BETA_1, type=float)
    parser.add_argument('--epochs', default=constants.EPOCHS, type=int)
    parser.add_argument('--images_path', default='', type=str)

    args = parser.parse_args()

    if args.mode == 'train':
        train(args.batch_size, args.learning_rate, args.beta_1, args.epochs, args.images_path)
    elif args.mode == 'generate':
        generate(args.batch_size, args.learning_rate, args.beta_1)

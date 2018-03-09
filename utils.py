import glob
import numpy
from scipy import misc
from skimage.transform import resize


def load_data(data_path, image_size):
    """
    Load images into a 4D numpy ndarray
    :param data_path: Directory path
    :return: 4D ndarray
    """
    input_data = []

    for index, image_path in enumerate(glob.glob(data_path + '/*')):
        print("Image Path:", image_path)
        image = misc.imread(image_path)
        try:
            image = resize(image, image_size)
        except Exception as e:
            print("Error:", e)
            continue

        if len(image.shape) is not 3:
            print("Issue with shape of this image")
        else:
            if image is not None:
                if image.shape[2] is 3:
                    input_data.append(image)
                else:
                    print("Number of channels is not three for this image:", image.shape)

    # Convert python array to a 4D ndarray
    return numpy.array(input_data)

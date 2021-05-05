import cv2
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from PIL import Image
import PIL


def load_image(img):
    load_img(img)


# convert the loaded image to a 3d numpy array and expand samples
def convert_to_3d_numpy_array(loaded_img):
    data = img_to_array(loaded_img)
    sample_data = expand_dims(data, 0)
    return sample_data


# prepare iterator using .flow() to generate random batches of transferred images
def prepare_iterator(directory, image_prefix, saved_format):
    iterator = data_generator.flow(
        samples, batch_size=1,
        save_to_dir=directory,
        save_prefix=image_prefix,
        save_format=saved_format)
    return iterator


sample_image = "sample_dublin.png"
image = load_img(sample_image)

samples = convert_to_3d_numpy_array(image)

# image augmentation generator
data_generator = ImageDataGenerator(zoom_range=[0.5, 1.0], rotation_range=40,
                                    width_shift_range=0.2, height_shift_range=0.2,
                                    shear_range=0.2, horizontal_flip=True,
                                    rescale=1/255.)

prepared_iterator = prepare_iterator("augmented_images", "sample_dublin", "png")

# generate augmented images and save to augmented_images folder
for i in range(2):
    batch = prepared_iterator.next()
    image = batch[0].astype('uint8')

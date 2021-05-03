import cv2
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from PIL import Image
import PIL

# sample image
img = load_img("dublin.png")
# convert to a 3d numpy array and expand samples
data = img_to_array(img)
samples = expand_dims(data, 0)
# image augmentation generator
data_generator = ImageDataGenerator(zoom_range=[0.5, 1.0], rescale=1/255.)
# prepare iterator
iterator = data_generator.flow(samples, batch_size=1)
# generate augmented images and save to augmented_images folder
for i in range(2):
    batch = iterator.next()
    image = batch[0].astype('uint8')
    image_id = str(i+1)
    path_to_store = "augmented_images/dublin_" + image_id + ".jpg"
    cv2.imwrite(path_to_store, image)



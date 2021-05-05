import numpy as np
from matplotlib import pyplot
from PIL import Image
from skimage import util

img = Image.open("dublin.png")
img = np.array(img)

# add various noise
noise_gs_img = util.random_noise(img, mode="gaussian")
noise_salt_img = util.random_noise(img, mode="salt")
noise_pepper_img = util.random_noise(img, mode="pepper")
noise_sp_img = util.random_noise(img, mode="s&p")
noise_speckle_img = util.random_noise(img, mode="speckle")

# temporarily visualize on a plot
pyplot.subplot(2,3,1), pyplot.title("dublin original")
pyplot.imshow(img)
pyplot.subplot(2,3,2),pyplot.title("gaussian")
pyplot.imshow(noise_gs_img)
pyplot.subplot(2,3,3), pyplot.title("salt")
pyplot.imshow(noise_salt_img)
pyplot.subplot(2,3,4), pyplot.title("pepper")
pyplot.imshow(noise_pepper_img)
pyplot.subplot(2,3,5),pyplot.title("salt & pepper")
pyplot.imshow(noise_sp_img)
pyplot.subplot(2,3,6), pyplot.title("speckle")
pyplot.imshow(noise_speckle_img)
pyplot.show()

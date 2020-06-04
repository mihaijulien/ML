import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2

image_color = mpimg.imread('images/image_lane_c.jpg')
plt.imshow(image_color)
plt.show()

image_copy = np.copy(image_color)

image_copy[ (image_copy[:,:,0] < 210) | (image_copy[:,:,1] < 210) | (image_copy[:,:,2] < 210) ] = 0  # any value that is not white colour

plt.imshow(image_copy, cmap = 'gray')
plt.show()
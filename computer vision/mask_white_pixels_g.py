import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2

image_color = mpimg.imread('images/image_lane_c.jpg')
plt.imshow(image_color)
plt.show()
print(image_color.shape)

image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
plt.imshow(image_gray, cmap = 'gray')
plt.show()

image_copy = np.copy(image_gray)

image_copy[ (image_copy[:,:] < 230) ] = 0  # any value that is not white colour

plt.imshow(image_copy, cmap = 'gray')
plt.show()
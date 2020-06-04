import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2

image = cv2.imread('images/self_driving_car.jpg')

print('Height = ', int(image.shape[0]), 'pixels')
print('Width = ', int(image.shape[1]), 'pixels')

cv2.imshow('Self Driving Car', image)
cv2.waitKey()
cv2.destroyAllWindows()

# The order of B and R is reversed
plt.imshow(image)
plt.show()

gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Self Driving Car in Grayscale!', gray_img)
cv2.waitKey()
cv2.destroyAllWindows()

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow('HSV Image', hsv_image)
cv2.waitKey()
cv2.destroyAllWindows()

plt.imshow(hsv_image[:, :, 0])
plt.title('Hue channel')

plt.imshow(hsv_image[:, :, 1])
plt.title('Saturation channel')

plt.imshow(hsv_image[:, :, 2])
plt.title('Value channel')

# Splitting BGR channels
B, G, R = cv2.split(image)
print(B.shape)

# Let's view the R, B, G channels
cv2.imshow("Blue Channel!", B) # it comes up as a grayscale image
cv2.waitKey(0)
cv2.destroyAllWindows()

# Let's try to create our own 3D image out of the blue channel!
zeros = np.zeros(image.shape[:2], dtype = "uint8")

cv2.imshow("Blue Channel!", cv2.merge([B, zeros, zeros]))
cv2.waitKey(0)
cv2.destroyAllWindows()

# Let's merge RGB channels to create our original image
image_merged = cv2.merge([B, G, R]) 
cv2.imshow("Merged Image!", image_merged) 
cv2.waitKey(0)
cv2.destroyAllWindows()

# Let's merge our original image while adding more green! 
image_merged = cv2.merge([B, G+100, R]) 
cv2.imshow("Merged Image with some added green!", image_merged) 
cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2
import numpy as np

image = cv2.imread('images/test_image.jpg')
lane = np.copy(image)

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(gray, 50, 150)
    return canny

canny_img = canny(lane)

cv2.imshow("Result", canny_img)
cv2.waitKey()
cv2.destroyAllWindows()
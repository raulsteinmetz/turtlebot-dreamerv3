import cv2
import numpy as np

def get_hsv(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv_image = param['hsv']
        print("HSV at ({}, {}): {}".format(x, y, hsv_image[y, x]))

im_path = 'images/WIN_20250129_14_12_40_Pro.jpg'
image = cv2.imread(im_path)
if image is None:
    print("Error: Image not found.")
    exit()
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

cv2.namedWindow('Image')
cv2.setMouseCallback('Image', get_hsv, param={'hsv': hsv})

while True:
    cv2.imshow('Image', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

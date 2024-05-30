import cv2  
import numpy as np  

img = cv2.imread('peppers.png')

# Create a vertically symmetric image. flip(img, 0) function creates a vertically symmetric image by flipping the input image (img) around the horizontal axis. The second argument 0 indicates the axis of flipping, where 0 stands for vertical flipping.
symmetric_vertical = cv2.flip(img, 0)

# Create a horizontally symmetric image. flip(img, 1) function creates a horizontally symmetric image by flipping the input image (img) around the vertical axis. The second argument 1 indicates the axis of flipping, where 1 stands for horizontal flipping.
symmetric_horizontal = cv2.flip(img, 1)

cv2.imshow('original', img)
cv2.imshow('Symmetric Vertical', symmetric_vertical)
cv2.imshow('Symmetric Horizontal', symmetric_horizontal)

cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2  # Provides image processing functionality using the OpenCV library
import numpy as np  # Provides scientific computing functionality using the NumPy library

#Imread function loads an image from a file.
image = cv2.imread('peppers.png')

# Resize the image to a specific size.
image_resized = cv2.resize(image, (256, 256))

# Convert the image to grayscale. BGR2GRAYSCALE is used because OpenCV reads images as BGR and not RGB
gray_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

# Apply a binary threshold to the image.
# If pixels are greater than 127, it converts to 255, if less than 0. Thus, the image consists of only two different values.
ret, binary_threshold = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# Display the resized image in a window.
cv2.imshow('Resized Image', image_resized)
# Display the grayscale image in a window.
cv2.imshow('Gray Image', gray_image)
# Display the thresholded image in a window.
cv2.imshow('Threshold', binary_threshold)

# This function is often used to keep the OpenCV window open until the user decides to close it.
cv2.waitKey(0)
#  It's usually called after cv2.waitKey() to ensure that all windows are closed properly when the program ends or when it's no longer needed.
cv2.destroyAllWindows()

import cv2
import numpy as np

img = cv2.imread('peppers.png')

# shape property returns the dimensions of an array.
# Returns the height, width, and number of color channels of the loaded image.
height, width, _ = img.shape  

# Calculates the quarter values of the height and width of the image.
quarter_height = height // 2
quarter_width = width // 2

# Indexing operator selects a specific region of an array.
top_left = img[:quarter_height, :quarter_width]
top_right = img[:quarter_height, quarter_width:]
bottom_left = img[quarter_height:, :quarter_width]

# zeros_like function creates an array similar to the specified array but sets all elements to zero. Creates empty images for the blue, green, and red channels.
blue_channel = np.zeros_like(top_left)
green_channel = np.zeros_like(top_left)
red_channel = np.zeros_like(top_left)

# Indexing operator selects a specific region of an array and selects the blue, green, and red channels from the respective image regions.
blue_channel[:, :, 0] = img[quarter_height:, quarter_width:, 0]  # 0 represent Blue channel
green_channel[:, :, 1] = img[quarter_height:, :quarter_width, 1]  # 1 represent Green channel
red_channel[:, :, 2] = img[:quarter_height, quarter_width:, 2]  # 2 represent Red channel

cv2.imshow('Top Left', top_left)
cv2.imshow('Top Right (Red Channel)', red_channel)
cv2.imshow('Bottom Left (Green Channel)', green_channel)
cv2.imshow('Bottom Right (Blue Channel)', blue_channel)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load input image
img = cv2.imread('im4.jpg', 0) # im3.bmp or im4.jpg

# Histogram equalization
equ = cv2.equalizeHist(img)

# Local histogram equalization
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe_img = clahe.apply(img)

# Display images
plt.figure(figsize=(10, 10))

plt.subplot(3, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(3, 2, 2)
plt.hist(img.ravel(),256,[0,256])

plt.subplot(3, 2, 3)
plt.imshow(equ, cmap='gray')
plt.title('Image after Histogram Equalization')
plt.axis('off')

plt.subplot(3, 2, 4)
plt.hist(equ.ravel(),256,[0,256])

plt.subplot(3, 2, 5)
plt.imshow(clahe_img, cmap='gray')
plt.title('Image after Local Histogram Equalization')
plt.axis('off')

plt.subplot(3, 2, 6)
plt.hist(clahe_img.ravel(),256,[0,256])

plt.show()

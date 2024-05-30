import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('noisy5.jpg')

# Median filter
median_filtered = cv2.medianBlur(image, 3)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Noisy Image')
plt.subplot(1, 2, 2), plt.imshow(median_filtered, cmap='gray'), plt.title('Median filter applied')
plt.show()

#Min-Max filter
def min_filter(image, size):
    return cv2.erode(image, np.ones((size, size), np.uint8))

def max_filter(image, size):
    return cv2.dilate(image, np.ones((size, size), np.uint8))


image = cv2.imread('noisy3.jpg', 0)

min_filtered = min_filter(image, 1)
max_filtered = max_filter(image, 1)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1), plt.imshow(image, cmap='gray'), plt.title('Noisy Image')
plt.subplot(1, 3, 2), plt.imshow(min_filtered, cmap='gray'), plt.title('Min Filter applied')
plt.subplot(1, 3, 3), plt.imshow(max_filtered, cmap='gray'), plt.title('Max Filter applied')
plt.show()

#Midpoint filter
def midpoint_filter(image, size):
    min_filtered = cv2.erode(image, np.ones((size, size), np.uint8))
    max_filtered = cv2.dilate(image, np.ones((size, size), np.uint8))
    return (min_filtered + max_filtered) / 2

image = cv2.imread('noisy3.jpg', 0)

midpoint_filtered = midpoint_filter(image, 1)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Noisy Image')
plt.subplot(1, 2, 2), plt.imshow(midpoint_filtered, cmap='gray'), plt.title('Midpoint filter applied')
plt.show()

#Alpha-Trimmed Mean Filter
def alpha_trimmed_mean_filter(image, size, d):
    pad_size = size // 2
    padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)
    output = np.zeros(image.shape, np.float32)
    
    for i in range(pad_size, padded_image.shape[0] - pad_size):
        for j in range(pad_size, padded_image.shape[1] - pad_size):
            region = padded_image[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1].flatten()
            region.sort()
            trimmed_region = region[d//2:-d//2] if d > 0 else region
            output[i - pad_size, j - pad_size] = np.mean(trimmed_region)
    
    return output

image = cv2.imread('noisy5.jpg', 0)

d = 2 
alpha_trimmed_filtered = alpha_trimmed_mean_filter(image, 2, d)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Noisy Image')
plt.subplot(1, 2, 2), plt.imshow(alpha_trimmed_filtered, cmap='gray'), plt.title('Alpha-Trimmed Mean filter applied')
plt.show()
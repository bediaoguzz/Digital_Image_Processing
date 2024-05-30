import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('noisy5.jpg', 0)

#Arithmetic mean filter
mean_filtered = cv2.blur(image, (1, 3))
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Noisy Image')
plt.subplot(1, 2, 2), plt.imshow(mean_filtered, cmap='gray'), plt.title('Arithmetic mean filter applied')
plt.show()

# Geometric mean filter
def geometric_mean_filter(image, size):
    output = np.zeros(image.shape, np.float32)
    pad_size = size // 2
    padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)
    
    for i in range(pad_size, padded_image.shape[0] - pad_size):
        for j in range(pad_size, padded_image.shape[1] - pad_size):
            region = padded_image[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1]
            output[i - pad_size, j - pad_size] = np.prod(region.flatten())**(1.0 / (size * size))
    
    return output
geometric_filtered = geometric_mean_filter(image, 1)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Noisy Image')
plt.subplot(1, 2, 2), plt.imshow(geometric_filtered, cmap='gray'), plt.title('Geometric mean filter applied')
plt.show()

# Harmonic Mean Filter
def harmonic_mean_filter(image, size):
    output = np.zeros(image.shape, np.float32)
    pad_size = size // 2
    padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)
    
    for i in range(pad_size, padded_image.shape[0] - pad_size):
        for j in range(pad_size, padded_image.shape[1] - pad_size):
            region = padded_image[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1]
            output[i - pad_size, j - pad_size] = len(region.flatten()) / np.sum(1.0 / (region.flatten() + 1e-10))
    
    return output
image = cv2.imread('noisy4.jpg', 0)
harmonic_filtered = harmonic_mean_filter(image, 3)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Noisy Image')
plt.subplot(1, 2, 2), plt.imshow(harmonic_filtered, cmap='gray'), plt.title('Harmonic mean filter applied')
plt.show()

#Contraharmonic_mean_filter
def contraharmonic_mean_filter(image, size, Q):
    output = np.zeros(image.shape, np.float32)
    pad_size = size // 2
    padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)
    
    for i in range(pad_size, padded_image.shape[0] - pad_size):
        for j in range(pad_size, padded_image.shape[1] - pad_size):
            region = padded_image[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1]
            numerator = np.sum(np.power(region, Q + 1))
            denominator = np.sum(np.power(region, Q))
            output[i - pad_size, j - pad_size] = numerator / (denominator + 1e-10)
    
    return output
image = cv2.imread('noisy4.jpg', 0)
Q = 1.5  # Q parametresi
contraharmonic_filtered = contraharmonic_mean_filter(image, 3, Q)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Noisy Image')
plt.subplot(1, 2, 2), plt.imshow(contraharmonic_filtered, cmap='gray'), plt.title('Conrtaharmonic mean filter applied')
plt.show()
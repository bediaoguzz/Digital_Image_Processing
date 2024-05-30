import cv2
import numpy as np
from matplotlib import pyplot as plt

# def adaptive_local_noise_reduction_filter(image, size, var_noise):
#     pad_size = size // 2
#     padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)
#     output = np.zeros(image.shape, np.float32)

#     mean_local = cv2.blur(image, (size, size))
#     var_local = cv2.blur(image**2, (size, size)) - mean_local**2
    
#     for i in range(pad_size, padded_image.shape[0] - pad_size):
#         for j in range(pad_size, padded_image.shape[1] - pad_size):
#             local_mean = mean_local[i - pad_size, j - pad_size]
#             local_var = var_local[i - pad_size, j - pad_size]
#             k = max(0, (local_var - var_noise) / local_var) if local_var != 0 else 0
#             output[i - pad_size, j - pad_size] = local_mean + k * (image[i - pad_size, j - pad_size] - local_mean)
    
#     return output

# image = cv2.imread('noisy3.jpg', 0)

# var_noise = 25  # Gürültü varyansı
# adaptive_local_filtered = adaptive_local_noise_reduction_filter(image, 3, var_noise)

# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Noisy Image')
# plt.subplot(1, 2, 2), plt.imshow(adaptive_local_filtered, cmap='gray'), plt.title('Adaptive local noise reduction filter applied')
# plt.show()

def adaptive_mean_filter(image, size):
    pad_size = size // 2
    padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)
    output = np.zeros(image.shape, np.float32)
    
    mean_local = cv2.blur(image, (size, size))
    variance_local = cv2.blur(np.square(image), (size, size)) - np.square(mean_local)
    mean_global = np.mean(image)
    variance_global = np.var(image)
    
    for i in range(pad_size, padded_image.shape[0] - pad_size):
        for j in range(pad_size, padded_image.shape[1] - pad_size):
            local_mean = mean_local[i - pad_size, j - pad_size]
            local_var = variance_local[i - pad_size, j - pad_size]
            if local_var < variance_global:
                output[i - pad_size, j - pad_size] = local_mean
            else:
                output[i - pad_size, j - pad_size] = image[i - pad_size, j - pad_size]
    
    return output

image = cv2.imread('noisy4.jpg', 0)

adaptive_mean_filtered = adaptive_mean_filter(image, 3)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Noisy Image')
plt.subplot(1, 2, 2), plt.imshow(adaptive_mean_filtered, cmap='gray'), plt.title('Adaptive mean filter applied')
plt.show()

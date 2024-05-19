import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from PIL import Image
import cv2

# def save_img(img, path, is_fft=False):
#     if is_fft:
#         # Compute the magnitude spectrum and use a logarithmic scale
#         img = np.abs(img)
#         img = np.fft.fftshift(img)  # Shift the zero frequency components to the center of the spectrum
#         img = np.log1p(img)  # Use log scale for better visibility

#     cv2.imwrite(path, img)


# def save_img(data, filename, is_fft=False):
#     if is_fft:
#         # Compute the magnitude spectrum and use a logarithmic scale
#         data = np.abs(data)
#         data = np.fft.fftshift(data)  # Shift the zero frequency components to the center of the spectrum
#         data = np.log1p(data)  # Use log scale for better visibility
#         data = 255 * data / np.max(data)  # Normalize to 0-255
#         data = data.astype(np.uint8)
#     else:
#         if data.dtype != np.uint8:
#             data = 255 * (data - np.min(data)) / (np.max(data) - np.min(data))
#             data = data.astype(np.uint8)
    
#     cv2.imwrite(filename, data)


def save_img(data, filename, is_fft=False):
    plt.figure()
    if is_fft:
        # Compute the magnitude spectrum and use a logarithmic scale
        data = np.abs(data)
        data = np.fft.fftshift(data)  # Shift the zero frequency components to the center of the spectrum
        data = np.log1p(data)  # Use log scale for better visibility
    plt.imshow(data, cmap='gray')
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    # plt.close()
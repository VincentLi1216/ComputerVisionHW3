import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from PIL import Image

def plot(data, title):
    plot.i += 1
    plt.subplot(2, 2, plot.i)
    plt.imshow(data, cmap='gray')
    plt.title(title)
plot.i = 0

# Load the image and convert to grayscale
im = Image.open('imgs/Lenna.jpg')
if im.mode != 'L':
    im = im.convert('L')
data = np.array(im, dtype=float)
plot(data, 'Original')

# Define a simple lowpass filter kernel (e.g., averaging filter)
kernel_size = 9  # Size of the kernel (9x9)
kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)

# Pad the kernel to the size of the image and shift the kernel center to the center of the image
kernel_padded = np.zeros_like(data)
kernel_center = tuple(np.array(kernel_padded.shape) // 2 - np.array(kernel.shape) // 2)
kernel_padded[kernel_center[0]:kernel_center[0]+kernel.shape[0], kernel_center[1]:kernel_center[1]+kernel.shape[1]] = kernel
kernel_shifted = np.fft.fftshift(kernel_padded)

# Perform FFT on the image
fft_img = np.fft.fft2(data)

# Perform FFT on the shifted kernel
fft_kernel = np.fft.fft2(kernel_shifted)

# Apply lowpass filter in the frequency domain by multiplying
fshift = fft_img * fft_kernel

# Inverse FFT to get the filtered image back in spatial domain
img_freq = np.fft.ifft2(fshift)
img_filtered = np.real(img_freq)

# Normalize the lowpass filter output
img_filtered = (img_filtered - img_filtered.min()) / (img_filtered.max() - img_filtered.min()) * 255

plot(img_filtered, 'Lowpass Filtered (Freq Domain)')

plt.show()

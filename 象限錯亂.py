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

# Define a highpass filter kernel
kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])

# Pad the kernel to the size of the image
kernel_padded = np.pad(kernel, [(0, data.shape[0] - kernel.shape[0]), (0, data.shape[1] - kernel.shape[1])], mode='constant')

# Perform FFT on the image and the padded kernel
fft_img = np.fft.fft2(data)
fft_kernel = np.fft.fft2(kernel_padded)

# Apply highpass filter in the frequency domain by multiplying
fshift = fft_img * fft_kernel

# Inverse FFT to get the filtered image back in spatial domain
img_freq = np.fft.ifft2(fshift)
img_freq = np.fft.ifftshift(img_freq)  # Shift back the zero-frequency component to the center of the spectrum

# Normalize the highpass filter output
img_filtered = np.real(img_freq)
img_filtered = (img_filtered - img_filtered.min()) / (img_filtered.max() - img_filtered.min()) * 255
plot(img_filtered, 'Highpass Filtered (Freq Domain)')

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from PIL import Image

def plot(data, title, is_fft=False):
    plot.i += 1
    plt.subplot(2, 2, plot.i)
    if is_fft:
        # Compute the magnitude spectrum and use a logarithmic scale
        data = np.abs(data)
        data = np.fft.fftshift(data)  # Shift the zero frequency components to the center of the spectrum
        data = np.log1p(data)  # Use log scale for better visibility
    plt.imshow(data, cmap='gray')
    plt.title(title)
plot.i = 0

# Load the image and convert to grayscale
im = Image.open('imgs/Lenna.jpg')
# im = Image.open('imgs/img2.png')
if im.mode != 'L':
    im = im.convert('L')
data = np.array(im, dtype=float)
plot(data, 'Original')

# Perform FFT on the image
fft_img = np.fft.fft2(data)

# Plot the FFT image
plot(fft_img, 'FFT of Image', is_fft=True)

plt.show()

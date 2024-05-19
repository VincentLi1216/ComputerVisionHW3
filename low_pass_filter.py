import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from PIL import Image


def plot(data, title):
    plot.i += 1
    plt.subplot(2, 2, plot.i)
    plt.imshow(data, cmap="gray")
    plt.title(title)


plot.i = 0

# Load the image and convert to grayscale
im = Image.open("imgs/Lenna.jpg")
if im.mode != "L":
    im = im.convert("L")
data = np.array(im, dtype=float)
plot(data, "Original")

# Define padding size (same as half of kernel size)
pad_size = 1  # For 3x3 kernel
data_padded = np.pad(data, pad_size, mode="constant", constant_values=0)

# A very simple and very narrow lowpass filter (mean filter)
kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9  # Normalize the kernel
lowpass_3x3 = ndimage.convolve(data_padded, kernel)
lowpass_3x3 = lowpass_3x3[pad_size:-pad_size, pad_size:-pad_size]  # Remove padding
# Normalize the lowpass filter output
lowpass_3x3 = (
    (lowpass_3x3 - lowpass_3x3.min()) / (lowpass_3x3.max() - lowpass_3x3.min()) * 255
)
plot(lowpass_3x3, "Simple 3x3 Lowpass")

# Similarly, add padding and then remove it after convolution for the 5x5 kernel
pad_size = 2  # For 5x5 kernel
data_padded = np.pad(data, pad_size, mode="constant", constant_values=0)
kernel = (
    np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]
    )
    / 25
)  # Normalize the kernel
lowpass_5x5 = ndimage.convolve(data_padded, kernel)
lowpass_5x5 = lowpass_5x5[pad_size:-pad_size, pad_size:-pad_size]  # Remove padding
# Normalize the lowpass filter output
lowpass_5x5 = (
    (lowpass_5x5 - lowpass_5x5.min()) / (lowpass_5x5.max() - lowpass_5x5.min()) * 255
)
plot(lowpass_5x5, "Simple 5x5 Lowpass")

# Use the Gaussian filter directly for lowpass filtering
gauss_lowpass = ndimage.gaussian_filter(data, 3)
# Normalize the Gaussian lowpass filter output
gauss_lowpass = (
    (gauss_lowpass - gauss_lowpass.min())
    / (gauss_lowpass.max() - gauss_lowpass.min())
    * 255
)
plot(gauss_lowpass, r"Gaussian Lowpass, $\sigma = 3 pixels$")

plt.show()

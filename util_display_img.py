import matplotlib.pyplot as plt
import numpy as np


def plot(data, title, subplot_index,is_fft=False, plot_size=(4,3)):
    if is_fft:
        # Compute the magnitude spectrum and use a logarithmic scale
        data = np.abs(data)
        data = np.fft.fftshift(data)  # Shift the zero frequency components to the center of the spectrum
        data = np.log1p(data)  # Use log scale for better visibility

    plt.subplot(plot_size[0], plot_size[1], subplot_index)
    plt.imshow(data, cmap='gray')
    plt.title(title)
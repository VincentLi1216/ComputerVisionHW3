import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from PIL import Image
import cv2

from util_display_img import plot
from util_save_img import save_img


def fft_highpass(img):

    # Define a highpass filter kernel
    
    kernel = np.array([
                        [-1, -1, -1],
                        [-1,  8, -1], 
                        [-1, -1, -1]])
    kernel = np.array([
                        [ 0, -1,  0],
                        [-1,  4, -1], 
                        [ 0, -1,  0]])

    # Pad the kernel to the size of the image and shift the kernel center to the center of the image
    kernel_padded = np.zeros_like(img)
    kernel_center = tuple(
        np.array(kernel_padded.shape) // 2 - np.array(kernel.shape) // 2
    )
    kernel_padded[
        kernel_center[0] : kernel_center[0] + kernel.shape[0],
        kernel_center[1] : kernel_center[1] + kernel.shape[1],
    ] = kernel
    kernel_shifted = np.fft.fftshift(kernel_padded)

    # Perform FFT on the image
    fft_img = np.fft.fft2(img)

    # Perform FFT on the shifted kernel
    fft_kernel = np.fft.fft2(kernel_shifted)

    # Apply highpass filter in the frequency domain by multiplying
    fshift = fft_img * fft_kernel

    # Inverse FFT to get the filtered image back in spatial domain
    img_freq = np.fft.ifft2(fshift)
    img_filtered = np.real(img_freq)
    # Normalize the highpass filter output
    img_filtered = np.uint8(np.abs(img_freq))
    # img_filtered[img_filtered>50] = 255 


    return img_filtered, fft_kernel


def fft_lowpass(img, kernel_size=9):
    # Define a simple lowpass filter kernel (e.g., averaging filter)
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)

    # Pad the kernel to the size of the image and shift the kernel center to the center of the image
    kernel_padded = np.zeros_like(img)
    kernel_center = tuple(
        np.array(kernel_padded.shape) // 2 - np.array(kernel.shape) // 2
    )
    kernel_padded[
        kernel_center[0] : kernel_center[0] + kernel.shape[0],
        kernel_center[1] : kernel_center[1] + kernel.shape[1],
    ] = kernel
    kernel_shifted = np.fft.fftshift(kernel_padded)

    # Perform FFT on the image
    fft_img = np.fft.fft2(img)

    # Perform FFT on the shifted kernel
    fft_kernel = np.fft.fft2(kernel_shifted)

    # Apply lowpass filter in the frequency domain by multiplying
    fshift = fft_img * fft_kernel

    # Inverse FFT to get the filtered image back in spatial domain
    img_freq = np.fft.ifft2(fshift)
    img_filtered = np.real(img_freq)

    # Normalize the highpass filter output
    img_filtered = np.uint8(np.abs(img_freq))


    return img_filtered, fft_kernel


if __name__ == "__main__":
    # Load the image and convert to grayscale
    im = Image.open("imgs/Lenna.jpg")
    if im.mode != "L":
        im = im.convert("L")  # 轉換為灰度影像
    img = np.array(im, dtype=float)

    high_pass_fft, kernel_fft = fft_highpass(img)

    plt.figure(figsize=(5, 10))
    plot(img, "Original", 1)
    plot(high_pass_fft, "Highpass Filtered (Freq Domain)", 2)
    plot(kernel_fft, "Highpass Filter Kernel", 3, is_fft=True)

    plt.tight_layout()
    plt.show()

    low_pass, kernel_fft = fft_lowpass(img, 5)

    plt.figure(figsize=(5, 5))
    plot(img, "Original", 1)
    plot(low_pass, "Lowpass Filtered (Freq Domain)", 2)
    plot(kernel_fft, "Lowpass Filter Kernel", 3, is_fft=True)

    save_img(high_pass_fft, "output/fft_high_pass.png")
    # save_img(low_pass, 'output/fft_low_pass.png')
    # save_img(kernel_fft, 'output/fft_low_pass_kernel.png', True)

    plt.tight_layout()
    plt.show()


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from util_conv_filter import conv_highpass, conv_lowpass
from util_fft_filter import fft_highpass, fft_lowpass
from util_display_img import plot
from util_save_img import save_img
from util_img2fft import img2fft

def main():
    plot_size = (3,2)

    # Load the image and convert to grayscale
    im = Image.open('imgs/Lenna.jpg')
    if im.mode != 'L':
        im = im.convert('L')  # 轉換為灰度影像
    img = np.array(im, dtype=float)

    high_pass_fft, kernel_fft = fft_highpass(img)

    kernel_img = Image.open('imgs/highpass_kernal.png')
    img_fft = img2fft(im)


    plt.figure(figsize=(5, 10))
    plot(img, 'Original',1, plot_size=plot_size)
    # plot(high_pass_fft, 'Highpass Filtered (Freq Domain)', 2, plot_size=plot_size)
    plot(img_fft, "img fft", 2, plot_size=plot_size, is_fft=True)
    plot(kernel_img, 'Highpass Filter Kernel', 3, plot_size=plot_size)
    plot(kernel_fft, 'Highpass Filter Kernel', 4, is_fft=True, plot_size=plot_size)
    plot(high_pass_fft, 'Highpass Filtered (Freq Domain)', 5, plot_size=plot_size)
    plot(img2fft(high_pass_fft), 'Highpass Filtered (Freq Domain)', 6, is_fft=True, plot_size=plot_size)
 
    plt.tight_layout()
    plt.show()

    low_pass, kernel_fft = fft_lowpass(img, 5)

    plt.figure(figsize=(5, 10))
    plot(img, 'Original', 1)
    plot(low_pass, 'Lowpass Filtered (Freq Domain)', 2)
    plot(kernel_fft, 'Lowpass Filter Kernel', 3, is_fft=True)

    save_img(high_pass_fft, 'output/fft_high_pass.png')
    # save_img(low_pass, 'output/fft_low_pass.png')
    # save_img(kernel_fft, 'output/fft_low_pass_kernel.png', True)

    plt.tight_layout()
    plt.show()
    



main()
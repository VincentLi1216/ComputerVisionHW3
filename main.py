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
    im = Image.open('imgs/white_house.png')
    if im.mode != 'L':
        im = im.convert('L')  # 轉換為灰度影像
    img = np.array(im, dtype=float)


    '''
    highpass filter
    '''
    high_pass_fft, high_kernel_fft = fft_highpass(img)

    # kernel array
    high_kernel_img = Image.open('imgs/highpass_kernal.png')


    '''
    lowpass filter
    '''
    low_pass_fft, low_kernel_fft = fft_lowpass(img, 5)


    low_kernel_img = Image.open('imgs/lowpass_kernal.png')
    '''
    highpass plot
    '''
    plt.figure(figsize=(10, 10))
    plot(img, 'Original',1, plot_size=plot_size)
    plot(img2fft(im), "Original(fft)", 2, plot_size=plot_size, is_fft=True)
    plot(high_kernel_img, 'Highpass Filter Kernel', 3, plot_size=plot_size)
    plot(high_kernel_fft, 'Highpass Filter Kernel(fft)', 4, is_fft=True, plot_size=plot_size)
    plot(high_pass_fft, 'Highpass Filtered', 5, plot_size=plot_size)
    plot(img2fft(high_pass_fft), 'Highpass Filtered(fft)', 6, is_fft=True, plot_size=plot_size)
    
    plt.tight_layout()
    plt.show()
 
    '''
    lowpass plot
    '''
    plt.figure(figsize=(10, 10))
    plot(img, 'Original', 1, plot_size=plot_size)
    plot(img2fft(im), "Original(fft)", 2, plot_size=plot_size, is_fft=True)
    plot(low_kernel_img, 'Lowpass Filter Kernel', 3, plot_size=plot_size)
    plot(low_kernel_fft, 'Lowpass Filter Kernel(fft)', 4, is_fft=True, plot_size=plot_size)
    plot(low_pass_fft, 'Lowpass Filtered', 5, plot_size=plot_size)
    plot(img2fft(low_pass_fft), 'Lowpass Filtered(fft)', 6, is_fft=True, plot_size=plot_size)

    plt.tight_layout()
    plt.show()

    '''
    conv filter
    '''
    conv_highpass_filters = conv_highpass(img, filter_name='3x3')
    conv_lowpass_filters = conv_lowpass(img, filter_name='5x5')
    plt.figure(figsize=(10, 10))
    plot(conv_highpass_filters, "highpass(conv)", 1, plot_size=plot_size)
    plot(conv_lowpass_filters, "lowpass(conv)", 2, plot_size=plot_size)
    plt.tight_layout()
    plt.show()


    



main()
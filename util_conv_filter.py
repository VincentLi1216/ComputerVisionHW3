import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from PIL import Image

from util_display_img import plot

plot_size = (4,3)

def conv_highpass(data, filter_name):
    # 高通濾波器
    filters = []
    
    if filter_name == '3x3':
        # 簡單的 3x3 高通濾波器
        kernel_highpass_3x3 = np.array([[ 0, -1,  0],
                                        [-1,  4, -1],
                                        [ 0, -1,  0]])
        highpass_3x3 = ndimage.convolve(data, kernel_highpass_3x3)
        highpass_3x3 = np.real(highpass_3x3)
        highpass_3x3 = np.uint8(np.abs(highpass_3x3))
        # highpass_3x3 = (highpass_3x3 - highpass_3x3.min()) / (highpass_3x3.max() - highpass_3x3.min()) * 255
        return highpass_3x3

    if filter_name == '5x5':
        # 簡單的 5x5 高通濾波器
        kernel_highpass_5x5 = np.array([[-1, -1, -1, -1, -1],
                                        [-1,  1,  2,  1, -1],
                                        [-1,  2,  4,  2, -1],
                                        [-1,  1,  2,  1, -1],
                                        [-1, -1, -1, -1, -1]])
        highpass_5x5 = ndimage.convolve(data, kernel_highpass_5x5)
        highpass_5x5 = (highpass_5x5 - highpass_5x5.min()) / (highpass_5x5.max() - highpass_5x5.min()) * 255
        return highpass_5x5

    if filter_name == 'gaussian':
        # 高斯高通濾波器
        lowpass_gaussian = ndimage.gaussian_filter(data, 3)
        gauss_highpass = data - lowpass_gaussian
        gauss_highpass = (gauss_highpass - gauss_highpass.min()) / (gauss_highpass.max() - gauss_highpass.min()) * 255
        return gauss_highpass


def conv_lowpass(data, filter_name):
    # 低通濾波器
    filters = []
    
    if filter_name == '3x3':
        # 簡單的 3x3 低通濾波器
        kernel_lowpass_3x3 = np.array([[1, 1, 1],
                                    [1, 1, 1],
                                    [1, 1, 1]]) / 9
        lowpass_3x3 = ndimage.convolve(data, kernel_lowpass_3x3)
        lowpass_3x3 = (lowpass_3x3 - lowpass_3x3.min()) / (lowpass_3x3.max() - lowpass_3x3.min()) * 255
        return lowpass_3x3

    if filter_name == '5x5':
        # 簡單的 5x5 低通濾波器
        kernel_lowpass_5x5 = np.array([[1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1]]) / 25
        lowpass_5x5 = ndimage.convolve(data, kernel_lowpass_5x5)
        lowpass_5x5 = (lowpass_5x5 - lowpass_5x5.min()) / (lowpass_5x5.max() - lowpass_5x5.min()) * 255
        return lowpass_5x5

    if filter_name == 'gaussian':
        # 高斯低通濾波器
        gauss_lowpass = ndimage.gaussian_filter(data, 3)
        gauss_lowpass = (gauss_lowpass - gauss_lowpass.min()) / (gauss_lowpass.max() - gauss_lowpass.min()) * 255
        return gauss_lowpass



if __name__ == '__main__':
    # 載入圖像並轉換為灰度
    im = Image.open('imgs/Lenna.jpg')
    if im.mode != 'L':
        im = im.convert('L')  # 轉換為灰度影像
    data = np.array(im, dtype=float)

    # 應用濾波器
    highpass_filters = conv_highpass(data, filter_name='3x3')
    lowpass_filters = conv_lowpass(data, filter_name='3x3')

    # 繪圖
    plt.figure(figsize=(10, 15))
    plot(data, 'Input Image', 1)
    plot(highpass_filters, 'Simple 3x3 Highpass', 2)
    plot(lowpass_filters, 'Simple 3x3 Lowpass', 3)


    plt.tight_layout()
    plt.show()

import numpy as np
from scipy import signal

img = np.array([[0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]])
kernel = np.array([[0, 0, 0, 0, 0],
                   [0, 1, 2, 3, 0],
                   [0, 4, 5, 6, 0],
                   [0, 7, 8, 9, 0],
                   [0, 0, 0, 0, 0]])

fft_img = np.fft.fft2(img)
fft_kernel = np.fft.fft2(kernel)

fshift = (fft_img* fft_kernel)  # Convelution theory at frequency domain

img_freq = np.fft.ifft2(fshift)

img_freq = np.fft.ifftshift(np.round(np.real(img_freq)))


print(img_freq)

img_real = signal.convolve2d( img,kernel,'same')

print(img_real)




# 想一下卷積原理 動手做一下計算  看一下傅立葉轉換公式　np.pad是會用到的指令
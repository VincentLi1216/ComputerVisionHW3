import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy import signal
from PIL import Image

# 載入影像並轉換成灰階
image = Image.open('imgs/Lenna.jpg').convert('L')
image = np.array(image)

# 定義高通和低通濾波器（簡單的例子）
size = image.shape
low_pass = np.ones((size[0], size[1]))
high_pass = np.ones((size[0], size[1]))

# 高通濾波器：保留邊緣
for i in range(size[0]):
    for j in range(size[1]):
        if i >= size[0]//3 and i < 2*size[0]//3 and j >= size[1]//3 and j < 2*size[1]//3:
            high_pass[i, j] = 0

# 低通濾波器：只保留中間區域的頻率
for i in range(size[0]):
    for j in range(size[1]):
        if i < size[0]//3 or i >= 2*size[0]//3 or j < size[1]//3 or j >= 2*size[1]//3:
            low_pass[i, j] = 0

high_pass = high_pass - low_pass

# 時域卷積操作
convolved_low = signal.convolve2d(image, low_pass, boundary='symm', mode='same')
convolved_high = signal.convolve2d(image, high_pass, boundary='symm', mode='same')

# 頻域相乘操作
image_fft = fftpack.fft2(image)
low_fft = fftpack.fft2(low_pass)
high_fft = fftpack.fft2(high_pass)

mul_low = image_fft * low_fft
mul_high = image_fft * high_fft

# 逆傅立葉變換
ifft_low = fftpack.ifft2(mul_low).real
ifft_high = fftpack.ifft2(mul_high).real

# 顯示結果
plt.figure(figsize=(12, 8))
plt.subplot(231), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(232), plt.imshow(high_pass, cmap='gray'), plt.title('High Pass Filter')
plt.subplot(233), plt.imshow(low_pass, cmap='gray'), plt.title('Low Pass Filter')
plt.subplot(234), plt.imshow(ifft_low, cmap='gray'), plt.title('Low Pass Filtered (Freq Domain)')
plt.subplot(235), plt.imshow(convolved_low, cmap='gray'), plt.title('Low Pass Filtered (Time Domain)')
plt.subplot(236), plt.imshow(ifft_high, cmap='gray'), plt.title('High Pass Filtered (Freq Domain)')
plt.show()

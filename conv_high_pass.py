import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from PIL import Image

def plot(data, title):
    plot.i += 1
    plt.subplot(2, 2, plot.i)
    plt.imshow(data, cmap='gray')  # 確保使用灰度顯示
    plt.title(title)
plot.i = 0

# Load the image and convert to grayscale
im = Image.open('imgs/Lenna.jpg')
if im.mode != 'L':
    im = im.convert('L')  # 轉換為灰度影像
data = np.array(im, dtype=float)
plot(data, 'Original')

# A very simple and very narrow highpass filter
kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])
highpass_3x3 = ndimage.convolve(data, kernel)
# Normalize the highpass filter output
highpass_3x3 = (highpass_3x3 - highpass_3x3.min()) / (highpass_3x3.max() - highpass_3x3.min()) * 255
plot(highpass_3x3, 'Simple 3x3 Highpass')

# A slightly "wider", but still very simple highpass filter
kernel = np.array([[-1, -1, -1, -1, -1],
                   [-1,  1,  2,  1, -1],
                   [-1,  2,  4,  2, -1],
                   [-1,  1,  2,  1, -1],
                   [-1, -1, -1, -1, -1]])
highpass_5x5 = ndimage.convolve(data, kernel)
# Normalize the highpass filter output
highpass_5x5 = (highpass_5x5 - highpass_5x5.min()) / (highpass_5x5.max() - highpass_5x5.min()) * 255
plot(highpass_5x5, 'Simple 5x5 Highpass')

# Another way of making a highpass filter is to simply subtract a lowpass
# filtered image from the original. Here, we use a simple Gaussian filter
# to "blur" (i.e., a lowpass filter) the original.
lowpass = ndimage.gaussian_filter(data, 3)
gauss_highpass = data - lowpass
# Normalize the Gaussian highpass filter output
gauss_highpass = (gauss_highpass - gauss_highpass.min()) / (gauss_highpass.max() - gauss_highpass.min()) * 255
plot(gauss_highpass, r'Gaussian Highpass, $\sigma = 3 pixels$')

plt.show()

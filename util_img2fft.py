import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from PIL import Image


def img2fft(im):

    # im = Image.open('imgs/img2.png')
    try:
        if im.mode != 'L':
            im = im.convert('L')
    except:
        pass
    data = np.array(im, dtype=float)

    # Perform FFT on the image
    fft_img = np.fft.fft2(data)
    return fft_img

    

# Transform RGB to HSV, L*a*b* and CMYK.
# Divide in 10 channel colour.

from PIL import Image
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
import cv2


class SymptomSegmentation:
    def __init__(self, image):
        self.R, self.G, self.B = cv2.split(image)
        self.r1 = R / (G + 1E-6)
        self.r2 = B / (G + 1E-6)


class ColourTransformation:
    def __init__(self, image):
        im = Image.fromarray(image)
        im = im.convert('CMYK')

        self.RGB = image
        self.HSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        self.Lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        self.CMYK = np.array(im)

        self.R, self.G, self.B = cv2.split(self.RGB)
        self.H, self.S, self.V = cv2.split(self.HSV)
        self.L, self.a, self.b = cv2.split(self.Lab)
        self.C, self.M, self.Y, self.K = cv2.split(self.CMYK)

    def show_channels(self):
        plt.figure()
        plt.subplot(6, 2, 1), plt.imshow(self.RGB)
        plt.subplot(6, 2, 2), plt.imshow(self.H, cmap='gray')
        plt.subplot(6, 2, 3), plt.imshow(self.S, cmap='gray')
        plt.subplot(6, 2, 4), plt.imshow(self.V, cmap='gray')
        plt.subplot(6, 2, 5), plt.imshow(self.L, cmap='gray')
        plt.subplot(6, 2, 6), plt.imshow(self.a, cmap='gray')
        plt.subplot(6, 2, 7), plt.imshow(self.b, cmap='gray')
        plt.subplot(6, 2, 8), plt.imshow(self.C, cmap='gray')
        plt.subplot(6, 2, 9), plt.imshow(self.M, cmap='gray')
        plt.subplot(6, 2, 10), plt.imshow(self.Y, cmap='gray')
        plt.subplot(6, 2, 11), plt.imshow(self.K, cmap='gray')
        plt.show()


    pass


#resize(self.RGB, (self.RGB[0] // 4, self.RGB[1] // 4), anti_aliasing=True)
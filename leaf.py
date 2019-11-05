# Transform RGB to HSV, L*a*b* and CMYK.
# Divide in 10 channel colour.

from PIL import Image
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
import cv2
from segment import segment_leaf


class Segmentation:
    def __init__(self, image_path, filling_mode, smooth, marker_intensity):
        # read image and segment leaf
        self.original, self.output_image = segment_leaf((image_path), filling_mode, smooth, marker_intensity)
        self.original = cv2.rotate(self.original, cv2.ROTATE_90_COUNTERCLOCKWISE)
        self.output_image = cv2.rotate(self.output_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def show(self):
        plt.figure()
        plt.subplot(1, 2, 1), plt.imshow(self.original)
        plt.subplot(1, 2, 2), plt.imshow(self.output_image)
        plt.show()


class SymptomSegmentation:
    def __init__(self, image):
        self.R, self.G, self.B = cv2.split(image)
        self.r1 = self.R / (self.G + 1E-6)
        self.r2 = self.B / (self.G + 1E-6)
        self.M1 = (self.r1 > 1).astype(np.int)
        self.M2 = (self.r2 > 1).astype(np.int)
        self.M3 = (self.r1 > .9).astype(np.int)
        self.M4 = (self.r2 > .67).astype(np.int)
        self.Ma = self.M1 | self.M2
        self.Mb = self.M3 & self.M4
        self.M = self.Ma | self.Mb


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
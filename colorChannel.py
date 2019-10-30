# Transform RGB to HSV, L*a*b* and CMYK.
# Divide in 10 channel colour.

from PIL import Image
import numpy as np
import cv2


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

    pass

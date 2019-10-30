# Transform RGB to HSV, L*a*b* and CMYK.
# Divide in 10 channel colour.

from skimage.color import rgb2hsv
from skimage.color import rgb2lab
from PIL import Image
import numpy as np
import cv2


class ColourTransformation:
    def __init__(self, image):
        im = Image.fromarray(image)
        im = im.convert('CMYK')

        self.image = image
        self.HSV = rgb2hsv(image)
        self.Lab = rgb2lab(image)
        self.CMYK = np.array(im)

        self.H, self.S, self.V = cv2.split(self.HSV)
        self.L, self.a, self.b = cv2.split(self.Lab)
        self.C, self.M, self.Y, self.K = cv2.split(self.CMYK)

    pass

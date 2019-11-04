from colorChannel import ColourTransformation

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2


# Load image in BGR.
imagePath = 'Images/DSC_0010.jpg'
img = mpimg.imread(imagePath)

im = ColourTransformation(img)

im.show_channels()


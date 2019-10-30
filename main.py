import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from colorChannel import ColourTransformation

# Load image in BGR.
imagePath = 'Images/DSC_0010.jpg'
img = mpimg.imread(imagePath)

im = ColourTransformation(img)



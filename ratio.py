import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2


# Load image in BGR.
imagePath = 'Images/DSC_0010.jpg'
img = mpimg.imread(imagePath)

R, G, B = cv2.split(img)

r1 = R/(G + 1E-6)
r2 = B/(G + 1E-6)



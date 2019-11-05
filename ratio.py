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

M1 = (r1 > 1).astype(np.int)
M2 = (r2 > 1).astype(np.int)
M3 = (r1 > .9).astype(np.int)
M4 = (r2 > .67).astype(np.int)

Ma = (M1 | M2)
Mb = (M3 & M4)

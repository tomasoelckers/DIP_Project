from leaf import ColourTransformation
from leaf import SymptomSegmentation
from leaf import Segmentation

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np

# Load image in BGR.
# imagePath = 'Images/DSC_0010.jpg'
imagePath = 'Images/DSC_0010_marked.jpg'
# imagePath = 'Images/alga 1.jpeg'
img = (mpimg.imread(imagePath))
img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
# imS = Segmentation(imagePath, 'no', True, 0)
# imS.show()

imSS = SymptomSegmentation(img)
plt.figure(1)
plt.imshow(imSS.img)
plt.show()

imCT = ColourTransformation(imSS.img)
histogram = imCT.histogram('Gray')
# configure and draw the histogram figure
plt.figure(2)
plt.title("Grayscale Histogram")
plt.xlabel("grayscale value")
plt.ylabel("pixels")
plt.xlim([0.0, len(histogram)])
plt.ylim([min(histogram), max(histogram)])
plt.plot(histogram)
plt.show()
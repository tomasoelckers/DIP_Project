from leaf import ColourTransformation
from leaf import SymptomSegmentation
from leaf import Segmentation
from Color_Sharpen import Run as ColorSharpen

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np

# Load image segmented
'-------------------'
# imagePath = 'Images/DSC_0010.jpg'
imagePath = 'Images/DSC_0010_marked.jpg'
# imagePath = 'Images/alga 1.jpeg'
img = (mpimg.imread(imagePath))
img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

# Implemention of Tarik Method
'-----------------------------'
image = ColorSharpen(img)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Segmentation of the Symptoms
'-----------------------------'
imSS = SymptomSegmentation(img)
imSS_1 = SymptomSegmentation(image)

plt.figure(1)
plt.subplot(1, 2, 1)
plt.imshow(imSS.img)
plt.subplot(1, 2, 2)
plt.imshow(imSS_1.img)
plt.show()

# Split and Get 10 Different Channel of Colour and its histogram
'---------------------------------------------'
imCT = ColourTransformation(imSS.img)
imCT_1 = ColourTransformation(imSS_1.img)


# Plot Histogram
'---------------'
histogram = imCT.histogram('Gray')
histogram_1 = imCT_1.histogram('Gray')
# configure and draw the histogram figure
plt.figure(2)
plt.title("Grayscale Histogram")
plt.xlabel("grayscale value")
plt.ylabel("pixels")
plt.xlim([0.0, len(histogram)])
plt.ylim([min(histogram), max(histogram)])
plt.plot(histogram, 'r', label='Original')
plt.plot(histogram_1, 'b', label='Tarik Method')
plt.legend()
plt.show()

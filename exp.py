from leaf import ColourTransformation
from leaf import SymptomSegmentation
from leaf import Segmentation

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np

# Load image in BGR.
imagePath = 'Images/DSC_0010.jpg'
# imagePath = 'Images/DSC_0010_marked.jpg'
# imagePath = 'Images/alga 1.jpeg'
img = (mpimg.imread(imagePath))
img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
imS = Segmentation(imagePath, 'flood', 0)
imS.show()

img = np.hstack((imS.original, imS.output_image))

plt.figure(1)
plt.imshow(img)
plt.show()

from leaf import ColourTransformation
from leaf import SymptomSegmentation
from leaf import Segmentation

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np


# Load image in BGR.
imagePath = 'Images/DSC_0010.jpg'
#imagePath = 'Images/alga 1.jpeg'
img = mpimg.imread(imagePath)

imS = Segmentation(imagePath, 'no', True, 0)
imS.show()

imSS = SymptomSegmentation(imS.output_image)
mask = imSS.M.astype(np.uint8)
mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
res = cv2.bitwise_or(img, mask)


plt.figure()
plt.imshow(res)
plt.show()

imCT = ColourTransformation(img)

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from skimage.filters import gaussian
from skimage.segmentation import active_contour

# Load image in BGR.
imagePath = 'Images/DSC_0010.jpg'
img = mpimg.imread(imagePath)

# Converting an image to RGB from BGR.
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Converting an image to gray scale from BGR.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Plot image in rgb and gray scale.
plt.figure(1)
plt.subplot(121), plt.imshow(img)
plt.subplot(122), plt.imshow(gray, cmap='gray')
plt.show()

s = np.linspace(0, 2*np.pi, 400)
r = 100 + 100*np.sin(s)
c = 220 + 100*np.cos(s)
init = np.array([r, c]).T

snake = active_contour(gaussian(gray, 3),
                       init, alpha=0.015, beta=10, gamma=0.001,
                       coordinates='rc')

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(img, cmap=plt.cm.gray)
ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img.shape[1], img.shape[0], 0])

plt.show()


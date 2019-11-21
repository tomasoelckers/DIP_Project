from PIL import Image
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from segment import segment_leaf, main
from Color_Sharpen import Run as ColorSharpen
from os import walk


# Segmentation in Construction...
class Segmentation:
    def __init__(self, image_path, filling_mode, smooth, marker_intensity):
        # Read image and segment leaf.
        original, output_image = main(image_path, filling_mode, smooth, marker_intensity)
        original = cv2.rotate(original, cv2.ROTATE_90_COUNTERCLOCKWISE)  # Rotate image 90 degrees CC.
        output_image = cv2.rotate(output_image, cv2.ROTATE_90_COUNTERCLOCKWISE)  # Rotate image 90 degrees CC.
        self.original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)  # Change format from BGR to RGB.
        self.output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)  # Change format from BGR to RGB.

    def show(self):
        plt.figure()
        plt.subplot(1, 2, 1), plt.imshow(self.original)
        plt.subplot(1, 2, 2), plt.imshow(self.output_image)
        plt.show()


# Segmentation of the Symptom.
class SymptomSegmentation:
    def __init__(self, image):
        R, G, B = cv2.split(image)  # Split each RGB channel.
        r1 = R / (G + 1E-6)  # First symptom ratio mask.
        r2 = B / (G + 1E-6)  # Second symptom ratio mask.
        # Logic operation aiming final mask with symptoms.
        M1 = (r1 > 1).astype(np.int)
        M2 = (r2 > 1).astype(np.int)
        M3 = (r1 > .9).astype(np.int)
        M4 = (r2 > .67).astype(np.int)
        Ma = M1 | M2  # Or logic operation.
        Mb = M3 & M4  # And logic operation.
        M = Ma | Mb  # Or logic operation.
        mask = M.astype(np.uint8)  # Change format to uint8.

        # Applying Mask to each channel.
        R = cv2.bitwise_and(R, R, mask=mask)
        G = cv2.bitwise_and(G, G, mask=mask)
        B = cv2.bitwise_and(B, B, mask=mask)

        # Merge 3 final channel, RGB.
        self.img = cv2.merge((R, G, B))


# Split and Transform RGB to 10 Channel Colour of HSV, L*a*b and CMYK space colour.
class ColourTransformation:
    def __init__(self, image):
        im = Image.fromarray(image)  # Transform image to a PIL image object.
        im = im.convert('CMYK')  # Convert the RGB image to CMYK.

        self.Gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Change format from RGB to Grayscale.
        self.RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Change format from BGR to RGB.
        self.HSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)  # Change format from RGB to HSV.
        self.Lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)  # Change format from RGB to Lab.
        self.CMYK = np.array(im)  # CMYK PIL object to a numpy array.

        self.R, self.G, self.B = cv2.split(self.RGB)  # Split each channel.
        self.H, self.S, self.V = cv2.split(self.HSV)  # Split each channel.
        self.L, self.a, self.b = cv2.split(self.Lab)  # Split each channel.
        self.C, self.M, self.Y, self.K = cv2.split(self.CMYK)  # Split each channel.

    # Get histogram array for each channel.
    def histogram(self, channel):
        if channel == 'R':
            histogram, bin_edges = np.histogram(self.R, bins=256, range=(0, 256))
        elif channel == 'G':
            histogram, bin_edges = np.histogram(self.G, bins=256, range=(0, 256))
        elif channel == 'B':
            histogram, bin_edges = np.histogram(self.B, bins=256, range=(0, 256))
        elif channel == 'Gray':
            histogram, bin_edges = np.histogram(self.Gray, bins=256, range=(0, 256))
        elif channel == 'H':
            histogram, bin_edges = np.histogram(self.H, bins=256, range=(0, 256))
        elif channel == 'S':
            histogram, bin_edges = np.histogram(self.S, bins=256, range=(0, 256))
        elif channel == 'V':
            histogram, bin_edges = np.histogram(self.V, bins=256, range=(0, 256))
        elif channel == 'L':
            histogram, bin_edges = np.histogram(self.L, bins=256, range=(0, 256))
        elif channel == 'a':
            histogram, bin_edges = np.histogram(self.a, bins=256, range=(0, 256))
        elif channel == 'b':
            histogram, bin_edges = np.histogram(self.b, bins=256, range=(0, 256))
        elif channel == 'C':
            histogram, bin_edges = np.histogram(self.C, bins=256, range=(0, 256))
        elif channel == 'M':
            histogram, bin_edges = np.histogram(self.M, bins=256, range=(0, 256))
        elif channel == 'Y':
            histogram, bin_edges = np.histogram(self.Y, bins=256, range=(0, 256))
        elif channel == 'K':
            histogram, bin_edges = np.histogram(self.K, bins=256, range=(0, 256))

        histogram = histogram[1:]  # Delete the first element of the array

        return histogram

    def show_channels(self):
        plt.figure()
        plt.subplot(6, 2, 1), plt.imshow(self.RGB)
        plt.subplot(6, 2, 2), plt.imshow(self.H, cmap='gray')
        plt.subplot(6, 2, 3), plt.imshow(self.S, cmap='gray')
        plt.subplot(6, 2, 4), plt.imshow(self.V, cmap='gray')
        plt.subplot(6, 2, 5), plt.imshow(self.L, cmap='gray')
        plt.subplot(6, 2, 6), plt.imshow(self.a, cmap='gray')
        plt.subplot(6, 2, 7), plt.imshow(self.b, cmap='gray')
        plt.subplot(6, 2, 8), plt.imshow(self.C, cmap='gray')
        plt.subplot(6, 2, 9), plt.imshow(self.M, cmap='gray')
        plt.subplot(6, 2, 10), plt.imshow(self.Y, cmap='gray')
        plt.subplot(6, 2, 11), plt.imshow(self.K, cmap='gray')
        plt.show()

    def save(self, path):
        cv2.imwrite(path + '/' + 'RGB.png', self.RGB)
        cv2.imwrite(path + '/' + 'H.png', self.H)
        cv2.imwrite(path + '/' + 'S.png', self.S)
        cv2.imwrite(path + '/' + 'V.png', self.V)
        cv2.imwrite(path + '/' + 'L.png', self.L)
        cv2.imwrite(path + '/' + 'a.png', self.a)
        cv2.imwrite(path + '/' + 'b.png', self.b)
        cv2.imwrite(path + '/' + 'C.png', self.C)
        cv2.imwrite(path + '/' + 'M.png', self.M)
        cv2.imwrite(path + '/' + 'Y.png', self.Y)
        cv2.imwrite(path + '/' + 'K.png', self.K)


def dip(folder_path):
    # Images names
    '-------------'
    f = []
    for (dirpath, dirnames, filenames) in walk(folder_path):
        f.extend(filenames)
        break
    folderSave = 'DataResult/'
    i = 1
    for imagePath in f:
        print(imagePath)
        # Load image segmented
        '-------------------'
        img = mpimg.imread(folder_path + '/' + imagePath)
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Implemention of Tarik Method
        '-----------------------------'
        image = ColorSharpen(img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Segmentation of the Symptoms
        '-----------------------------'
        imSS = SymptomSegmentation(img)
        imSS_1 = SymptomSegmentation(image)

        mpimg.imsave(folderSave + str(i) + '.png', imSS.img)
        mpimg.imsave(folderSave + str(i) + str(i) + '.png', imSS_1.img)

        # Split and Get 10 Different Channel of Colour and its histogram
        '---------------------------------------------'
        imCT = ColourTransformation(imSS.img)
        imCT_1 = ColourTransformation(imSS_1.img)

        # Plot Histogram
        '---------------'
        histogram = imCT.histogram('Gray')
        histogram_1 = imCT_1.histogram('Gray')
        # configure and draw the histogram figure
        plt.figure()
        plt.title("Grayscale Histogram")
        plt.xlabel("grayscale value")
        plt.ylabel("pixels")
        plt.xlim([0.0, len(histogram)])
        plt.ylim([min(histogram), max(histogram)])
        plt.plot(histogram, 'r', label='Original ')
        plt.plot(histogram_1, 'b', label='Ancuti Method')
        plt.legend()
        plt.savefig(folderSave + 'histogram_' + str(i) + '.png')

        print(imagePath + ', ' + str(sum(histogram)) + ', ' + str(sum(histogram_1)))
        i += 1

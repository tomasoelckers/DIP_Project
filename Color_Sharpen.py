import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.filters import median_filter

'''
Step 1: Color Balancing.
	* Gray-World Algorithm.
	* Adjust single-color channels:
		1) Red: Irc(x) = Ir(x) + a(Ig'-Ir')(1-Ir(x))Ig(x); Ic' denotes the average of channel c, a=1
		2) Blue: Ibc(x) = Ib(x) + a(Ig'-Ib')(1-Ib(x))Ig(x); a=1
Step 2: Gamma Correction (Always normalizing).
Step 3: Sharpening (Always normalizing).
	1) Gaussian blur filter convolutioned with original white-balanced image. (Magnifies high-frequency noise)
	2) Histogram stretching or equalization.
	2) Normalized Unsharp Masking.

'''


# =====================
# Gray-World Algorithm
# =====================
def GrayWorld(Img):
	Ilab = cv2.cvtColor(Img, cv2.COLOR_BGR2LAB)
	avg_a = np.average(Ilab[:, :, 1])
	avg_b = np.average(Ilab[:, :, 2])
	Ilab[:, :, 1] = Ilab[:, :, 1] - (
				(avg_a - 128) * (Ilab[:, :, 0] / 255.0) * 1.1)  # Adjust for intern cv2 LAB adjustments.
	Ilab[:, :, 2] = Ilab[:, :, 2] - (
				(avg_b - 128) * (Ilab[:, :, 0] / 255.0) * 1.1)  # Adjust for intern cv2 LAB adjustments.
	I = cv2.cvtColor(Ilab, cv2.COLOR_LAB2BGR)
	return I  # Return a BGR image.


# ===============================
# Single Color Channel Balancing
# ===============================
def Color_adj(Img, a=1):  # Receives an uint8 image
	# info = np.iinfo(Img.dtype) #Format information of Img.
	# Imax = info.max
	# Inorm = I.astype(np.float64)/info.max #Normalize I
	Imax = Img.max()
	Inorm = cv2.normalize(Img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
	b, g, r = cv2.split(Inorm)  # Split the color channels individually.
	b_avg = np.average(b)
	g_avg = np.average(g)
	r_avg = np.average(r)
	r_adj = r + a * (g_avg - r_avg) * (1 - r) * g  # Red channel adjustment.
	g_adj = g + a * (r_avg - g_avg) * (1 - g) * r  # Green channel adjustment.
	b_adj = b + a * (g_avg - b_avg) * (1 - b) * g  # Blue channel adjustment.
	Iout = cv2.merge(
		(b * Imax, g_adj * Imax, r_adj * Imax))  # Reinstate the color space with the color balance adjustments.
	Iout = Iout.astype(np.uint8)
	return Iout


# ===========
# Sharpening
# ===========
def Gamma(Img, gamma = 1):
	invGamma = 1 / gamma
	table = np.array([((i/255)**invGamma)*255
					  for i in np.arange(0,256)]).astype(np.uint8)
	return cv2.LUT(Img, table)


def CLAHE(Img, clipLimit=1, tileGridSize=(8,8)):  # Contrast Limiting Adaptive Histogram Equalization (aka CLAHE)
	#clahe = cv2.createCLAHE()
	#Img = clahe.apply(Img)
	lab = cv2.cvtColor(Img, cv2.COLOR_BGR2LAB)
	lab_planes = cv2.split(lab)
	clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(5,5))
	lab_planes[0] = clahe.apply(lab_planes[0])
	lab = cv2.merge(lab_planes)
	Img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
	#Img = cv2.convertScaleAbs(Img, None, 1, -20) #Adjust global contrast and brightness.
	return Img


def unsharp(Iin, sigma, strength): #Unsharp. Create a blurry version of the original and substract it from the original. Assumes Img is not RGB, but one color channel.
	MF = median_filter(Iin, sigma)
	#GF = cv2.GaussianBlur(Iin, (5,5), 0)
	LP = cv2.Laplacian(MF, cv2.CV_64F)
	#N = CLAHE((Iin-GF)) #Histogram Equalization.
	#Iin = cv2.normalize(Iin, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_64F)
	sharp = Iin - strength * LP #Actual substraction.
	#sharp = (Iin + N) / 2 #Sharpen method from Ancuti's paper.
	#Inorm = cv2.normalize(sharp, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_64F)
	sharp[sharp>255] = 255 #Avoid oversaturation.
	sharp[sharp<0] = 0 #Avoid undersaturation.
	#sharp = sharp.astype(np.uint8) #Return it as uint8 type.
	return sharp


def Sharpen(Img):
	Isharp = np.zeros_like(Img) #Empty array of same dimensions of input image.
	for i in range(2): #Apply unsharping to each color channel (b, g, r).
		Isharp[:, :, i] = unsharp(Img[:,:,i], 10, 0.8)
	Isharp = Isharp.astype(np.uint8) #To return it to image type.
	return Isharp

def ImgShow(Original, New):
	Orginal = cv2.cvtColor(Original, cv2.COLOR_BGR2RGB)
	New = cv2.cvtColor(New, cv2.COLOR_BGR2RGB)
	plt.figure()
	plt.subplot(1, 2, 1)
	plt.imshow(Orginal)
	plt.title("Original")
	plt.subplot(1, 2, 2)
	plt.imshow(New)
	plt.title("Sharpened")
	plt.show()

def Run(Img):
	#Ig = Gamma(Img, 0.4)
	#Ig = cv2.convertScaleAbs(Ig, None, 1.2, 0)
	Iwb = GrayWorld(Img) #White balance.
	Ica = Color_adj(Iwb) #Color balance.
	#Ica = cv2.convertScaleAbs(Ica, None, 0.9, -15)
	Ig = Gamma(Img, 0.85)
	Ihe = CLAHE(Ig, clipLimit=1.5, tileGridSize=(20,20))
	#Ihe = cv2.convertScaleAbs(Ihe, None, 0.7, 0)
	#Isharp = Sharpen(Ica) #Sharpen
	return cv2.cvtColor(Ihe, cv2.COLOR_BGR2RGB)

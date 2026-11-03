import cv2
import numpy as np
from utils import erode, dilate

def apply_prewitt(img):

    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    img_prewittx = np.abs(cv2.filter2D(img.astype(float), -1, kernelx))
    img_prewitty = np.abs(cv2.filter2D(img.astype(float), -1, kernely))

    return np.sqrt(img_prewittx**2 + img_prewitty**2)


def apply_sobel(img, k=5):
    
    sobel_x = cv2.Sobel(img, cv2.CV_8U,1,0,ksize=k)
    sobel_y = cv2.Sobel(img, cv2.CV_8U,1,0,ksize=k)
    return sobel_x+sobel_y


def gaussian_kernel_2D(sigma, size):

	H,W = size

	mu_x = W//2
	mu_y = H//2

	xx, yy = np.meshgrid(np.arange(W), np.arange(H))

	exponent = .5*np.square((xx-mu_x)/sigma) + .5*np.square((yy-mu_y)/sigma)
	return np.exp(-exponent)


def freq_filter(img, sigma, mode, k_blr=10):

	# img = cv2.GaussianBlur(img, (5,5), 0)

	# clahe = cv2.createCLAHE(clipLimit=5)
	# img = clahe.apply(img)
	# img = cv2.bilateralFilter(img, k_blr, k_blr * 2, k_blr / 2)


	filt = gaussian_kernel_2D(sigma, img.shape)

	fft_img = np.fft.fft2(img)
	fft_img = np.fft.fftshift(fft_img)

	if mode=="HP":
		filt = (1-filt)
	elif mode=="LP":
		pass
	else:
		raise ValueError(f"Filtering mode \"{mode}\" not recognized.")
	
	fft_img = np.fft.ifftshift(fft_img*filt)
	filt_img = np.abs(np.fft.ifft2(fft_img))

	log_filt_img = np.log(filt_img+1)
	filt_img_bin = log_filt_img>(np.mean(log_filt_img)+np.std(log_filt_img))

	filt_img_bin = erode(filt_img_bin.astype('uint8'), k=5)
	
	return filt_img, filt_img_bin
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def rgblcc(img):
	
	I = cv.normalize(img.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
	Red = I[:,:,0]
	Green = I[:,:,1]
	Blue = I[:,:,2]
	DFT2d_Red = np.fft.fft2(Red)
	DFT2d_Green = np.fft.fft2(Green)
	DFT2d_Blue = np.fft.fft2(Blue)
	[M, N, C] = I.shape
	# Calculate the gaussian 
	sigma = 2
	Nr = np.fft.ifftshift(list(range(-int(M*0.5),int(M*0.5))))
	Nc = np.fft.ifftshift(list(range(-int(N*0.5),int(N*0.5))))
	[Nc, Nr] = np.meshgrid(Nc, Nr)
	dft_gauss_kernel = np.exp(-2*np.pi**2*sigma**2*((Nc/M)*(Nc/M) + (Nr/N)*(Nr/N)))
	dft_Red_convolved = DFT2d_Red*dft_gauss_kernel
	dft_Green_convolved = DFT2d_Green*dft_gauss_kernel
	dft_Blue_convolved = DFT2d_Blue*dft_gauss_kernel
	Red_convolved = np.fft.ifft2(dft_Red_convolved)
	Green_convolved = np.fft.ifft2(dft_Green_convolved)
	Blue_convolved = np.fft.ifft2(dft_Blue_convolved)
	MRed = np.real(Red_convolved)
	MGreen = np.real(Green_convolved)
	MBlue = np.real(Blue_convolved)
	output = np.zeros((M,N,3))
	for i in range(M):
		for j in range(N):
			output[i][j][0] = Red[i][j]**(2**(2*MRed[i][j]-1))
			output[i][j][1] = Green[i][j]**(2**(2*MGreen[i][j]-1))
			output[i][j][2] = Blue[i][j]**(2**(2*MBlue[i][j]-1))
	fig,a = plt.subplots(2,2)
	a[0][0].imshow(I)
	a[0][0].set_title('Original Image')
	a[0][1].imshow(output)
	a[0][1].set_title('Resultant Image')
	I = np.mean(I,2)
	I = I.flatten()
	output = np.mean(output,2)
	output = output.flatten()
	a[1][0].hist(I,255)
	a[1][0].set_title('input histogram')
	a[1][1].hist(output,255)
	a[1][1].set_title('output histogram')
	plt.show()

if __name__ == '__main__':
	img = cv.imread('cloudy.jpeg')
	img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
	result = rgblcc(img)
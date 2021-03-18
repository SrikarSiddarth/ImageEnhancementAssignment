import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# choice applies two different techniques.
# choice=1 applies "Normalized images" method.
# choice=0 applies "Un-normalized images" method.
def localColorCorrection(img,choice):
	sigma = 2
	if choice:
		# simple normalization the image 
		# I = img.astype('float')/255.0
		# inbuilt normalization function
		I = cv.normalize(img.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
		I = np.mean(I,2)
		DFT2d_I = np.fft.fft2(I)
	else:
		# taking the inverse of the intensity values
		I = img.astype('float')
		I = np.mean(I,2)
		Inv = 255-I
		DFT2d_I = np.fft.fft2(Inv)

	
	[ysize,xsize] = I.shape
	Nr = np.fft.ifftshift(list(range(-int(ysize*0.5),int(ysize*0.5))))
	Nc = np.fft.ifftshift(list(range(-int(xsize*0.5),int(xsize*0.5))))
	[Nc, Nr] = np.meshgrid(Nc, Nr)
	dft_gauss_kernel = np.exp(-2*np.pi**2*sigma**2*((Nc/xsize)*(Nc/xsize) + (Nr/ysize)*(Nr/ysize)))
	dft_I_convolved = DFT2d_I*dft_gauss_kernel
	I_convolved = np.fft.ifft2(dft_I_convolved)
	M = np.real(I_convolved)
	output = np.zeros((ysize,xsize))
	if choice:
		for i in range(ysize):
			for j in range(xsize):
				output[i][j] = I[i][j]**(2**(2*M[i][j]-1))
	else:
		for i in range(ysize):
			for j in range(xsize):
				output[i][j] = 255.0*((I[i][j]/255.0)**(2**((128-M[i][j])/128.0)))

	fig,a = plt.subplots(2,2)
	a[0][0].imshow(I/255,cmap='gray')
	a[0][0].set_title('Original Image')
	a[0][1].imshow(output/255,cmap='gray')
	a[0][1].set_title('Resultant Image')
	I = I.flatten()
	output = output.flatten()
	a[1][0].hist(I,255)
	a[1][0].set_title('input histogram')
	a[1][1].hist(output,255)
	a[1][1].set_title('output histogram')
	plt.show()

if __name__ == '__main__':
	img = cv.imread('cloudy.jpeg')
	img = c.cvtColor(img,cv.COLOR_BGR2RGB)
	result = localColorCorrection(img,0)



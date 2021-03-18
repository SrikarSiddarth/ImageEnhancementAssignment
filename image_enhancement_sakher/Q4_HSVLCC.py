import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as color


def hsvlcc(img):
	
	I = cv.normalize(img.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
	HSV = color.rgb_to_hsv(I)

	Value = HSV[:,:,2]
	DFT2d_Value = np.fft.fft2(Value)

	[M, N] = Value.shape
	# Calculate the gaussian 
	sigma = 4
	Nr = np.fft.ifftshift(list(range(-int(M*0.5),int(M*0.5))))
	Nc = np.fft.ifftshift(list(range(-int(N*0.5),int(N*0.5))))
	[Nc, Nr] = np.meshgrid(Nc, Nr)
	dft_gauss_kernel = np.exp(-2*(np.pi**2)*(sigma**2.0)*((Nc/float(N))*(Nc/float(N)) + (Nr/float(M))*(Nr/float(M))))
	dft_Value_convolved = DFT2d_Value*dft_gauss_kernel

	Value_convolved = np.fft.ifft2(dft_Value_convolved)

	MValue = np.real(Value_convolved)

	output = np.zeros((M,N,3))
	output[:,:,0] = HSV[:,:,0]
	output[:,:,1] = HSV[:,:,1]
	for i in range(M):
		for j in range(N):
			output[i][j][2] = Value[i][j]**(2**(2*MValue[i][j]-1))
	
	output = color.hsv_to_rgb(output)



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
	result = hsvlcc(img)
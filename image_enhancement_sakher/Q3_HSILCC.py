import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def hsilcc(img):
	
	F = cv.normalize(img.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
	r = F[:,:,0]
	g = F[:,:,1]
	b = F[:,:,2]

	# from rgb to hsi
	# calculate the hue
	th = np.arccos(0.5*((r-g)+(r-b))/(np.sqrt((r-g)**2 + (r-b)*(g-b) )+ np.spacing(1)))
	H = th
	H[b>g] = 2*np.pi - H[b>g]
	H = H/(2*np.pi)

	# calculate the saturation
	S = 1 - 3*(np.minimum(np.minimum(r,g),b))/(r+g+b+np.spacing(1))

	# Calculate the Intensity (mean of the colours)
	I = (r+g+b)/3
	# print(I[0,0])
	# print(H.shape, S.shape, I.shape)
	hsi = np.stack((H,S,I),axis=2)
	value = hsi[:,:,2]
	DFT2d_Value = np.fft.fft2(value)

	[M, N] = value.shape
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
	output[:,:,0] = hsi[:,:,0]
	output[:,:,1] = hsi[:,:,1]
	for i in range(M):
		for j in range(N):
			output[i][j][2] = value[i][j]**(2**(2*MValue[i][j]-1))
	# print(MValue[0][0])

	# from hsi to rgb
	HV = output[:,:,0]*2*np.pi
	SV = output[:,:,1]
	IV = output[:,:,2]
	R = np.zeros(HV.shape)
	G = np.zeros(HV.shape)
	B = np.zeros(HV.shape)
	# redgreen sector
	index = np.where((0<=HV) & (HV<2*np.pi/3))
	# index = np.array([i (0<=HV) and (HV<2*np.pi/3) for i in range])
	B[index] = IV[index]*(1-SV[index])
	R[index] = IV[index]*(1+SV[index]*np.cos(HV[index])/np.cos(np.pi/3 - HV[index]))
	G[index] = 3*IV[index] - R[index] - B[index]

	# BlueRed sector

	index = np.where((2*np.pi/3<=HV) & (HV<4*np.pi/3))
	R[index] = IV[index]*(1-SV[index])
	G[index] = IV[index]*(1+SV[index]*np.cos(HV[index]-2*np.pi/3)/np.cos(np.pi - HV[index]))
	B[index] = 3*IV[index] - R[index] - G[index]

	# BlueGreen Sector
	index = np.where((4*np.pi/3<=HV) & (HV<2*np.pi))
	G[index] = IV[index]*(1-SV[index])
	B[index] = IV[index]*(1+SV[index]*np.cos(HV[index]-4*np.pi/3)/np.cos(5*np.pi/3 - HV[index]))
	R[index] = 3*IV[index] - B[index] - G[index]

	C = np.stack((R,G,B),axis=2)
	C = np.maximum(np.minimum(C,1),0)



	fig,a = plt.subplots(2,2)
	a[0][0].imshow(F)
	a[0][0].set_title('Original Image')
	a[0][1].imshow(C)
	a[0][1].set_title('Resultant Image')
	F = np.mean(F,2)
	F = F.flatten()
	C = np.mean(C,2)
	C = C.flatten()
	a[1][0].hist(F,255)
	a[1][0].set_title('input histogram')
	a[1][1].hist(C,255)
	a[1][1].set_title('output histogram')
	plt.show()

if __name__ == '__main__':
	img = cv.imread('cloudy.jpeg')
	img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
	result = hsilcc(img)
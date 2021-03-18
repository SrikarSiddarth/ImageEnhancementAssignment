import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def vectorize(m):
	m = m.flatten('F')
	return np.reshape(m,(len(m),1))

def hsllcc(img):
	
	Image = cv.normalize(img.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
	# Find the minimum and maximum values of the image
	MinVal = np.ndarray.min(Image,2)		# or np.amin(Image,2) should work
	MaxVal = np.ndarray.max(Image,2)
	# Now calculate the Luminace value by adding the max and min values and divide by 2.
	Luminance = 0.5*(MinVal+MaxVal)
	temp = np.minimum(Luminance,1-Luminance)
	# Calculate the Saturation 
	saturation = 0.5*(MaxVal-MinVal)/(temp + (temp==0))
	i = np.argsort(Image,2)
	# print(i[0])
	x = Image.shape
	Matrice = np.zeros(x)
	for a in range(x[0]):
		for b in range(x[1]):
			for c in range(x[2]):
				Matrice[a,b,c] = Image[a,b,i[a,b,c]]
	i = i[:,:,2]
	Delta = Matrice[:,:,2]-Matrice[:,:,0]
	Delta += Delta==0
	Red = Image[:,:,0]
	Green = Image[:,:,1]
	Blue = Image[:,:,2]
	Hue = np.zeros(Red.shape)
	# If Red is max, then Hue = (G-B)/(max-min) 
	k = (i==0)
	Hue[k] = (Green[k]-Blue[k])/Delta[k]
	# If Green is max, then Hue = 2.0 + (B-R)/(max-min)
	k = (i==1)
	Hue[k] = 2 + (Blue[k]-Red[k])/Delta[k]
	# If Blue is max, then Hue = 4.0 + (R-G)/(max-min)
	k = (i==2)
	Hue[k] = 4 + (Red[k]-Green[k])/Delta[k]

	Hue = 60*Hue + 360*(Hue<0)
	Hue[Delta==0] = float('NaN')
	# Concatenating the Hue Saturation and Luminance
	Image[:,:,0] = Hue
	Image[:,:,1] = saturation
	Image[:,:,2] = Luminance
	HSL = Image
	# We take only the Luminance, Hue and saturation won't change
	Lum = HSL[:,:,2]
	DFT2d_Lum = np.fft.fft2(Lum)
	[M, N] = Lum.shape
	# Calculate the gaussian 
	sigma = 4
	Nr = np.fft.ifftshift(list(range(-int(M*0.5),int(M*0.5))))
	Nc = np.fft.ifftshift(list(range(-int(N*0.5),int(N*0.5))))
	[Nc, Nr] = np.meshgrid(Nc, Nr)
	dft_gauss_kernel = np.exp(-2*np.pi**2*sigma**2*((Nc/float(N))*(Nc/float(N)) + (Nr/float(M))*(Nr/float(M))))
	# Convolution of the Gaussian_fft and the Luminance_fft
	dft_Lum_convolved = DFT2d_Lum*dft_gauss_kernel
	# Calculate the inverse, and take the real values
	Lum_convolved = np.fft.ifft2(dft_Lum_convolved)
	MLum = np.real(Lum_convolved)
	# Application of the LCC formula
	output = np.zeros((M,N,3))
	for i in range(M):
		for j in range(N):
			output[i][j][2] = Lum[i][j]**(2**(2*MLum[i][j]-1))
	# Output will have the same Hue and Saturation, and we change only
	# the Luminance
	output[:,:,1]=HSL[:,:,1]
	output[:,:,0]=HSL[:,:,0]
	Luminance=output[:,:,2]
	Delta = Image[:,:,1]*np.minimum(Luminance,1-Luminance)
	m0 = Luminance - Delta
	m2 = Luminance + Delta
	tailleN = Hue.shape
	Hue = np.minimum(np.maximum(vectorize(Hue),0),360)/60
	m0 = vectorize(m0)
	m2 = vectorize(m2)
	F = Hue - np.round(Hue/2)*2
	Matrice = np.append(np.append(m0,m0+(m2-m0)*np.abs(F)),m2)
	Matrice = np.reshape(Matrice,(len(Matrice),1))
	# print(Matrice.shape)
	num = len(m0)
	j = np.array([[2,1,0],[1,2,0],[0,2,1],[0,1,2],[1,0,2],[2,0,1],[2,1,0]])*num
	k = (np.floor(Hue)).astype(int)			# ommiting +1 because python follows 0 based indexing
	x = np.arange(1,num+1).T
	x = np.reshape(x,(len(x),1))
	y = np.append(np.append(Matrice[j[k,0]+x-1],Matrice[j[k,1]+x-1],axis=1),Matrice[j[k,2]+x-1],axis=1)
	# matlab reshaping is different from that of regular numpy reshape() therefore we should add an argument order='F' to it.
	Image = y.reshape([tailleN[0],tailleN[1],3],order='F')

	fig,a = plt.subplots(2,2)
	a[0][0].imshow(img)
	a[0][0].set_title('Original Image')
	a[0][1].imshow(Image)
	a[0][1].set_title('Resultant Image')
	img = np.mean(img,2)
	img = img.flatten()
	Image = np.mean(Image,2)
	Image = Image.flatten()
	a[1][0].hist(img,255)
	a[1][0].set_title('input histogram')
	a[1][1].hist(Image,255)
	a[1][1].set_title('output histogram')
	plt.show()

if __name__ == '__main__':
	img = cv.imread('cloudy.jpeg')
	img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
	result = hsllcc(img)
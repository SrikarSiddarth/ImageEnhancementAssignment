import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def SimplestColorBalance(imageData,s):
	# Get the number of pixels
	# print(imageData.shape)
	pageSize = imageData.shape[0]*imageData.shape[1]
	# Transforme the image into an array
	vecteur_nt = np.copy(np.reshape(imageData,[1,pageSize]))

	vecteur_t = np.sort(vecteur_nt)
	vecteur_t = vecteur_t.T
	vecteur_nt = vecteur_nt.T
	# y,z = np.where(np.isclose(vecteur_t, 0))
	# print(y)
	s1=s/2
	s2=s1
	# Get the position of the quantiles
	pos_v_min = (np.floor(pageSize*s1/100.0)).astype(int)
	pos_v_max = (np.floor(pageSize*(1-s2/100.0)-1)).astype(int)
	# Get the values of (quantiles)
	v_min=vecteur_t[pos_v_min-1]
	v_max=vecteur_t[pos_v_max-1]
	# print(v_max, v_min)
	# Replace the values that are greater than v_max with v_max,
	# and those which are smaller than v_min with v_min
	idx_min=vecteur_nt<v_min
	idx_max=vecteur_nt>v_max
	vecteur_nt[idx_min]=v_min
	vecteur_nt[idx_max]=v_max
	for i in range(pageSize):
		vecteur_nt[i]=((vecteur_nt[i]-v_min)*(255-0))/((v_max-v_min)+0);
	result = vecteur_nt.reshape(imageData.shape)
	return result

def periodique(imageData):
	[M,N]=imageData.shape
	# print(M,N)
	img_prd=np.zeros((M*2,N*2))
	img_prd[0:M-1,0:N-1] = imageData[0:M-1,0:N-1]
	img_prd[M:M*2-1,0:N-1]=imageData[M-1:0:-1,0:N-1]
	img_prd[0:M-1,N:N*2-1]=imageData[0:M-1,N-1:0:-1]
	img_prd[M:M*2-1,N:N*2-1]=imageData[M-1:0:-1,N-1:0:-1]
	return img_prd

# inputs are :    image,
                # s: the % of saturation exp 0.2
                # lambda, exp 0.0001
def poisson(img,s,l):
	imageData = cv.normalize(img.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
	[M,N,_] = imageData.shape
	colour = [None]*3
	img_sat = [None]*3
	img_prd = [None]*3
	fft = [None]*3
	fft_u = [None]*3
	u = [None]*3
	result = [None]*3
	for i in range(3):
		colour[i] = imageData[:,:,i]
		# print(colour[i].shape)
		# Saturation
		img_sat[i] = SimplestColorBalance(colour[i],s)
		# Periodisation
		img_prd[i] = periodique(img_sat[i])
		# ffts
		fft[i] = np.fft.fft2(img_prd[i])

		[J,L] = fft[0].shape
		fft_u[i] = np.zeros((J,L))

		for m in range(J):
			for n in range(L):
				fft_u[i][m,n]=(((np.pi*m/J)**2+(np.pi*n/L)**2)/(l+(np.pi*m/J)**2+(np.pi*n/L)**2))*fft[i][m,n]
		
		u[i] = np.real(np.fft.ifft2(fft_u[i]))
		# saturation
		result[i] = SimplestColorBalance(u[i][0:M-1,0:N-1],s)
	result = cv.merge((result[0],result[1],result[2]))
	fig,a = plt.subplots(2,2)
	a[0][0].imshow(imageData/255)
	a[0][0].set_title('Original Image')
	a[0][1].imshow(result/255)
	a[0][1].set_title('Resultant Image')
	imageData = imageData[:,:,0].flatten()
	output = result[:,:,0].flatten()
	a[1][0].hist(imageData,255)
	a[1][0].set_title('input histogram')
	a[1][1].hist(output,255)
	a[1][1].set_title('output histogram')
	plt.show()

if __name__ == '__main__':
	img = cv.imread('cloudy.jpeg')
	img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
	result = poisson(img,np.exp(0.2),np.exp(0.0001))
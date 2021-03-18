import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# def vectorize(m):
# 	m = m.flatten('F')
# 	return m[...,np.newaxis]

# inputs are :   image name
               # s : % of saturation, exp 0.2
def SimplestColorBalanceNVG(img,s):
	# imageData = cv.normalize(img.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
	imageData = np.mean(img,2)
	
	# Get the number of pixels
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
	fig,a = plt.subplots(2)
	a[0].imshow(imageData/255,cmap='gray')
	a[0].set_title('Original Image')
	a[1].imshow(result/255,cmap='gray')
	a[1].set_title('Ouput Image')
	plt.show()


if __name__ == '__main__':
	img = cv.imread('cloudy.jpeg')
	img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
	result = SimplestColorBalanceNVG(img,0.2)
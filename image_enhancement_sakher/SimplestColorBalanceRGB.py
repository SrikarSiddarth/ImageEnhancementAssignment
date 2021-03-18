import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# def vectorize(m):
# 	m = m.flatten('F')
# 	return m[...,np.newaxis]

# inputs are :   image name
               # s : % of saturation, exp 0.2
def SimplestColorBalanceRGB(img,s):
	imageData = cv.normalize(img.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
	# imageData = np.mean(img,2)
	# imageData = img/255.0
	# Get the number of pixels
	pageSize = imageData.shape[0]*imageData.shape[1]
	# Transforme the image into an array
	vecteur_nt = np.copy(np.reshape(imageData,[pageSize,3],order='F'))
	vecteur_t = np.sort(vecteur_nt.T,kind='heapsort')
	vecteur_t = vecteur_t.T
	amin = vecteur_t[0,:]
	amax = vecteur_t[pageSize-1,:]

	# Balance the %
	s1=s/2
	s2=s1
	# Get the position of the quantiles
	pos_v_min = (np.floor(pageSize*s1/100.0)).astype(int)
	pos_v_max = (np.floor(pageSize*(1-s2/100.0)-1)).astype(int)
	# Get the values of (quantiles)
	v_min=vecteur_t[pos_v_min-1,:]
	v_max=vecteur_t[pos_v_max-1,:]
	# print(v_max, v_min)
	# Replace the values that are greater than v_max with v_max,
	# and those which are smaller than v_min with v_min
	idx_min_r=vecteur_nt[:,0]<v_min[0]
	idx_min_g=vecteur_nt[:,1]<v_min[1]
	idx_min_b=vecteur_nt[:,2]<v_min[2]
	idx_max_r=vecteur_nt[:,0]>v_max[0]
	idx_max_g=vecteur_nt[:,1]>v_max[1]
	idx_max_b=vecteur_nt[:,2]>v_max[2]

	vecteur_nt[idx_min_r,0]=v_min[0]
	vecteur_nt[idx_min_g,1]=v_min[1]
	vecteur_nt[idx_min_b,2]=v_min[2]
	vecteur_nt[idx_max_r,0]=v_max[0]
	vecteur_nt[idx_max_g,1]=v_max[1]
	vecteur_nt[idx_max_b,2]=v_max[2]
	for c in range(3):
		for i in range(pageSize):
			vecteur_nt[i,c]=((vecteur_nt[i,c]-v_min[c])*(amax[c]-amin[c]))/((v_max[c]-v_min[c])+amin[c]);
	result = vecteur_nt.reshape(imageData.shape,order='F')
	# print(np.mean(imageData))
	# print(np.mean(vecteur_nt))
	# print(imageData.shape)
	fig,a = plt.subplots(2)
	a[0].imshow(imageData)
	a[0].set_title('Original Image')
	a[1].imshow(result)
	a[1].set_title('Ouput Image')
	plt.show()


if __name__ == '__main__':
	img = cv.imread('cloudy.jpeg')
	img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
	result = SimplestColorBalanceRGB(img,30)
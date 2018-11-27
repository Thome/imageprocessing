import matplotlib, matplotlib.pyplot as plt
import numpy as np

# shape = (number of lines, length of lines) (altura, largura)

def imread(filename):
	image = plt.imread(filename)
	if (image.dtype == 'float32'): #if image is .png
		image = (np.multiply(image,255)).astype('uint8')
	return image

def nchannels(image): #3 if RBG, 1 if grayscale
	if(len(image.shape)<3):
		return 1
	elif (len(image.shape)==3):
		return 3

def isgray(image):
	if(nchannels(image)==1):
		return true
	else:
		return false

def size(image): #(altura, largura) => (largura, altura)
	return [image.shape[1], image.shape[0]]

def rgb2gray(image):
	return np.dot(image, [0.299, 0.587, 0.114])

def imreadgray(filename):
	image = imread(filename)
	#if(nchannels(image)==3):
	if(NOT isgray(image)): #if rgb
		image = rgb2gray(image)
	return image

def imshow(image):
	if(isgray(image)):
		plt.imshow(image, cmap='gray')
	else:
		plt.imshow(image)
	plt.show()

def thresh(image, threshold):
	imagecopy = image.copy()
	imagecopy[imagecopy>=threshold] = 255
	imagecopy[imagecopy<threshold] = 0
	return imagecopy

def negative(image):
	return 255 - image

def contrast(f,r,m):
	return r * (f - m) + m

def hist(image):
	if(isgray(image)):
		histo = np.zeros(256)
		for linha in image:
			for pixel in linha:
				histo[pixel] += 1
	else: #if rgb
		histo = np.zeros((256,3),dtype='uint8')
		for linha in image:
			for pixel in linha:
				for color in range(3): #1=r,2=g,3=b
					pixelcor = pixel[color]
					histo[pixelcor][color] += 1
	return histo

def showhist1(histo):
	x = np.arange(256).astype('uint8')
	plt.xlabel('Intensidades')
	plt.ylabel('Frequencia')
	if(isgray(image)):
		plt.bar(x, height= histo, color='grey')
		plt.show()
	else:
		plt.bar(x, height= histo[:,0], color='red')
		plt.show()
		plt.bar(x, height= histo[:,1], color='green')
		plt.show()
		plt.bar(x, height= histo[:,2], color='blue')
		plt.show()

def showhist(histo,bin=1):
	heights = np.arange(0).astype('uint8')
	contador = 0
	plt.xlabel('Intensidades')
	plt.ylabel('Frequencia')
	if(histo.shape[1] == 1): #if histo.shape == (XL,1L) 
		while(contador<256):
			aux = 0
			for i in range(bin):
				if(contador+i<256):
					aux += histo[contador+i]
			heights = np.append(heights,aux)
			contador += bin
		x = np.arange(len(heights)).astype('uint8')
		plt.bar(x, height= heights, color= 'grey')
		plt.show()
	else:
		while(contador<256):
			aux = [0,0,0]
			for i in range(bin):
				if(contador+i<256):
					aux[0] += histo[:,0][contador+i]
					aux[1] += histo[:,1][contador+i]
					aux[2] += histo[:,2][contador+i]
			heights = np.append(heights,aux)
			contador += bin
		x = np.arange(len(heights)).astype('uint8')
		plt.bar(x, height= heights[:,0], color= 'red')
		plt.show()
		plt.bar(x, height= heights[:,1], color='green')
		plt.show()
		plt.bar(x, height= heights[:,2], color='blue')
		plt.show()

def histeq(image):
	histo = hist(image)
	if(isgray(image)):
		npixels = sum(histo)]
		fdp = []
		aux = 0
		for i in range(256):
			aux += histo[i]/npixels
			fdp = np.append(fdp, aux)
		return floor(255 * fdp)
	

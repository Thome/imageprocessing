import matplotlib, matplotlib.pyplot as plt
import numpy as np
import math

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
		return True
	else:
		return False

def size(image): #(altura, largura) => (largura, altura)
	return [image.shape[1], image.shape[0]]

def rgb2gray(image):
	return np.dot(image, [0.299, 0.587, 0.114])

def imreadgray(filename):
	image = imread(filename)
	if(not isgray(image)): #if rgb
		image = (rgb2gray(image)).astype('uint8')
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
		histo = (np.zeros(256)).astype(int)
		for linha in image:
			for pixel in linha:
				histo[pixel] += 1
	else: #if rgb
		histo = (np.zeros((256,3))).astype(int)
		for linha in image:
			for pixel in linha:
				for color in range(3): #1=r,2=g,3=b
					pixelcor = pixel[color]
					histo[pixelcor][color] += 1
	return histo

def altshowhist(histo):
	x = np.arange(256).astype('uint8')
	if(len(histo.shape)==1): #if histo.shape == (256L,)
		plt.xlabel('Intensidades')
		plt.ylabel('Frequencia')
		plt.bar(x, height= histo, color='grey')
		plt.show()
	else:					 #if histo.shape == (256L,3L)
		fig, ax = plt.subplots()
		index = np.arange(256)
		bar_width = 0.35
		opacity = 0.4
		error_config = {'ecolor': '0.3'}

		rects1 = ax.bar(index, histo[:,0], bar_width,
                alpha=opacity, color='r',
                error_kw=error_config,
                label='Red')
		rects2 = ax.bar(index+bar_width, histo[:,1], bar_width,
                alpha=opacity, color='g',
                error_kw=error_config,
                label='Green')
		rects3 = ax.bar(index+bar_width*2, histo[:,2], bar_width,
                alpha=opacity, color='b',
                error_kw=error_config,
                label='Blue')

		ax.set_xlabel('Intensidades')
		ax.set_ylabel('Frequencia')
		ax.legend()
		fig.tight_layout()
		plt.show()

def showhist(histo,Bin=1):
	contador = 0
	if(len(histo.shape) == 1): #if histo.shape == (256L,) 
		heights = np.arange(0).astype('uint8')
		while(contador<256):
			heights = np.append(heights,sum(histo[contador:contador+Bin]))
			contador += Bin
		x = np.arange(len(heights)).astype('uint8')
		plt.xlabel('Intensidades')
		plt.ylabel('Frequencia')
		plt.bar(x, height= heights, color= 'grey')
		plt.show()
	else:					   #if histo.shape == (256L,3L)

		while(contador<256):
			aux = np.array([[0,0,0]])
			aux[0,0] = sum(histo[:,0][contador:contador+Bin])
			aux[0,1] = sum(histo[:,1][contador:contador+Bin])
			aux[0,2] = sum(histo[:,2][contador:contador+Bin])
			if(contador == 0):
				heights = aux
			else:
				heights = np.append(heights,aux,0)
			contador += Bin

		fig, ax = plt.subplots()
		index = np.arange(len(heights))
		bar_width = 0.35
		opacity = 0.4
		error_config = {'ecolor': '0.3'}

		rects1 = ax.bar(index, heights[:,0], bar_width,
                alpha=opacity, color='r',
                error_kw=error_config,
                label='Red')
		rects2 = ax.bar(index+bar_width, heights[:,1], bar_width,
                alpha=opacity, color='g',
                error_kw=error_config,
                label='Green')
		rects3 = ax.bar(index+bar_width*2, heights[:,2], bar_width,
                alpha=opacity, color='b',
                error_kw=error_config,
                label='Blue')

		ax.set_xlabel('Bins')
		ax.set_ylabel('Intensidades')
		ax.legend()
		fig.tight_layout()
		plt.show()

def histeq(image):
	histo = hist(image)
	npixels = sum(histo)
	fdp = [float(p)/npixels for p in histo]
	H = [math.floor(255 * round(sum(i),8)) for i in [fdp[0:j] for j in range(1,len(fdp)+1)]]
	return np.array([[H[px] for px in row] for row in image], dtype='uint8')

def clamp(value, L):
	return min(max(value,0), L-1)

def convolve(image, mask):
	convolution = np.ndarray(image.shape, dtype='uint8')
	a = (mask.shape[0]-1)/2
	b = (mask.shape[1]-1)/2
	altura = image.shape[0]
	largura = image.shape[1]
	for x in range(altura):
		for y in range(largura):
			if(isgray(image)):
				soma = 0
			else:
				soma = [0,0,0]
			for s in range(-a,a+1):
				for t in range(-b,b+1):
					w = mask[s+1,t+1]
					f = image[clamp(x+s,altura),clamp(y+t,largura)]
					soma += w * f
			convolution[x,y] = soma
	return convolution

def maskBlur():
	buf = np.array([[1,2,1],[2,4,2],[1,2,1]],dtype='uint8')
	mask = np.ndarray(shape=(3,3),dtype='uint8',buffer=buf)
	return mask * 1.0/16

def blur(image):
	return convolve(image, maskBlur())

def seSquare3():
	return np.array([[1,1,1],[1,1,1],[1,1,1]],dtype='uint8')

def seCross3():
	return np.array([[0,1,0],[1,1,1],[0,1,0]],dtype='uint8')

def getmin(img,alt,lar,x,y,S):
	if(isgray(img)):
		return min([img[clamp(x+s[0],alt),clamp(y+s[1],lar)] for s in S])
	else:
		aux = np.array([img[clamp(x+s[0],alt),clamp(y+s[1],lar)] for s in S])
		return [min(aux[:,0]),min(aux[:,1]),min(aux[:,2])]

def erode(image, bin_elem):
	newimage = image.copy()
	tuplas = [(x,y) for x in range(-1,2) 
				for y in range(-1,2)
				if bin_elem[x+1][y+1] == 1]
	altura = image.shape[0]
	largura = image.shape[1]
	for i in range(altura):
		for j in range(largura):
			newimage[i,j] = getmin(image,altura,largura,i,j,tuplas)
	return newimage

def getmax(img,alt,lar,x,y,S):
	if(isgray(img)):
		return max([img[clamp(x+s[0],alt),clamp(y+s[1],lar)] for s in S])
	else:
		aux = np.array([img[clamp(x+s[0],alt),clamp(y+s[1],lar)] for s in S])
		return [max(aux[:,0]),max(aux[:,1]),max(aux[:,2])]

def dilate(image, bin_elem):
	newimage = image.copy()
	tuplas = [(x,y) for x in range(-1,2)
				for y in range(-1,2)
				if bin_elem[x+1][y+1] == 1]
	altura = image.shape[0]
	largura = image.shape[1]
	for i in range(altura):
		for j in range(largura):
			newimage[i,j] = getmax(image,altura,largura,i,j,tuplas)
	return newimage

import matplotlib, matplotlib.pyplot as plt
import numpy as np

# shape = (number of lines, length of lines) (altura, largura)
'''def xyz(img,alt,lar,x,y,S):
	return min([img[clamp(x+s[0],alt),clamp(y+s[1],lar)] for s in S])
'''
#[(x,y) for x in range(-1,2) if wc[x+1][y+1] == 1 for y in range(-1,2)]
#[item for sublist in a for item in sublist]
#

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
		npixels = sum(histo)
		fdp = []
		aux = 0
		for i in range(256):
			aux += histo[i]/npixels
			fdp = np.append(fdp, aux)
		return floor(255 * fdp)
	else:
		nrpixels = sum(histo[:,0])
		ngpixels = sum(histo[:,1])
		nbpixels = sum(histo[:,2])
		fdp = []
		aux = [0,0,0]
		for i in range(256):
			aux[0] += histo[:,0][i]/nrpixels
			aux[1] += histo[:,1][i]/ngpixels
			aux[2] += histo[:,2][i]/nbpixels
			fdp = np.append(fdp, aux)
		return floor(255 * fdp)

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

def erode(image, bin_elem):
	newimage = image.copy()
	a = (bin_elem.shape[0]-1)/2
	b = (bin_elem.shape[1]-1)/2
	altura = image.shape[0]
	largura = image.shape[1]
	for i in range(altura):
		for j in range(largura):
			if(isgray(newimage)):
				menor = 256
			else:
				menor = [256,256,256]
			for s in range(-a,a+1):
				for t in range(-b,b+1):
					w = bin_elem[s+1,t+1]
					if(w):
						f = image[clamp(i+s,altura),clamp(j+t,largura)]
						if(isgray(newimage)):
							menor = min(menor, f)
						else:
							menor = np.minimum(menor, f)
			newimage[i,j] = menor
	return newimage

def dilate(image, bin_elem):
	newimage = image.copy()
	a = (bin_elem.shape[0]-1)/2
	b = (bin_elem.shape[1]-1)/2
	altura = image.shape[0]
	largura = image.shape[1]
	for i in range(altura):
		for j in range(largura):
			if(isgray(newimage)):
				maior = -1
			else:
				maior = [-1,-1,-1]
			for s in range(-a,a+1):
				for t in range(-b,b+1):
					w = bin_elem[s+1,t+1]
					if(w):
						f = image[clamp(i+s,altura),clamp(j+t,largura)]
						if(isgray(newimage)):
							maior = max(maior, f)
						else:
							maior = np.maximum(maior, f)
			newimage[i,j] = maior
	return newimage

from operator import add
from pilib import pilib as pi
from collections import deque
import numpy as np
import math

#Aux
def bin_img(thresh):
	img = pi.imreadgray('mario.jpg')
	return pi.thresh(img, thresh)

def imgtest(B):
	h,w = B.shape
	for i in range(h):
		for j in range(w):
			if (i==j or i==w/2):
				B[i,j] = 255
			else:
				B[i,j] = 0
	return(B)

def intersec(A,B):
	h,w = A.shape
	inter = A==B
	C = np.zeros([h,w], dtype=np.uint8)
	for i in range(h):
		for j in range(w):
			C[i,j] = A[i,j] if inter[i,j] else 0
	return np.array(C)

def union(A,B):
	h,w = A.shape
	C = np.zeros([h,w], dtype=np.uint8)
	for i in range(h):
		for j in range(w):
			C[i,j] = max(A[i,j],B[i,j])
	return np.array(C)

#========================================================================================================#	

#Q1
def adjacencia(n):
	if(n == 4):
		return [[-1,0],[0,-1],[0,1],[1,0]]
	elif(n==8):
		return [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]]

def obtervizinhos(coordenada, adjacentes):
	vizinhos = [list(map(add,coordenada,vizinho)) for vizinho in adjacentes]
	return vizinhos

def v(img,nl_nc,i,j):
	if i >= nl_nc[0] or i < 0 or j >= nl_nc[1] or j < 0:
		return -1
	return img[i,j]

def rotular_img(imagem, n):
	img = imagem.copy()
	rotulo_atual = 1
	pilha = deque([])
	nl_nc = pi.size(img)
	n_linhas = nl_nc[0]
	n_colunas = nl_nc[1]
	adjacentes = adjacencia(n)
	for i in range(n_linhas):
		for j in range(n_colunas):
			if img[i,j] == 255:
				pilha.append((i,j))
				img[i,j] = rotulo_atual
				while pilha:
					coords = pilha.popleft()
					vizinhos = obtervizinhos(coords,adjacentes)
					for pixel in vizinhos:
						if v(img,nl_nc,pixel[0],pixel[1]) == 255:
							img[pixel[0],pixel[1]] = rotulo_atual
							pilha.append(pixel)
				rotulo_atual += 1
	return np.array(img, np.uint8)


#Q2
def rotToRgb(img):
	res = np.array([[[(pix*59)%255, (pix*73)%255, (pix*83)%255] for pix in row]
																for row in img], np.uint8)
	return res

def gera_img(b_img, n_adj):
	img = rotular_img(b_img, n_adj)
	return rotToRgb(img)

#Q3
def morf_grad(I, N, B):
	if N == 1:
		img = I - pi.erode(I, B)
	if N == 2:
		img = pi.dilate(I, B) - I
	if N == 3:
		img = pi.dilate(I, B) - pi.erode(I, B)
	return np.array(img, np.uint8)

#Q4
def cond_dilate(I, M, B):
	img = pi.dilate(I, B)
	i = 0
	print(img)
	for i in range(pi.size(img)[0]):
		intersec = img[i] == M[i]
		novalinha = [x if y else 0 for x,y in zip(intersec,img[i])]
		img[i] = novalinha
	return img

#Q5
def img_dif(A, B):
	return np.array_equal(A, B)

#Q6
def ext_comp(A, Y, elem):
	return 0

#Q7
def complem(B):
	return np.array([[-1 if x==-1 else (x+1)%2 for x in row] for row in B])

#Q8
def rotate(B, n):
	n = n % 8
	while(n):
		B = nr(B)
		n -= 1
	return B

def nr(B):
	h,w = B.shape
	img = np.zeros([h,w], dtype=np.uint8)
	for i in range(h):
		for j in range(w):
			if i==j:
				img[i,j] = B[w/2,j]
			elif i==w/2:
				img[i,j] = B[w-1-j,j]
			elif i+j==w-1:
				img[i,j] = B[i,w/2]
			elif j==w/2:
				img[i,j] = B[i,i]
			else:
				d = int(math.sqrt( (h/2 - i)**2 + (w/2 - j)**2))
				if (j>w/2 and i+j<w-1):
					img[i,j] = B[i,j-d]
				elif (i<j and i>w/2):
					img[i,j] = B[i-d,j]
				elif (j<w/2 and i+j>w-1):
					img[i,j] = B[i,j+d]
				elif (i>j and i<w/2):
					img[i,j] = B[i+d,j]
				elif (i+j>w-1 and i<w/2):
					img[i,j] = B[-j+w-1,i+d]
				elif (i<j and j<w/2):
					img[i,j] = B[-j+w-1-d,i]
				elif (i+j<w-1 and i>w/2):
					img[i,j] = B[-j+w-1,i-d]
				elif (i>j and j>w/2):
					img[i,j] = B[-j+w-1+d,i]
	return img

#Q9
def hitmiss(A, B):
	return intersec(pi.erode(A,B),pi.erode(complem(A),complem(B)))

def rotmiss(I, B, nList):
	h,w = I.shape
	U = np.zeros([h,w], dtype=np.uint8)
	#for n in nList:
	return 0




'''
a = np.array([[1,1,1],[0,0,0],[0,0,0]])
b = np.array([[1,1,0],[1,0,0],[1,1,1]])
print(union(a,b))

b = np.array([[0,0,-1],[0,1,1],[-1,1,-1]])
a = np.zeros([5,5], dtype=np.uint8)
a[1] = [0,1,1,1,0]
a[2] = [0,1,1,1,0]
a[3] = [0,1,1,1,0]
print(hitmiss(a,b))'''
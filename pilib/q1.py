from operator import add
from pilib import pilib as pi
from collections import deque

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
		return img[i,j]
	return -1

def rotular_img_binaria(imagem, n):
	rotulo_atual = 1
	pilha = deque([])
	nl_nc = size(imagem)
	n_linhas = nl_nc[0]
	n_colunas = nl_nc[1]
	adjacentes = adjacencia(n)
	for i in range(n_linhas):
		for j in range(n_colunas):
			if imagem[i,j] == 255:
				imagem[i,j] = rotulo_atual
				pilha.append((i,j))
				while(pilha.size()):
					coords = pilha.dequeue()
					vizinhos = obtervizinhos(coords,adjacentes)
					for pixel in vizinhos:
						if v(imagem,nl_nc,pixel[0],pixel[1]) == 255:
							imagem[pixel] = rotulo_atual
							pilha.append(pixel)
				rotulo_atual = rotulo_atual + 1
	return imagem

adj = adjacencia(8)
v = obtervizinhos([2,2],adj)
print(v)

a = pi.imread("shirt.png")
pi.imshow(a)

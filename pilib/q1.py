from operator import add

def adjacencia(n):
	if(n == 4):
		return [[-1,0],[0,-1],[0,1],[1,0]]
	else if(n==8):
		return [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]]

def obtervizinhos(coordenada, adjacentes):
	vizinhos = [list(map(add,coordenada,vizinho)) for vizinho in adjacentes]

def rotular_img_binaria(imagem, n):
	rotulo_atual = 1
	pilha = []
	img_num_linhas = size(imagem)[0]
	img_num_colunas = size(imagem)[1]
	adjacentes = adjacencia(n)
	for i in range(imagem_numero_de_linhas):
		for j in range(imagem_numero_de_colunas):
			if imagem[i,j] == 255:
				imagem[i,j] = rotulo_atual
				pilha.append([i,j])
				while(pilha.len() > 0):
					coords = pilha.pop()
					vizinhos = obtervizinhos(coords,adjacentes)
					for pixel in vizinhos:
						if pixel == 255:
							pixel = rotulo_atual
							pilha.append([i,j])
				rotulo_atual = rotulo_atual + 1

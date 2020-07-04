import numpy as np
from PIL import Image as im 
import pyopencl as cl
import math
import time

# Função que gera o kernel quadrado de ordem tam_kernel.
def kernel_peso(variancia,tam_kernel):
    soma_peso = 0
    matriz = np.zeros(shape=(tam_kernel,tam_kernel))

    # Para cada elemento é aplicada a função de densidade gaussiana
    for i in range(tam_kernel):
        for j in range(tam_kernel):
            # Os valores x e y devem ser focados no centro do kernel, por isso temos o deslocamento pela metade do tam_kernel
            matriz[i][j] = gaussian_func(i-(tam_kernel//2),j-(tam_kernel//2),variancia)
            soma_peso += matriz[i][j]

    # Normalização do kernel - Prevenindo anormalidades na iluminação da imagem
    for i in range(tam_kernel):
        for j in range(tam_kernel):
            matriz[i][j] /=soma_peso
    
    return matriz  

# Função que realiza a aplicação da função de densidade (gaussiana) de duas dimensões
def gaussian_func(x,y,variancia):
    return pow(math.e,(-(pow(x,2)+pow(y,2)))/(2*pow(variancia,2)))/(2*math.pi*pow(variancia,2))

# Função que realiza o desfoque gaussiano a partir da operação de convolução entre o kernel e a imagem
def gaussian_blur(kernel,img):
    tempo_inicio = time.time()
    # Dimensões das matrizes
    dim_x = img.shape[0]
    dim_y = img.shape[1]
    dim_xi = kernel.shape[0]
    dim_yi = kernel.shape[1]
    output = np.zeros(img.shape, dtype=np.uint8)

    # OPERAÇÃO DE CONVOLUÇÃO
    for i in range(dim_x):
        for j in range(dim_y):
            conv_soma = 0
            for ii in range(dim_xi):
                for ji in range(dim_yi):
                    # Para lidar com os limites da imagem, foi optado por fazer a convolução apenas com elementos do kernel
                    # que estão no mesmo espaço da matriz da imagem.
                    if((ii-(dim_xi//2)+i) < dim_x and (ji-(dim_yi//2)+j) < dim_y and (ii-(dim_xi//2)+i) >=0 and (ji-(dim_yi//2)+j)>=0):
                        conv_soma += kernel[ii][ji]*img[(ii-(dim_xi//2)+i)][(ji-(dim_yi//2)+j)]
            output[i][j] = conv_soma
    print("O tempo decorrido foi de: %s segundos" % (time.time() - tempo_inicio ) ) 
    return output

# Carregando a imagem em níveis de cinza
img_original = im.open("ney.jpg")
img_original = img_original.convert('L')
img_original = np.asarray(img_original, dtype = np.float32)

img_desfocada =  img_original.copy()

kernel = kernel_peso(1,7)

for i in range(3):
    img_desfocada = gaussian_blur(kernel,img_desfocada)


img_desfocada = im.fromarray(img_desfocada)
img_desfocada.show()

import numpy as np
from PIL import Image as im 
import pyopencl as cl
import math
import time
import os

float_formatter = "{:.20f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

# Função que gera o kernel quadrado de ordem tam_kernel.
def kernel_peso(variancia,tam_kernel):
    soma_peso = 0
    matriz = np.zeros(shape=(tam_kernel,tam_kernel),dtype=np.float32)

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
    
    os.environ["PYOPENCL_CTX"] = ""
    context = cl.create_some_context()
    queue = cl.CommandQueue(context)
    print(context)

    # Saída do host é definida
    h_output = np.zeros(img.shape, dtype = np.float32)


    d_kernel = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf = kernel)
    d_img = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf = img)
    d_out =  cl.Buffer(context,cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = h_output)
  
    program = cl.Program(context, """
    __kernel void gaussian_blur(const int kernel_size, const int dim_x, const int dim_y, __global float*  matrix, __global const float* img, __global float* outimg)
{ 
    int i = get_global_id(0);
    int j = get_global_id(1);
    float conv = 0;

    if(i < dim_x && j < dim_y){
        outimg[j*dim_x+i] = 0;
        for(int ii=0; ii < kernel_size ; ii++){
            for(int ji = 0; ji <kernel_size; ji++){
                if(((ii-(kernel_size/2)+j)*dim_x) + (ji-(kernel_size/2)+i) >= 0 && ((ii-(kernel_size/2)+j)*dim_x) + (ji-(kernel_size/2)+i) < dim_x*dim_y)
                  conv += matrix[ii*kernel_size+ji] * img[((ii-(kernel_size/2)+j)*dim_x) + (ji-(kernel_size/2)+i) ] ; 
                   
            }
        }
        outimg[j*dim_x + i] = conv;
    }
}
    """).build()
    gblur = program.gaussian_blur
    gblur.set_scalar_arg_dtypes([np.int32,np.int32,np.int32,None,None,None])
    tempo_inicio = time.time()
   
    # Dimensões das matrizes
    dim_x = img.shape[0]
    dim_y = img.shape[1]
    kernel_size = kernel.shape[0]
    print(img.shape)

    gblur(queue,img.shape, None,kernel_size,dim_x,dim_y,d_kernel,d_img,d_out)
    cl.enqueue_copy(queue,h_output,d_out)
    queue.finish()
    print("O tempo decorrido foi de: %s segundos" % (time.time() - tempo_inicio ) ) 
    return h_output

# Carregando a imagem em níveis de cinza
img_original = im.open("ney.jpg")
img_original = img_original.convert('L')
img_original = np.asarray(img_original, dtype = np.float32)

img_desfocada =  img_original.copy()

kernel = kernel_peso(1,3)

for i in range(1):
    img_desfocada = gaussian_blur(kernel,img_desfocada)


img_desfocada = im.fromarray(img_desfocada)
img_original =  im.fromarray(img_original)
img_original.show()
img_desfocada.show()
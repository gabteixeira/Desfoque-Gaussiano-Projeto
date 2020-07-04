# Desfoque Gaussiano Paralelizado
Esse projeto foi desenvolvido para a conclusão da ACIEPE: Programação paralela: das threads aos FPGAs - uma introdução ao tema (2020). Uma paralelização da operação de convolução do desfoque gaussiano utilizando OpenCL. 


## Versões
Esse projeto apresenta duas versões de código: gaussian_paralela1.py (que possui a operação de convolução paralelizada) e gaussian_sequencial.py (que é a versão sequencial do código).

**É preciso alertar que a versão paralela está apresentando problemas na saída devido a conversão dos arrays multidimensionais do python para o array unidimensional do C, apesar de que mesmo assim realiza desfoque desejado de forma paralela, que é o objetivo desse projeto.**

## Rodando na sua máquina
Utilize o comando ```git clone https://github.com/gabteixeira/Desfoque-Gaussiano-Projeto.git``` para fazer o download do projeto.

Considerando que você já possua o Python 3 em sua máquina, caso você não possua as bibliotecas necessárias, instale com:
```
pip install numpy Pillow pyopencl
```

para executar o código, basta executar:
```
python <nome-da-versão>
```

Para selecionar a imagem que você deseja realizar o desfoque:

```
img_original = im.open("<nome-da-imagem>.png")
```
**Note que a imagem deve estar na mesma pasta que o projeto.**

Para selecionar a ordem do Kernel, basta trocar onde está o 3:
```
kernel = kernel_peso(1,3)
```

Lembre-se, que quanto mais passagens você fizer, mais desfocada a imagem ficará :).

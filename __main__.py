#!/home/fer/.virtualenvs/cv/bin/python
import cv2 
import numpy as np
import os
import matplotlib.pyplot as plt
## Direcciones de donde se va sacar las Imagenes
path_o = '/home/fer/Control_servo_visual/Code/Practico_2.0/Pictures/'
path_w = '/home/fer/Control_servo_visual/Code/Practico_2.0/Modificadas/'

## Funcion para la lectura de las imagenes en un direccion especificada
def data(path,aux=1):
    images = []
    index = os.listdir(path)
    index.sort()
    for img in index:
        pictures = cv2.imread(os.path.join(path,img), aux)
        images.append([pictures])
    if aux==1:
        img = np.array(images,dtype=np.uint8).reshape(len(images),pictures.shape[0],pictures.shape[1],pictures.shape[2])
    else:
        img = np.array(images,dtype=np.uint8).reshape(len(images),pictures.shape[0],pictures.shape[1])
    return img

def show(img, new):
    ## Mostrar las imagenes por pantalla
    cv2.imshow('Normal', img)
    cv2.imshow('Modificada', new)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return None

def media(img, N):
    ## Definicionde la matrix de convolucion
    print('Filtro Utilizando la media')
    w =np.ones((N,N), dtype = np.float64)
    new = conv(img, w)
    #show(img, new)
    return new

def kernel(seccion, mascara):
    sum = 0
    for i in range(0, seccion.shape[0]):
        for j in range(0, seccion.shape[1]):
            sum = sum + seccion[i, j]*mascara[i, j]
            
    return sum

def gauss(img, w = np.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]])/273):
    print("Filtro utilizando una funcion Gauss")
    N = w.shape[0]
    
    ## definicion los valores del kernel
    a = int((N-1)/2)
    b = int((N-1)/2)

    ## Creacion de la matriz igual
    new = np.array(img, dtype=np.float64)

    ## Bucle donde se ejecuta el algoritmo
    for i in range(a, img.shape[0]-a):
        for j in range(b, img.shape[1]- b):
            A = img[i-a:i+(a+1),j-b:j+(b+1)]
            new[i,j] = (kernel(A, w))
            
    new = np.array(new, dtype = np.uint8)
    return new 

def mediana(img, N):

    print("Filtro Utilizando Mediana")
    
    ## definicion los valores del kernel
    a = int((N-1)/2)
    b = int((N-1)/2)

    ## Creacion de la matriz igual
    new = np.copy(img)

    ## Bucle donde se ejecuta el algoritmo
    for i in range(a, img.shape[0]-a):
        for j in range(b, img.shape[1]- b):
            A = img[i-a:i+(a+1),j-b:j+(b+1)]
            ordenada =np.sort(A, axis = None)
            mitad = int((ordenada.shape[0]/2)+1)
            new[i,j] = ordenada[mitad]

    new = np.array(new, dtype = np.uint8)
    #new1 = cv2.medianBlur(img, N)
    #show(new1, new)
    return new
    
def conv(img, w):
    ## Se consideran kernel cuadrados
    N = w.shape[0]
    
    ## definicion los valores del kernel
    a = int((N-1)/2)
    b = int((N-1)/2)

    ## Creacion de la matriz igual
    new = np.copy(img)

    ## creacion de lo que se va a dividir
    n = np.sum(w) 

    ## Bucle donde se ejecuta el algoritmo
    for i in range(a, img.shape[0]-a):
        for j in range(b, img.shape[1]- b):
            A = img[i-a:i+(a+1),j-b:j+(b+1)]
            new[i,j] = np.trace(A@w.T)*(1/n)

    new = cv2.convertScaleAbs(new)
    return new

def lines_o(pixels,a = 0.4):
    if pixels >= a:
        f = 255
    else:
        f = 0.0
    return f

def amarillo(pixels, a = 0.4):
    if pixels >= a:
        r = 255
        g = 233
    else:
        r = 0
        g = 0
    return r, g

def lines(img):
    ## Filtro de las imagenes y generacion de lineas de un color respectivo
    img = img/255.0
    print("Filtro de Lineas Horizontales")
    w = np.array([[-1,-1,-1],[2,2,2],[-1,-1,-1]], dtype = np.float64)
    w1 = np.array([[-1,-1,2],[-1,2,-1],[2,-1,-1]], dtype = np.float64)
    w2 = np.array([[2,-1,-1],[-1,2,-1],[-1,-1,2]], dtype = np.float64)
    N = w.shape[0]
    
    ## definicion los valores del kernel
    a = int((N-1)/2)
    b = int((N-1)/2)

    ## Creacion de la matriz igual
    blue= np.zeros((img.shape[0], img.shape[1]), dtype = np.float64)
    green = np.zeros((img.shape[0], img.shape[1]), dtype = np.float64)
    green1 = np.zeros((img.shape[0], img.shape[1]), dtype = np.float64)
    red = np.zeros((img.shape[0], img.shape[1]), dtype = np.float64)
    red1 = np.zeros((img.shape[0], img.shape[1]), dtype = np.float64)

    ## Bucle donde se ejecuta el algoritmo
    for i in range(a, img.shape[0]-a):
        for j in range(b, img.shape[1]- b):
            A = img[i-a:i+(a+1),j-b:j+(b+1)]
            ## Seccion para los 3 colores binarios
            blue[i,j] = lines_o(np.trace(A@w.T),a = 0.3)
            green[i,j] = lines_o(np.trace(A@w))
            red[i,j] = lines_o(np.trace(A@w1.T), a = 0.35)
            ## Seccion para el cor amarillo
            red1[i,j], green1[i,j] = amarillo(np.trace(A@w2.T), a = 0.3)

    ## Seccion apra generar el color amrillo del sistem
    red = cv2.bitwise_or(red, red1)
    green = cv2.bitwise_or(green, green1)
    
    ## Generar la imagen nuevamente
    new = cv2.merge([blue, green, red])
    ## Funcion de Opencv para generar la convolucion
    #new = cv2.filter2D(img, -1, w)
    #show(img, new)
    return new 

def roberts(img):
    ## Calculo de lineas usando Roberts
    print("Calculo de bordes usando Roberts")

    kernelx = np.array([[0, 0, 0],[0, 0, 1],[0, -1, 0]], dtype = int)
    kernely = np.array([[0, 0, 0],[0, 1, 0],[0, 0, -1]], dtype = int)
    N = kernelx.shape[0]
    
    ## definicion los valores del kernel
    a = int((N-1)/2)
    b = int((N-1)/2)

    ## Creacion de la matriz igual
    convolucion_x = np.zeros((img.shape[0], img.shape[1]), dtype=np.float64)
    convolucion_y = np.zeros((img.shape[0], img.shape[1]), dtype=np.float64)

    ## Bucle donde se ejecuta el algoritmo
    for i in range(a, img.shape[0]-a):
        for j in range(b, img.shape[1]- b):
            A = img[i-a:i+(a+1),j-b:j+(b+1)]
            convolucion_x[i, j] = np.trace(A@kernelx.T)
            convolucion_y[i, j] = np.trace(A@kernely.T)
    absx = cv2.convertScaleAbs(convolucion_x)
    absy = cv2.convertScaleAbs(convolucion_y)

    ## Prueba usando funciones open cv
    x = cv2.filter2D(img, -1, kernelx)
    y = cv2.filter2D(img, -1, kernely)

    xabs = cv2.convertScaleAbs(x)
    yabs = cv2.convertScaleAbs(y)

    Roberts_1 = cv2.addWeighted(xabs, 0.5, yabs, 0.5, 0)

    Roberts = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)
    return Roberts
    
def prewitt(img):
    ## Calculo de lineas usando Roberts
    print("Calculo de bordes usando Prewitt")

    kernelx = np.array([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]], dtype = int)
    kernely = np.array([[-1, -1, -1],[0, 0, 0],[1, 1, 1]], dtype = int)
    N = kernelx.shape[0]
    
    ## definicion los valores del kernel
    a = int((N-1)/2)
    b = int((N-1)/2)

    ## Creacion de la matriz igual
    convolucion_x = np.zeros((img.shape[0], img.shape[1]), dtype=np.float64)
    convolucion_y = np.zeros((img.shape[0], img.shape[1]), dtype=np.float64)

    ## Bucle donde se ejecuta el algoritmo
    for i in range(a, img.shape[0]-a):
        for j in range(b, img.shape[1]- b):
            A = img[i-a:i+(a+1),j-b:j+(b+1)]
            convolucion_x[i, j] = np.trace(A@kernelx.T)
            convolucion_y[i, j] = np.trace(A@kernely.T)

    absx = cv2.convertScaleAbs(convolucion_x)
    absy = cv2.convertScaleAbs(convolucion_y)

    ## Prueba usando funciones open cv
    x = cv2.filter2D(img, -1, kernelx)
    y = cv2.filter2D(img, -1, kernely)

    xabs = cv2.convertScaleAbs(x)
    yabs = cv2.convertScaleAbs(y)

    Prewitt2 = cv2.addWeighted(xabs, 0.5, yabs, 0.5, 0)

    Prewitt1 = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)

    return Prewitt1

def sobel(img):
    ## Calculo de lineas usando Roberts
    print("Calculo de bordes usando sobel")

    kernelx = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]], dtype = int)
    kernely = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]], dtype = int)
    N = kernelx.shape[0]
    
    ## definicion los valores del kernel
    a = int((N-1)/2)
    b = int((N-1)/2)

    ## Creacion de la matriz igual
    convolucion_x = np.zeros((img.shape[0], img.shape[1]), dtype=np.float64)
    convolucion_y = np.zeros((img.shape[0], img.shape[1]), dtype=np.float64)

    ## Bucle donde se ejecuta el algoritmo
    for i in range(a, img.shape[0]-a):
        for j in range(b, img.shape[1]- b):
            A = img[i-a:i+(a+1),j-b:j+(b+1)]
            convolucion_x[i, j] = np.trace(A@kernelx.T)
            convolucion_y[i, j] = np.trace(A@kernely.T)

    ## Calculo del sobel de manera manual
    absx = cv2.convertScaleAbs(convolucion_x)
    absy = cv2.convertScaleAbs(convolucion_y)

    ## Prueba usando funciones open cv
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    sobelxabs = cv2.convertScaleAbs(sobelx)
    sobelyabs = cv2.convertScaleAbs(sobely)
    
    return sobelxabs, sobelyabs, absx, absy

def frei_chen(img):
    ## Calculo de lineas usando Roberts
    print("Calculo de bordes usando Frei_Chen")

    kernelx = np.array([[-1, 0, 1],[-np.sqrt(2), 0, np.sqrt(2)],[-1, 0, 1]], dtype = int)
    kernely = np.array([[-1, -np.sqrt(2), -1],[0, 0, 0],[1, np.sqrt(2), 1]], dtype = int)
    N = kernelx.shape[0]
    
    ## definicion los valores del kernel
    a = int((N-1)/2)
    b = int((N-1)/2)

    ## Creacion de la matriz igual
    convolucion_x = np.zeros((img.shape[0], img.shape[1]), dtype=np.float64)
    convolucion_y = np.zeros((img.shape[0], img.shape[1]), dtype=np.float64)

    ## Bucle donde se ejecuta el algoritmo
    for i in range(a, img.shape[0]-a):
        for j in range(b, img.shape[1]- b):
            A = img[i-a:i+(a+1),j-b:j+(b+1)]
            convolucion_x[i, j] = np.trace(A@kernelx.T)
            convolucion_y[i, j] = np.trace(A@kernely.T)

    ## Calculo del sobel de manera manual
    absx = cv2.convertScaleAbs(convolucion_x)
    absy = cv2.convertScaleAbs(convolucion_y)
    
    return absx, absy
def gradient(img):
    ## Funcion para calcular el gradiente de una imagen
    a = 1
    b = 1
    ## definicion de las matrices vacias para el cal;culo del gradiente respectivo
    gradiente_x = np.zeros((img.shape[0], img.shape[1]), dtype=np.float64)
    gradiente_y = np.zeros((img.shape[0], img.shape[1]), dtype=np.float64)
    magnitude = np.zeros((img.shape[0], img.shape[1]), dtype=np.float64)
    angle = np.zeros((img.shape[0], img.shape[1]), dtype=np.float64)
    
    for i in range(a, img.shape[0]-a):
        for j in range(b, img.shape[1]-b):
            gradiente_x[i, j] = int(img[i+1, j])-int(img[i-1, j])
            gradiente_y[i, j] = int(img[i, j+1])-int(img[i, j-1])
            magnitude[i, j] = np.sqrt(gradiente_x[i, j]**2+gradiente_y[i, j]**2)
            angle[i, j] = np.arctan2(gradiente_y[i, j], gradiente_x[i, j])
            
    gradient_x_c = cv2.convertScaleAbs(gradiente_x)
    gradient_y_c = cv2.convertScaleAbs(gradiente_y)

    magnitude_c = cv2.convertScaleAbs(magnitude)

    suma = cv2.addWeighted(gradient_x_c, 0.5, gradient_y_c, 0.5, 0)
    
    return gradient_x_c, gradient_y_c, suma, magnitude_c, angle

def laplacian(img):
    ## Funcion para calcular el gradiente de una imagen
    a = 1
    b = 1
    ## definicion de las matrices vacias para el cal;culo del gradiente respectivo
    laplaciano = np.zeros((img.shape[0], img.shape[1]), dtype=np.float64)
    
    for i in range(a, img.shape[0]-a):
        for j in range(b, img.shape[1]-b):
            laplaciano[i, j] = 5*img[i, j] - (int(img[i+1, j])+int(img[i-1, j])+int(img[i, j+1])+int(img[i, j-1]))

    laplaciano_c = cv2.convertScaleAbs(laplaciano)
    suma = cv2.addWeighted(img, 0.5, laplaciano_c, 0.5, 0)
    return laplaciano_c, suma


def highboost(img):
    paso_bajo = media(img, 5)
    paso_alto = cv2.absdiff(img, paso_bajo)
    return paso_alto

def highboos_f(img, A = 1):
    a = (A-1)*img
    paso_alto = highboost(img)

    suma = cv2.convertScaleAbs(a + paso_alto)

    return suma


def filtro_canny(img,a = 10, b = 200):
    canny = cv2.Canny(img, a, b)
    return canny

def Log_o(pixel):
    if -0.1<pixel<0.1:
        f=1
    else:
        f=0;
    return f

## Laplaciano del gaussiano
def Log(img,alpha = 0.6):
    img = img/255
    ## Funcion para calcular el gradiente de una imagen
    a = 0
    b = 0
    ## definicion de las matrices vacias para el cal;culo del gradiente respectivo
    laplaciano_gauss = np.zeros((img.shape[0], img.shape[1]), dtype=np.float64)
    
    for i in range(a, img.shape[0]-a):
        for j in range(b, img.shape[1]-b):
            x = img[i, j]
            factor = np.exp((x**2)/(2*(alpha)**2))
            laplaciano_gauss[i, j] = ((x**2-2*(alpha)**2)/(alpha**4))*factor
            #laplaciano_gauss[i, j] = Log_o(laplaciano_gauss[i, j])

    Final = cv2.normalize(laplaciano_gauss, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    laplaciano_c = cv2.convertScaleAbs(laplaciano_gauss)

    return Final 

def gauss_matrix(img, s = 1, N = 5):
    print("Grafica del el kernel a aplicar")
    a = int((N-1)/2)
    b = int((N-1)/2)+1
    x = np.arange(-a, b, 1)
    y = np.arange(-a, b, 1)
    x, y = np.meshgrid(x, y)

    factor_1 = ((1)/(2*np.pi*(s)**2))
    factor_2 = np.exp(-(x**2+y**2)/(2*(s)**2))

    G = factor_1*factor_2

    maximo = np.amax(G)
    minimo = np.amin(G)
    normado =G/(np.sum(G))
    plt.matshow(normado)
    plt.colorbar()
    plt.show()
    
    return normado

def diferencia_gauss(img):
    G1 = gauss_matrix(img, 1.5, 5)
    G2 = gauss_matrix(img, 0.9, 5)

    a = gauss(img, G2)
    b = gauss(img, G1)
    
    diferencce = cv2.absdiff(a, b)

    return diferencce, a, b

def main():
    ## Lectura de una sola imagen del sistema
    imgs = data(path_o, 0)
    img = imgs[0,:,:]

    ## Pregunta 1 filtro media
    modificada = media(img, 3)
    
    opencv = cv2.blur(img, (3,3))

    show(img, modificada)

    #G1 = gauss_matrix(img, 1.5, 5)
    #modificada = gauss(img, G1)
    #show(img, modificada)

    #modificada = mediana(img,5)
    #modificada1 = cv2.medianBlur(img, 5)

    #lineas = lines(modificada)

    #Robert1 = roberts(img)

    #Prewitt1 = prewitt(img)

    #sobelx, sobely, sobelxm, sobelym = sobel(img)
    #show(sobelx, sobelxm)
    #show(sobely, sobelym)

    ## Frein chen
    #frei_x ,frei_y = frei_chen(img)
    #show(frei_x, sobelxm)

    # Gradientes de una imagen 
    #gradienx, gradienty, suma, magnitud, angle = gradient(img)
    #show(img, suma)
    #show(img, magnitud)
    #show(gradienx, gradienty)
    #show(img, angle)


    ## Calculo del laplaciano de una imagen
    #laplaciano, suma = laplacian(img)
    #show(img, laplaciano)
    #show(img, suma)

    ## Filtro  High Boost
    #high = highboost(img)
    #show(img, high)
    #high_f = highboos_f(img, 1.1)
    #show(high, high_f)
    
    ## Filtro canny usando Opencv
    #canny = filtro_canny(img, 240, 250)
    #show(img, canny)

    ## Lapaciano de gaussiana
    #gauss = Log(img, 1)
    #show(img, gauss)

    #gauss_matrix(img, 1, 5)

    ## seccion de la diferencia gausianna
    #diference, a, b = diferencia_gauss(img)
    #show(a, b)
    #show(img, diference)
    
    return modificada, opencv, img

if __name__ == '__main__':
    try:
        manual, opencv, normal =main()

    except KeyboardInterrupt:
        print("Press Ctrl-c to terminate the while statement")
        pass

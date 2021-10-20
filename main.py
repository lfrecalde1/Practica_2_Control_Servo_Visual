#!/home/fer/.virtualenvs/cv/bin/python
import cv2                                                                                                              
import numpy as np                                                                                                      
import os                                                                                                               
import matplotlib.pyplot as plt                                                                                         
from functions import *

## Direcciones de donde se va sacar las Imagenes                                                                        
path_o = '/home/fer/Control_servo_visual/Code/Practico_2.0/Pictures/'                                                   
path_w = '/home/fer/Control_servo_visual/Code/Practico_2.0/Modificadas/'                                                
path_c = '/home/fer/Control_servo_visual/Code/Practico_2.0/Calibration/'
path_cw = '/home/fer/Control_servo_visual/Code/Practico_2.0/Results_calibration/'

def mediana(img, N, contador, path):
    print("Filtro Utilizando Mediana")

    ## definicion los valores del kernel
    a = int((N-1)/2)
    b = int((N-1)/2)
    
    ## Creacion de la matriz igual
    new = np.array(img, dtype=np.float32)

    ## Bucle donde se ejecuta el algoritmo
    for i in range(a, img.shape[0]-a):
        for j in range(b, img.shape[1]- b):
            A = img[i-a:i+(a+1),j-b:j+(b+1)]
            ordenada =np.sort(A, axis = None)
            mitad = int((ordenada.shape[0]/2)+1)
            new[i,j] = ordenada[mitad]

    new = np.round(new)
    new = np.uint8(new)

    dst = cv2.medianBlur(img, N)

    ## guardar la imagen del sistema
    
    ## guardar la iamgen generada
    name = "Pregunta_22_{}.png".format(contador)
    name1 = "Pregunta_22_opencv_{}.png".format(contador)
    guardar(path, name, new)
    guardar(path, name1, dst)
    return new

def all(imgs, f, N ,path):
    contador = 0
    for img in imgs:
        dst= f(img, N, contador, path)
        show(img, dst)
        contador = contador + 1

def main():
    imgs = data(path_o, 0)
    ## Pregunta 1
    #Calibration(path_c, path_cw)

    ## Pregunta 2 filtro media
    #all(imgs, conv, 3, path_w)

    ## Pregunta 2 filtro Gaussiano
    #all(imgs, gauss_f, 3, path_w)

    ## Pregunta 2 filtro media
    all(imgs, mediana, 3, path_w)

    

if __name__=='__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Pres Ctrl-c to end the statemen")
        pass

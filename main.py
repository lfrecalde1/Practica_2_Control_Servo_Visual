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
path_l = '/home/fer/Control_servo_visual/Code/Practico_2.0/Lines/'


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
    #all(imgs, conv, 7, path_w)

    ## Pregunta 2 filtro Gaussiano
    #all(imgs, gauss_f, 7, path_w)

    ## Pregunta 2 filtro media
    #all(imgs, mediana, 7, path_w)

    ## Pregunta 3 Filtro de lineas
    #line , original = lines(path_l, path_w)
    #show(original, line)

    ## Pregunta 4 gradiente de las imagenes
    #x, y , s, m = gradient(path_w, path_o)

    ## Pregunta 5 Robert, Prewitt, Sobel y Frei-Chen
    ## Robertrs
    all(imgs, roberts, 3, path_w)
    
    ## Prewwit
    #all(imgs, prewitt, 3, path_w)

    ## Sobel
    #all(imgs, sobel, 3, path_w)

    ## Frei Chen
    #all(imgs, frei_chen, 3, path_w)

    ## Pregunta 6 Laplacian
    #all(imgs, laplacian, 3, path_w)

    ## Pregunta 7 High Boost
    #all(imgs, highboost, 7, path_w)

    ## Pregunta 8 High boost modificado
    #all(imgs, highboost_f, 7, path_w)

    #3 Pregunta 9 Lineas Canny
    #all(imgs, filtro_canny, 3, path_w)


if __name__=='__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Pres Ctrl-c to end the statemen")
        pass

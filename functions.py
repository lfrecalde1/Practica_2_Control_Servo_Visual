#!/home/fer/.virtualenvs/cv/bin/python                                                                                  
import cv2                                                                                                              
import numpy as np                                                                                                      
import os                                                                                                               
import matplotlib.pyplot as plt                                                                                         
                                                                                                                        
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

def Calibration(dir_r, dir_w):
    ## Definicion de la dimension de los frames y la dimension del cheesboard
    chessboardSize = (9,7)
    frameSize = (640,480)
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = data(dir_r, 1)

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)

        # If found, add object points, image points (after refining them)
        if ret == True:

            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, chessboardSize, corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(1000)


    cv2.destroyAllWindows()

    ############## CALIBRATION #######################################################

    ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

    print(cameraMatrix)


    ############## UNDISTORTION #####################################################


    img_1 = images[9, :, :]
    h,  w = img_1.shape[:2]
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))



    # Undistort
    dst = cv2.undistort(img_1, cameraMatrix, dist, None, newCameraMatrix)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    name = "Pregunta_1_calibrate.png"
    name_u = "Pregunta_1_Nocalibrate.png"
    guardar(dir_w, name, dst)
    guardar(dir_w, name_u, images[9, :, :] )

    ## seccion calcular el error generado en la estimacion de parametros
    mean_error = 0

    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error

    print( "total error: {}".format(mean_error/len(objpoints)) )
    return None

def guardar(direccion, name, new):
    cv2.imwrite(os.path.join(direccion, name), new)
    return None

def median(img, N):                                                                                                                                                                               
     kernel = np.ones((N,N), np.float32)
     suma = np.sum(kernel)
     kernel = kernel/suma                                                                         
     depth = -1                                                                                                          
     anchor = (-1, -1)                                                                                                   
     delta = 0.0                                                                                                         
     dst = cv2.filter2D(img, depth, kernel, cv2.BORDER_CONSTANT)                                                         
     return dst                                                                                                          
                 
## definicion los valores del kernel                                                                                                                                                                                                  
def operacion(a, b):
    sum = 0.0
    for i in range(0, a.shape[0]):
        for j in range(0, a.shape[1]):
            sum = sum + a[i,j]*b[i,j]
    return sum

def conv(img, N, contador, path):

    ## Matriz de convolucion
    w = np.ones((N, N), np.float32)
    
    a = int((N-1)/2)                                                                                                    
    b = int((N-1)/2)                                                                                                    
    ## Creacion de la matriz igual                                                                                      
    new = np.array(img, np.float32)
                                                                                                                        
    ## creacion de lo que se va a dividir                                                                               
    n = np.sum(w)                                                                                                       

    w =w/n
    ## Bucle donde se ejecuta el algoritmo                                                                              
    for i in range(a, img.shape[0]-a):                                                                                  
        for j in range(b, img.shape[1]- b):                                                                             
            A = img[i-a:i+(a+1),j-b:j+(b+1)]                                                                            
            new[i,j] = operacion(A, w)
            
    new = np.round(new)
    new = np.uint8(new)
    
    ## Generacion del resultado por opencv2
    dst = median(img, N)

    ## guardar la iamgen generada
    name = "Pregunta_2_{}.png".format(contador)
    name1 = "Pregunta_2_opencv{}.png".format(contador)
    guardar(path, name, new)
    guardar(path, name1, dst)
    
    return new    

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


def laplacian_gauss_matrix(img, s = 1, N = 5):
    print("Grafica del el kernel a aplicar")
    a = int((N-1)/2)
    b = int((N-1)/2)+1
    x = np.arange(-a, b, 1)
    y = np.arange(-a, b, 1)
    x, y = np.meshgrid(x, y)

    factor_1 = ((x**(2)+y**(2)-2*s**(2))/s**(4))
    factor_2 = np.exp(-(x**2+y**2)/(2*(s)**2))

    G = ((1)/(2*np.pi*s**2))*factor_1*factor_2

    maximo = np.amax(G)
    minimo = np.amin(G)
    normado =G/(np.max(G))
    plt.matshow(G)
    plt.colorbar()
    plt.show()
    
    return G


def laplacian_gauss(img, N, contador, path):

    w = laplacian_gauss_matrix(img, 1, N)

    print("Filtro utilizando una funcion Laplacian Gauss")
    N = w.shape[0]

    ## definicion los valores del kernel
    a = int((N-1)/2)
    b = int((N-1)/2)

    ## Creacion de la matriz igual
    new = np.array(img, dtype=np.float32)

    ## Bucle donde se ejecuta el algoritmo
    for i in range(a, img.shape[0]-a):
        for j in range(b, img.shape[1]- b):
            A = img[i-a:i+(a+1),j-b:j+(b+1)]
            new[i,j] = operacion(A, w)

    new = np.round(new)
    new = np.uint8(new)

    dst =cv2.GaussianBlur(img, (N,N), cv2.BORDER_CONSTANT)
    laplacian_opencv = cv2.Laplacian(dst, cv2.CV_64F)

    ## guardar la iamgen generada
    name = "Pregunta_final_{}.png".format(contador)
    #name1 = "Pregunta_final_opencv_{}.png".format(contador)

    new_final = Zero_crossing(laplacian_opencv)
    guardar(path, name, new_final)

    return new_final

def Zero_crossing(image):
    z_c_image = np.zeros(image.shape)

    # For each pixel, count the number of positive
    # and negative pixels in the neighborhood

    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            negative_count = 0
            positive_count = 0
            neighbour = [image[i+1, j-1],image[i+1, j],image[i+1, j+1],image[i, j-1],image[i, j+1],image[i-1, j-1],image[i-1, j],image[i-1, j+1]]
            d = max(neighbour)
            e = min(neighbour)
            for h in neighbour:
                if h>0:
                    positive_count += 1
                elif h<0:
                    negative_count += 1


            # If both negative and positive values exist in
            # the pixel neighborhood, then that pixel is a
            # potential zero crossing

            z_c = ((negative_count > 0) and (positive_count > 0))

            # Change the pixel value with the maximum neighborhood
            # difference with the pixel

            if z_c:
                if image[i,j]>0:
                    z_c_image[i, j] = image[i,j] + np.abs(e)
                elif image[i,j]<0:
                    z_c_image[i, j] = np.abs(image[i,j]) + d

    # Normalize and change datatype to 'uint8' (optional)
    z_c_norm = z_c_image/z_c_image.max()*255
    z_c_image = np.uint8(z_c_norm)

    return z_c_image

def gauss_f(img, N, contador, path):

    w = gauss_matrix(img, 1, N)

    print("Filtro utilizando una funcion Gauss")
    N = w.shape[0]

    ## definicion los valores del kernel
    a = int((N-1)/2)
    b = int((N-1)/2)

    ## Creacion de la matriz igual
    new = np.array(img, dtype=np.float32)

    ## Bucle donde se ejecuta el algoritmo
    for i in range(a, img.shape[0]-a):
        for j in range(b, img.shape[1]- b):
            A = img[i-a:i+(a+1),j-b:j+(b+1)]
            new[i,j] = operacion(A, w)

    new = np.round(new)
    new = np.uint8(new)

    dst =cv2.GaussianBlur(img, (N,N), cv2.BORDER_CONSTANT)

    ## guardar la iamgen generada
    name = "Pregunta_21_{}.png".format(contador)
    name1 = "Pregunta_21_opencv_{}.png".format(contador)
    guardar(path, name, new)
    guardar(path, name1, dst)

    return new 
   
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

    ## guardar la iamgen generada
    name = "Pregunta_22_{}.png".format(contador)
    name1 = "Pregunta_22_opencv_{}.png".format(contador)
    guardar(path, name, new)
    guardar(path, name1, dst)
    return new

def lines_o(pixel, a):
    if pixel >= a: 
        f = 255 
    else:
        f = 0
    return f

def amarillo(pixels, a = 0.4):
    if pixels >= a:
        r = 255
        g = 233
    else:
        r = 0
        g = 0
    return r, g


def lines(path, path_b):

    img = data(path, 0)

    ## Filtro de las imagenes y generacion de lineas de un color respectivo
    img = img[0,:,:]/255

    ## Image rotation horizontal to vertical
    #(h, w) = img.shape[:2]
    #(cX, cY) = (w//2, h//2)
    #factor = cv2.getRotationMatrix2D((cX, cY), 90, 1.0)
    #img = cv2.warpAffine(img, factor, (w, h))
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
            green[i,j] = lines_o(np.trace(A@w), a = 0.3)
            red[i,j] = lines_o(np.trace(A@w1.T), a = 0.25)
            ## Seccion para el cor amarillo
            red1[i,j], green1[i,j] = amarillo(np.trace(A@w2.T), a = 0.25)

    ## Seccion apra generar el color amrillo del sistem
    red = cv2.bitwise_or(red, red1)
    green = cv2.bitwise_or(green, green1)
    
    ## Generar la imagen nuevamente
    new = cv2.merge([blue, green, red])
    name = "Pregunta_3_{}.png".format(0)
    guardar(path_b, name, new)
    return new, img

def gradient(path,path_image):
    ## Cargar la iamgen deseada
    imgs = data(path_image, 0)
    img = imgs[0,:,:]
    ## Funcion para calcular el gradiente de una imagen
    print('Calculo del gradiente de la imagen')
    a = 1
    b = 1
    ## definicion de las matrices vacias para el cal;culo del gradiente respectivo
    gradiente_x = np.zeros((img.shape[0], img.shape[1]), dtype=np.float64)
    gradiente_y = np.zeros((img.shape[0], img.shape[1]), dtype=np.float64)
    magnitude = np.zeros((img.shape[0], img.shape[1]), dtype=np.float64)
    angle = np.zeros((img.shape[0], img.shape[1]), dtype=np.float64)
    
    for i in range(a, img.shape[0]-a):
        for j in range(b, img.shape[1]-b):
            gradiente_y[i, j] = int(img[i+1, j])-int(img[i-1, j])
            gradiente_x[i, j] = int(img[i, j+1])-int(img[i, j-1])
            magnitude[i, j] = np.sqrt(gradiente_x[i, j]**2+gradiente_y[i, j]**2)
            angle[i, j] = np.arctan2(gradiente_x[i, j], gradiente_y[i, j])
            
    gradient_x_c = cv2.convertScaleAbs(gradiente_x)
    gradient_y_c = cv2.convertScaleAbs(gradiente_y)
    magnitude_c = cv2.convertScaleAbs(magnitude)
    suma = cv2.addWeighted(gradient_x_c, 0.5, gradient_y_c, 0.5, 0)

    ## Seccion para guardar las imagenes resultantes
    name1 = "Pregunta_4_1{}.png".format(0)
    name2 = "Pregunta_4_2{}.png".format(0)
    name3 = "Pregunta_4_3{}.png".format(0)
    name4 = "Pregunta_4_4{}.png".format(0)
    name5 = "Pregunta_4_5{}.png".format(0)

    guardar(path, name1, gradient_x_c)
    guardar(path, name2, gradient_y_c)
    guardar(path, name3, suma)
    guardar(path, name4, magnitude_c)

    # Interest 
    u_i_min = 300
    u_i_max = 301
    v_i_min = 150
    v_i_max = 200
    
    factor_u = 40
    
    ## Draw Rectagle 
    ## Coordinates of rectagle
    start_point = (u_i_min-factor_u, v_i_min)
    end_point = (u_i_max+factor_u, v_i_max)

    color = (255, 255, 0)

    thickness = 2
    
    img_interts = cv2.rectangle(suma, start_point, end_point, color, thickness)
    img_interts = cv2.line(img_interts, (u_i_min,v_i_min), (u_i_min, v_i_max), (100, 0, 0), thickness)
    guardar(path, name5, img_interts)

    ## GENERATE DATA TO PLOT
    interes_zone = angle[u_i_min:u_i_max,v_i_min:v_i_max]
    x_zone = np.arange(150,200).reshape(1, interes_zone.shape[1])

    ## Fancy plot
    with plt.style.context(['science','grid']):
        fig, ax = plt.subplots()
        ax.plot(x_zone[0,:], interes_zone[0,:], label='$Angle$')
        ax.legend(title='')
        ax.legend( fontsize='x-small')
        ax.autoscale(tight=True)
        #ax[0].set(**pparam)
        ax.set_ylabel('Angle ($rad$)')
        ax.set_xlabel('Pixel')
        #plt.show()
        fig.savefig('/home/fer/Control_servo_visual/Code/Practico_2.0/Modificadas/Angle.eps', format = 'eps')

    return gradient_x_c, gradient_y_c, suma, magnitude_c

def roberts(img, N, contador, path):
    ## Calculo de lineas usando Roberts
    print("Calculo de bordes usando Roberts")

    kernelx = np.array([[0, 0, 0],[0, 0, 1],[0, -1, 0]], dtype = int)
    kernely = np.array([[0, 0, 0],[0, 1, 0],[0, 0, -1]], dtype = int)
    N = kernelx.shape[0]
    
    ## definicion los valores del kernel
    a = int((N-1)/2)
    b = int((N-1)/2)

    ## Creacion de la matriz igual
    convolucion_x = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    convolucion_y = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

    ## Bucle donde se ejecuta el algoritmo
    for i in range(a, img.shape[0]-a):
        for j in range(b, img.shape[1]- b):
            A = img[i-a:i+(a+1),j-b:j+(b+1)]
            convolucion_x[i, j] = operacion(A, kernelx)
            convolucion_y[i, j] = operacion(A, kernely)

    absx = cv2.convertScaleAbs(convolucion_x)
    absy = cv2.convertScaleAbs(convolucion_y)

    ## Prueba usando funciones open cv
    x = cv2.filter2D(img, -1, kernelx)
    y = cv2.filter2D(img, -1, kernely)

    xabs = cv2.convertScaleAbs(x)
    yabs = cv2.convertScaleAbs(y)

    ## generacion de la imagen usando opencv
    Roberts_1 = cv2.addWeighted(xabs, 0.5, yabs, 0.5, 0)

    ## generacion de la imagen manualmente
    Roberts = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)

    name = "Pregunta_5_{}.png".format(contador)
    name1 = "Pregunta_5_opencv_{}.png".format(contador)
    guardar(path, name, Roberts)
    guardar(path, name1, Roberts_1)

    return Roberts

def prewitt(img, N, contador, path):
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
            convolucion_x[i, j] = operacion(A, kernelx)
            convolucion_y[i, j] = operacion(A, kernely) 

    absx = cv2.convertScaleAbs(convolucion_x)
    absy = cv2.convertScaleAbs(convolucion_y)

    ## Prueba usando funciones open cv
    x = cv2.filter2D(img, -1, kernelx)
    y = cv2.filter2D(img, -1, kernely)

    xabs = cv2.convertScaleAbs(x)
    yabs = cv2.convertScaleAbs(y)

    # generar las iamgenes del sistema
    Prewitt1 = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)
    Prewitt2 = cv2.addWeighted(xabs, 0.5, yabs, 0.5, 0)

    ## Seccion para guardar las imagenes del sistema
    name = "Pregunta_5_1_{}.png".format(contador)
    name1 = "Pregunta_5_1_opencv_{}.png".format(contador)
    guardar(path, name, Prewitt1)
    guardar(path, name1, Prewitt2)
    return Prewitt1

def sobel(img, N, contador, path):
    ## Calculo de lineas usando Roberts
    print("Calculo de bordes usando sobel")

    kernelx = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]], dtype = int)
    kernely = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]], dtype = int)
    N = kernelx.shape[0]
    
    ## definicion los valores del kernel
    a = int((N-1)/2)
    b = int((N-1)/2)

    ## Creacion de la matriz igual
    convolucion_x = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    convolucion_y = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

    ## Bucle donde se ejecuta el algoritmo
    for i in range(a, img.shape[0]-a):
        for j in range(b, img.shape[1]- b):
            A = img[i-a:i+(a+1),j-b:j+(b+1)]
            convolucion_x[i, j] = operacion(A, kernelx)
            convolucion_y[i, j] = operacion(A, kernely)

    ## Calculo del sobel de manera manual
    absx = cv2.convertScaleAbs(convolucion_x)
    absy = cv2.convertScaleAbs(convolucion_y)

    ## Prueba usando funciones open cv
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    sobelxabs = cv2.convertScaleAbs(sobelx)
    sobelyabs = cv2.convertScaleAbs(sobely)

    ## Generacion de los nombres de la imagenes a ser guardadas
    name1 = "Pregunta_5_2_x_{}.png".format(contador)
    name2 = "Pregunta_5_2_y_{}.png".format(contador)
    name3 = "Pregunta_5_2_x_opencv_{}.png".format(contador)
    name4 = "Pregunta_5_2_y_opencv_{}.png".format(contador)

    guardar(path, name1, absx)
    guardar(path, name2, absy)
    guardar(path, name3, sobelxabs)
    guardar(path, name4, sobelyabs)

    ## Completa 
    new = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)
    
    return new


def frei_chen(img, N, contador, path):
    ## Calculo de lineas usando Roberts
    print("Calculo de bordes usando Frei_Chen")

    kernelx = np.array([[-1, 0, 1],[-np.sqrt(2), 0, np.sqrt(2)],[-1, 0, 1]], dtype = np.float32)
    kernely = np.array([[-1, -np.sqrt(2), -1],[0, 0, 0],[1, np.sqrt(2), 1]], dtype = np.float32)
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
            convolucion_x[i, j] = operacion(A, kernelx)
            convolucion_y[i, j] = operacion(A, kernely)

    ## Calculo del sobel de manera manual
    absx = cv2.convertScaleAbs(convolucion_x)
    absy = cv2.convertScaleAbs(convolucion_y)

    ## Generacion de los nombres de la imagenes a ser guardadas
    name1 = "Pregunta_5_3_x_{}.png".format(contador)
    name2 = "Pregunta_5_3_y_{}.png".format(contador)

    guardar(path, name1, absx)
    guardar(path, name2, absy)

    new = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)

    return new

def laplacian(img, N, contador, path):
    ## Funcion para calcular el gradiente de una imagen
    print("Calculando el gain")
    a = 1
    b = 1
    ## definicion de las matrices vacias para el cal;culo del gradiente respectivo
    gain = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    laplaciano = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

    for i in range(a, img.shape[0]-a):
        for j in range(b, img.shape[1]-b):
            gain[i, j] = 5*img[i, j] - (int(img[i+1, j])+int(img[i-1, j])+int(img[i, j+1])+int(img[i, j-1]))
            laplaciano[i, j] = int(img[i+1, j])+int(img[i-1, j])+int(img[i, j+1])+int(img[i, j-1])-4*img[i, j]

    gain_c = cv2.convertScaleAbs(gain)
    laplaciano_c = cv2.convertScaleAbs(laplaciano)

    name1 = "Pregunta_6_Laplacian_{}.png".format(contador)
    name2 = "Pregunta_6_Image_gain_{}.png".format(contador)
    name3 = "Pregunta_6_Laplacian_Original_{}.png".format(contador)

    guardar(path, name1, laplaciano_c)
    guardar(path, name2, gain_c)
    guardar(path, name3, img)

    return gain_c 


def highboost(img, N, contador, path):
    print("High boost normal")
    paso_bajo = median(img, N)
    paso_alto = cv2.absdiff(img, paso_bajo)
    name1 = "Pregunta_7_{}.png".format(contador)
    guardar(path, name1, paso_alto)
    return paso_alto

def highboost_f(img, N, contador, path):
    print("High Boost modificado")
    paso_bajo = median(img, N)
    a = 1
    aux = a*img
    aux = np.uint8(aux)
    paso_alto = cv2.absdiff(aux, paso_bajo)
    paso_alto = cv2.convertScaleAbs(paso_alto)
    name1 = "Pregunta_8_{}.png".format(contador)
    guardar(path, name1, paso_alto)
    return paso_alto


def filtro_canny(img, N, contador, path):
    print("Deteccionde lineas usando canny")
    a = 10
    b = 200
    ## filtro de lineas usando la funcion de opencv para canny
    canny = cv2.Canny(img, a, b)
    ## Guardaar la imagen en el sistema
    name1 = "Pregunta_9_{}.png".format(contador)
    guardar(path, name1, canny)
    return canny


def diferencia_gauss(img,path):
    contador = 0
    s1 = 1.5
    s2 = 0.9
    N = 5
    print("Diferencia Gaussiana")
    G1 = gauss_matrix(img, s1, N)
    G2 = gauss_matrix(img, s2, N)

    a = gauss(img, G2)
    b = gauss(img, G1)

    diferencce = cv2.absdiff(a, b)

    name1 = "Pregunta_10_{}.png".format(contador)
    guardar(path, name1, diferencce)

    return diferencce

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
            new[i,j] = (operacion(A, w))

    new = np.array(new, dtype = np.uint8)
    return new

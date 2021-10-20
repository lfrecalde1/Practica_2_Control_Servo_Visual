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
def median(img):                                                                                                                                                                               
     kernel = np.ones((3,3), np.float32)/9                                                                               
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
    dst = median(img)

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
   
                   

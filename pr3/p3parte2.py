import scipy.io
import numpy as np


#FUNCION DE ACTIVACION -> FUNCION SIGMOIDE
def sigmoide(x):
    return 1/(1+ np.exp(np.negative(x)))

def propagacionHaciaDelante(theta1,theta2,X):
    X=np.transpose(X)
    z2=np.matmul(theta1,X)
    a2=sigmoide(z2)
    a2=np.transpose(a2)
    a2=np.concatenate((np.atleast_2d(np.ones(len(a2),int)).T,a2),axis=1)
    z3=np.matmul(theta2,np.transpose(a2))
    a3=sigmoide(z3)
    return a3
    
def porcentajeRedNeuronal(h,y):
    #Busca el maximo de la matriz h
    maximo = np.argmax(h, axis = 0)
    #1 si está bien y 0 si está mal
    salida = np.reshape(y,5000)
    maximo = maximo +1
    comparacion=(maximo==salida)
    #Cuenta el numero de unos que tiene comparacion
    comparacion = comparacion*1 
    bienPredecidos= np.count_nonzero(comparacion==1)
    #Calculo del porcentaje
    porcentaje=(bienPredecidos/len(comparacion))*100
    return porcentaje
    

def redNeuronal():
    data = scipy.io.loadmat('ex3data1.mat')
    y = data['y']
    X = data ['X']
    weights = scipy.io.loadmat('ex3weights.mat')
    theta1, theta2 = weights['Theta1'], weights['Theta2']
    aux=np.ones(X.shape[0],dtype=int)
    X = np.concatenate((np.atleast_2d(aux).T,X),axis=1)
    h=propagacionHaciaDelante(theta1,theta2,X)
    porcentaje=porcentajeRedNeuronal(h,y)
    print(porcentaje)
   
    
    
redNeuronal()
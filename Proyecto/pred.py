import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt


def sigmoide(x):
    return 1/(1+ np.exp(np.negative(x)))

def sigmoideDerivada(z):
    sd = sigmoide(z) * (1 - sigmoide(z));
    return sd

def porcentajeRedNeuronal(Theta1, Theta2, X, y):
   m = X.shape[0]
   a1=np.hstack((np.ones((m,1)),X))
   z2=np.matmul(Theta1,np.transpose(a1))
   a2=sigmoide(z2)
   a2=np.vstack((np.ones((1,a2.shape[1])),a2))
   z3=np.matmul(Theta2,a2)
   a3=sigmoide(z3)
   h=a3
   maximo=np.argmax(h,axis=0)
   comparacion=(maximo == y[:,0])*1
   
   bienPredecidos = np.count_nonzero(comparacion)
   
   porcentaje = (bienPredecidos/m)*100
   
   return porcentaje
   
def pesosAleatorios(L_in,L_out):
    ini =0.12
    pesos = np.random.rand((L_in+1)*L_out)*(2*ini) - ini
    pesos = np.reshape(pesos, (L_out,1+L_in))
    return pesos      
    

def errorlmdb(X,y,Xval,yval,Theta1_ini,Theta2_ini):
    lmdb= np.array([0,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,15,20,30,50])
    num_etiquetas=21
    num_entradas=32
    num_ocultas=10 #Probar con distintos valores
    
    aux = np.reshape(Theta1_ini,(num_entradas+1)*num_ocultas)
    aux2 = np.reshape(Theta2_ini,(num_ocultas+1)*num_etiquetas)
    params_ini=np.concatenate((aux,aux2))

    porcentajeEnt = np.zeros(len(lmdb))
    porcentajeVal = np.zeros(len(lmdb))
    for i in range(0,len(lmdb)): #¿i == 1?
        res = opt.minimize(backprop,params_ini,args=(num_entradas,num_ocultas,num_etiquetas,X,y,lmdb[i]),jac=True,options={'maxiter':5000})
        grad = res.x
        
        
        Theta1 = np.reshape(grad[:(num_ocultas*(num_entradas+1))],(num_ocultas, (num_entradas+1)))
        Theta2 = np.reshape(grad[(num_ocultas*(num_entradas+1)):],(num_etiquetas, (num_ocultas+1)))
        
        porcentajeEnt[i] = porcentajeRedNeuronal(Theta1, Theta2, X, y)
        porcentajeVal[i] = porcentajeRedNeuronal(Theta1, Theta2, Xval, yval)
        
        
    plt.xlabel('lambda')
    plt.ylabel('')
    plt.plot(lmdb,porcentajeEnt,label="Entrenamiento", c='r')
    plt.plot(lmdb,porcentajeVal,label="Validacion", c= 'g')
    plt.legend()

    

def debugInitializeWeights(fan_in, fan_out):
    """
    Initializes the weights of a layer with fan_in incoming connections and
    fan_out outgoing connections using a fixed set of values.
    """

    # Set W to zero matrix
    W = np.zeros((fan_out, fan_in + 1))

    # Initialize W using "sin". This ensures that W is always of the same
    # values and will be useful in debugging.
    W = np.array([np.sin(w) for w in
                  range(np.size(W))]).reshape((np.size(W, 0), np.size(W, 1)))

    return W

def backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, reg):
    Theta1 = np.reshape(params_rn[:num_ocultas*(num_entradas+1)],(num_ocultas, (num_entradas+1)))
    Theta2 = np.reshape(params_rn[num_ocultas*(num_entradas+1):],(num_etiquetas, (num_ocultas+1)))
    m = X.shape[0]
    
    #Propagacion hacia delante
    a1 = np.vstack((np.ones(X.shape[0]),X.T))
    z2=np.matmul(Theta1,a1)
    a2 = np.vstack((np.ones(z2.shape[1]),z2))
    a2=sigmoide(a2)
    z3=np.matmul(Theta2,a2)
    a3=sigmoide(z3)
    h = a3
    etiqueta = np.identity(num_etiquetas)
    ycod = etiqueta[y[:,0].astype(int),:]
    J = np.sum(np.matmul((-ycod),np.log(h) )- np.matmul((1 - ycod),np.log(1 - h)))/m
    
    #Regularizacion
    regular = (reg/(2*m))*(np.sum(np.square(Theta1[:,:]))+np.sum(np.square(Theta2[:,:])))
    final = J+regular
    
    #Retro propagacion
    d3 = h.T - ycod
    d2 = np.matmul(Theta2.T,d3.T)[1:,:] *sigmoideDerivada(z2)
    grad1 = (Theta1 +np.matmul(d2,a1.T))/m
    grad2 = (Theta2 +np.matmul(d3.T,a2.T))/m
    #Regularizacion del gradiente
    reg1= (reg/m) * Theta1[:,1:]
    reg2= (reg/m) * Theta2[:,1:]

    #Regularizacion del gradiente    
    fingrad1 = grad1
    fingrad1[:,1:] += reg1
    fingrad2 = grad2
    fingrad2[:,1:] += reg2
    #Fin del gradiente
    aux = np.reshape(fingrad1,fingrad1.shape[0]*fingrad1.shape[1])
    aux2 = np.reshape(fingrad2, fingrad2.shape[0]*fingrad2.shape[1])
    grad =np.concatenate((aux,aux2))
    return final,grad


def main():
    #DATOS INICIALES
    num_etiquetas=21
    num_entradas=32
    num_ocultas=10 #Probar con distintos valores
    
    datamat = np.genfromtxt('student-mat-Modificado.csv', delimiter = ';')
    datapor = np.genfromtxt('student-por-Modificado.csv', delimiter = ';')
    
    #Datos de entrada
    entmat = np.vstack((datamat[:244],datamat[349:381]))
    entpor = np.vstack((datapor[:296],datapor[423:581]))
    ent = np.vstack((entmat,entpor))
    
    #Datos de validacion
    valmat = np.vstack((datamat[244:297],datamat[381:387]))
    valpor = np.vstack((datapor[296:360],datapor[581:615]))
    val = np.vstack((valmat,valpor))    
    
    #Datos test
    testmat = np.vstack((datamat[297:349],datamat[387:]))
    testpor = np.vstack((datapor[360:423],datapor[615:]))
    test = np.vstack((testmat,testpor))

    #Datos de todos
    Xent = ent[:,:32]
    Yent = ent[:,32:]
    Xval = val[:,:32]
    Yval = val[:,32:]
    Xtest = test[:,:32]
    ytest = test[:,32:]
    #Inicializacion de pesos aleatorios
    Theta1_ini = pesosAleatorios(num_entradas,num_ocultas)
    Theta2_ini = pesosAleatorios(num_ocultas,num_etiquetas)
    
    #Calculamos el error de lambda y cogemos el mejor
    errorlmdb(Xent,Yent,Xval,Yval,Theta1_ini,Theta2_ini)
    
    
    #Unimos los datos de entrenamiento y validacion
    Xent_val=np.concatenate((Xent,Xval))
    yent_val=np.concatenate((Yent,Yval))
    
    #Entrenamiento de la red neuronal
    #valorlambda = 1 #El que resulte de la grafica de errorlmbd
    
    aux = np.reshape(Theta1_ini,(num_entradas+1)*num_ocultas)
    aux2 = np.reshape(Theta2_ini,(num_ocultas+1)*num_etiquetas)
    params_ini=np.concatenate((aux,aux2))
    res = opt.minimize(backprop,params_ini,args=(num_entradas,num_ocultas,num_etiquetas,Xent_val,yent_val,15),jac=True,options={'maxiter':5000})
    grad = res.x
        
    Theta1 = np.reshape(grad[:num_ocultas*(num_entradas+1)],(num_ocultas, (num_entradas+1)))
    Theta2 = np.reshape(grad[num_ocultas*(num_entradas+1):],(num_etiquetas, (num_ocultas+1)))
    #Theta11 = np.reshape(params_rn(1:num_ocultas * (num_entradas + 1)), num_ocultas, (num_entradas + 1));
    #Theta21 = np.reshape(params_rn((1 + (num_ocultas * (num_entradas + 1))):end), num_etiquetas, (num_ocultas + 1));
    
    
    #Porcentaje 
    num_por = porcentajeRedNeuronal(Theta1, Theta2, Xtest, ytest)
    print(num_por)
    num_por = porcentajeRedNeuronal(Theta1, Theta2, Xent_val, yent_val)
    print(num_por)
    num_por = porcentajeRedNeuronal(Theta1, Theta2, Xent, Yent)
    print(num_por)
    num_por = porcentajeRedNeuronal(Theta1, Theta2, Xval, Yval)
    print(num_por)


main()
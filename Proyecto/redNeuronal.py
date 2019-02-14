import numpy as np
import scipy.io
import scipy.optimize as opt
import matplotlib.pyplot as plt


def sigmoide(x):
    return 1/(1+ np.exp(np.negative(x)))

def sigmoideDerivada(z):
    sd = sigmoide(z) * (1 - sigmoide(z));
    return sd

def porcentajeRedNeuronal(Theta1, Theta2, X, y):
   m = X.shape[0]
   a1=np.vstack((np.multiply(np.ones(m,1),X)))
   z2=Theta1*np.transpose(a1)
   a2=sigmoide(z2)
   a2=np.vstack((np.ones(1,a2.shape[1])*a2))
   z3=np.multiply(Theta2,a2)
   a3=sigmoide(z3)
   h=a3
   
   np.vstack((clase))=max(h)
   
   comparacion=(np.transpose(clase) == y)
   
   bienPredecidos = len(np.find_common_type(comparacion==1))
   
   porcentaje = (bienPredecidos/m)*100
   
   return porcentaje
   
       
    

def errorlmdb(X,y,Xval,yval,Theta1_ini,Theta2_ini):
    lmdb= np.array([0,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10])
    m = X.shape[0]
    num_etiquetas=1
    num_entradas=33
    num_ocultas=10 #Probar con distintos valores
    
    Theta_ini=np.zeros((X.shape[1],1))
    aux = np.reshape(Theta1_ini,(num_entradas+1)*num_ocultas)
    aux2 = np.reshape(Theta2_ini,(num_ocultas+1)*num_etiquetas)
    params_ini=np.concatenate((aux,aux2))
    #options = np.optimset('MaxIter', 5000);

    for i in range(X.shape[1]): #¿i == 1?
        bp = backprop(params_ini,num_entradas,num_ocultas,num_etiquetas,X,y,lmdb[i])
        
        [params_rn, J] = np.fmincg(bp, params_ini)
        #Theta11 = np.reshape(params_ini(1:num_ocultas * (num_entradas + 1)),num_ocultas, (num_entradas + 1))
        #Theta21 = np.reshape(params_ini((1 + (num_ocultas * (num_entradas +1))):end), num_etiquetas, (num_ocultas + 1))
 
        Theta11 = 0
        Theta21 = 0
        porcentajeEnt=[]
        porcentajeVal=[]
        porcentajeEnt[i] = porcentajeRedNeuronal(Theta11, Theta21, X, y)
        porcentajeVal[i] = porcentajeRedNeuronal(Theta11, Theta21, Xval, yval)
        
        
        plt.legend('Entrenamiento', 'Validacion')
        plt.xlabel('Lambda')
        plt.ylabel('Porcentaje acierto')
        plt.plot(lmdb, porcentajeEnt, 'LineWidth', 3, lmdb, porcentajeVal,'LineWidth', 3)

    

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

def backprop(params_ini,num_entradas,num_ocultas,num_etiquetas,Xent_val,Yent_val,valorlambda):
    Theta1 = np.reshape(params_rn[:num_ocultas*(num_entradas+1)],(num_ocultas, (num_entradas+1)))
    Theta2 = np.reshape(params_rn[num_ocultas*(num_entradas+1):],(num_etiquetas, (num_ocultas+1)))
    m = X.shape[0]
    
    yidentidad=np.eye(num_etiquetas)
    ycodificada=yidentidad(y,:)
    
    #Propagacion hacia delante
    a1 = np.vstack((np.ones(X.shape[0]),X.T))
    z2=np.matmul(Theta1,np.transpose(a1))
    a2=sigmoide(z2)
    a2 = np.vstack((np.ones(a2.shape[1]),a2))
    z3=np.matmul(Theta2,a2)
    a3=sigmoide(z3)
    h = a3 #¿Esto que es?
    
    J = np.sum(np.sum((((-ycodificada) .* np.transpose(log(h))) - ((1 - ycodificada) .* np.transpose(log(1 - h))))))/m
    J = J + (valorlambda/(2 * m)) * (np.sum(np.sum(Theta1(:, 2:end) .^ 2)) +np.sum(np.sum(Theta2(:, 2:end) .^ 2)))
    
    sig3 = np.transpose(a3) - ycodificada
    sig2 = (np.transpose(Theta2) * np.transpose(sig3)) .* sigmoideDerivada(np.vstack(ones(1, columns(z2)); z2));
    
    d1 = sig2(2:end, :) * a1;
    d2 = np.transpose(sig3) * np.transpose(a2);
    
    grad1 = ((1/m) * d1);
    grad2 = ((1/m) * d2);
    
    grad1(:, 2:end) += (valorlambda/m) * Theta1(:, 2:end);
    grad2(:, 2:end) += (valorlambda/m) * Theta2(:, 2:end);
    grad = np.vstack((grad1, grad2))
  
    return grad
   # return final,grad


def main():
    #DATOS INICIALES
    num_etiquetas=1
    num_entradas=33
    num_ocultas=10 #Probar con distintos valores
    
    datamat = np.genfromtxt('student-mat-Modificado.csv', delimiter = ';')
    datapor = np.genfromtxt('student-por-Modificado.csv', delimiter = ';')
    
    #Datos de entrada
    entmat = np.vstack((datamat[:244],datamat[349:381]))
    entpor = np.vstack((datapor[:296],datapor[423:581]))
    #ent = np.vstack((entmat,entpor))
    
    #Datos de validacion
    valmat = np.vstack((datamat[244:297],datamat[381:387]))
    valpor = np.vstack((datapor[296:360],datapor[581:615]))
    #val = np.vstack((valmat,valpor))    
    
    #Datos test
    testmat = np.vstack((datamat[297:349],datamat[387:]))
    testpor = np.vstack((datapor[360:423],datapor[615:]))
    #test = np.vstack((testmat,testpor))

    #Datos de matematicas
    Xent = entmat[:,:32]
    Yent = entmat[:,32:]
    Xval = valmat[:,:32]
    Yval = valmat[:,32:]
    
    #Inicializacion de pesos aleatorios
    Theta1_ini = debugInitializeWeights(num_entradas,num_ocultas)
    Theta2_ini = debugInitializeWeights(num_ocultas,num_etiquetas)
    
    #Calculamos el error de lambda y cogemos el mejor
    errorlmdb(Xent,Yent,Xval,Yval,Theta1_ini,Theta2_ini)
    
   # params_ini=np.concatenate((Theta1_ini,Theta2_ini))
    #options = np.optimset('MaxIter', 5000);
    
    #Unimos los datos de entrenamiento y validacion
    #Xent_val=np.concatenate((Xent,Xval))
    #Yent_val=np.concatenate((Yent,Yval))
    
    #Entrenamiento de la red neuronal
    #valorlambda = 1 #El que resulte de la grafica de errorlmbd
    #bp = backprop(params_ini,num_entradas,num_ocultas,num_etiquetas,Xent_val,Yent_val,valorlambda)
    #[params_rn, J] = np.fmincg(bp, params_ini, options);
    #Theta11 = np.reshape(params_rn(1:num_ocultas * (num_entradas + 1)), num_ocultas, (num_entradas + 1));
    #Theta21 = np.reshape(params_rn((1 + (num_ocultas * (num_entradas + 1))):end), num_etiquetas, (num_ocultas + 1));
    
    
    #Porcentaje 
    #num_por = porcentajeRedNeuronal(Theta11, Theta21, Xent, Yent)
    #print(num_por)


main()
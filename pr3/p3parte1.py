# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 10:23:20 2018

@author: Fvalley
"""

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

def sigmoide(x):
    return 1/(1+ np.exp(np.negative(x)))
 
def coste(th,entradas, salidas,reg):
    #Formula vectorizada
    a = sigmoide(np.matmul(entradas, th))
    b = np.matmul(np.log(a).T,salidas)
    c = np.matmul(np.log(1-a).T,(1-salidas))
    d= (b+c)/(len(entradas)*-1)
    #Termino de regularizacion
    e = (reg/2*len(entradas))*np.sum(np.square(th))
    return d +e

def gradiente(th,entradas,salidas,reg):
    #Formula vectorizada
    y = np.reshape(salidas, salidas.shape[0])
    a=sigmoide(np.matmul(entradas,th))-y
    b = (np.matmul(entradas.T,a))/len(entradas)
    #Termino de regularizacion
    c=reg*th/len(entradas)
    c[0]= 0
    d = b + c
    return d
 
def oneVsAll(X, y, num_etiquetas, reg):
    X = np.concatenate((np.atleast_2d(np.ones(X.shape[0])).T,X),axis=1)
    entrenador = np.zeros((num_etiquetas,X.shape[1]))
    for i in range(0, num_etiquetas):
        if(i==0):
            z = 10
        else:
            z = i
        entrenador[i]= opt.fmin_tnc(coste,entrenador[i],fprime=gradiente,args=(X,(y==z)*1,reg))[0]
    result = np.matmul(entrenador,X.T)
    maximo = np.argmax(result, axis = 0)
    maximo[maximo == 0] = 10
    comparacion=(maximo==y[:,0])*1
    bienPredecidos= np.count_nonzero(comparacion)
    #Calculo del porcentaje
    porcentaje=(bienPredecidos/len(comparacion))*100
    print(porcentaje)
    

data = loadmat('ex3data1.mat')
y= data['y']
x= data['X']

oneVsAll(x,y,10,0.1)
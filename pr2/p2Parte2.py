import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.preprocessing import PolynomialFeatures

def sigmoide(x):
    return 1/(1+ np.exp(np.negative(x)))

def coste(th,entradas, salidas, alpha):
    a = sigmoide(np.matmul(entradas, th))
    b = (np.matmul(np.log(a).T,salidas) + np.matmul(np.log(1-a).T,(1-salidas)))/(len(entradas)*-1)
    c = (alpha/2*len(entradas))*np.sum(np.square(th))
    d = b+c
    sol = d
    return sol

def gradiente(th, entradas, salidas, alpha):
    a= np.matmul(entradas.T,(sigmoide(np.matmul(entradas,th))-salidas))
    b=alpha*th
    b[0]= 0
    sol = (a+b)/len(entradas)
    th= sol
    return sol 

def plot_decisionboundary(x,y,theta,poly):
    neg = np.where(y==0)
    pos = np.where(y==1)
    plt.figure()
    
    x1_min,x1_max = x[:,0].min(), x[:,0].max()
    x2_min,x2_max = x[:,1].min(), x[:,1].max()
    xx1,xx2= np.meshgrid(np.linspace(x1_min,x1_max),np.linspace(x2_min,x2_max))
    h = sigmoide(poly.fit_transform(np.c_[xx1.ravel(),xx2.ravel()]).dot(theta))
    h = h.reshape(xx1.shape)
    plt.scatter(x[pos,0],x[pos,1],marker ='+',c='k')
    plt.scatter(x[neg,0],x[neg,1],marker ='o',c='y')
    plt.contour(xx1,xx2,h,[0.5],linewidths=1,colors='g')
    plt.savefig("boundary.jpg")
    plt.close()

def regresion(data):
    x =np.genfromtxt(data, delimiter = ',')
    poly = PolynomialFeatures(6)
    ent = x[:,:-1]
    entradas = poly.fit_transform(x[:,:-1])
    th = np.zeros(entradas.shape[1])
    alpha = 1
    #entradas = np.concatenate((np.atleast_2d(unos).T,entradas),axis=1)
    salidas = x[:, -1]
    res = opt.fmin_tnc(coste,th,fprime=gradiente,args=(entradas,salidas,alpha))
   
    plot_decisionboundary(ent,salidas,res[0],poly)


    
regresion('ex2data2.csv')
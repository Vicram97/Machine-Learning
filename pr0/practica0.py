# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 16:13:24 2018

@author: usuario_local
"""
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
 
def integra_mc(fun,a,b,num_puntos=1000):
  losx =[]
  losy =[]
  vx =[]
  vy =[]
  mx=0
  nd=0
  for funcion in range(num_puntos): #Representacion de la funcion
      fx = np.random.uniform(a,b)
      fy = fun(fx)
      
      vx.append(fx)
      vy.append(fy)
      if(mx < fy):
          mx = fy
  for valor in range(num_puntos): #numeros aleatorios x e y dispersos entre a y b
      x = np.random.uniform(a,b)
      y = np.random.uniform(0,mx)
      z = fun(x)
      losx.append(x)
      losy.append(y)
      if(z > y):
          nd+=1

  
  
      
      
  #GRAFICA CON LOS DATOS   
  plt.figure()
  plt.plot(losx,losy,"x",vx,vy,'o')
  plt.show()
  resul = (nd/num_puntos)*(b-a)*mx
  print(resul)
  r = quad(fun,a,b)
  print(r)

x= lambda a: -(a*a) +2*a

integra_mc(x,0,2)
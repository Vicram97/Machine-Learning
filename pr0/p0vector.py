# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 16:13:24 2018

@author: usuario_local
"""
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
 
def integra_mc(fun,a,b,num_puntos=10000):
  mx=0
  fx = np.random.rand(num_puntos)*(b-a) + a
  fy = fun(fx)
  mx = np.amax(fy)
  x = np.random.rand(num_puntos)*(b-a) + a
  y = np.random.rand(num_puntos)*mx
  z = fun(x)
  nd = np.sum(z>y)
        
      
  #GRAFICA CON LOS DATOS   
  plt.figure()
  plt.plot(x,y,"x",fx,fy,'o')
  plt.show()
  resul = (nd/num_puntos)*(b-a)*mx
  print(resul)
  r = quad(fun,a,b)
  print(r)

x= lambda a: -(a*a) +2*a

integra_mc(x,0,2)
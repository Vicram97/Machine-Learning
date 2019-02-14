import matplotlib.pyplot as plt #Biblioteca para representar funciones
import pandas as pd #Biblioteca para análisis de datos
import numpy as np #Biblioteca para operaciones numéricas


archivo = open("ex1data1.csv") #Abrimos el archivo csv
df = pd.read_csv(archivo, header=None, names=['poblacion', 'ingresos']) #Genera un dataframe con los datos
df.head(100) #Coge los 5 primeros datos del dataframe por defecto, sirve como prueba para saber si coge bien los datos del fichero

def calc_costo(df, th0, th1):
    pob = df['poblacion']
    ing = df['ingresos']
    pred = pob * th1 + th0
    return np.sum(np.square((pred - ing)) / len(df) / 2.0)

def grad_desc(df,alpha,th0, th1):
    length = len(df)
    df['prediccion'] = df['poblacion'] * th1 + th0
    th0 = th0 - alpha / length * np.sum((df['prediccion'] - df['ingresos']))
    th1 = th1 - alpha / length * np.sum(((df['prediccion'] - df['ingresos']) * df['poblacion']))
    return th0, th1

def main(df, iter):
    alpha = 0.01
    theta0, theta1 = 0, 0
    costo = 0
    for elem in range(iter):
        theta0, theta1 = grad_desc(df, alpha,theta0, theta1)
        
    costo = calc_costo(df, theta0, theta1)

    plt.plot(df['poblacion'], df['ingresos'], 'b.', df['poblacion'], df['poblacion']*theta1 + theta0, 'r-')
    plt.title('Método del gradiente')
    plt.xlabel('Población de la ciudad en 10.000s')
    plt.ylabel('Ingresos en $10000s')
    plt.text(5, 23, 'Costo: {}'.format(costo))
    plt.show()

main(df, 1500)


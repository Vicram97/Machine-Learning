#lambda = 3
#Todo
#Porcentaje entrenamiento
#32.73972602739726
#Porcentaje validacion
#24.840764331210192
#Porcentaje test
#15.92356687898089
#Matematicas
#Porcentaje entrenamiento
#44.927536231884055
#Porcentaje validacion
#49.152542372881356
#Porcentaje test
#25.0
#Portugues
#Porcentaje entrenamiento
#35.46255506607929
#Porcentaje validacion
#23.46938775510204
#Porcentaje test
#10.309278350515463
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

def sigmoide(x):
    return 1/(1+np.exp(-x))
def h(x,th):
    return sigmoide(np.matmul(x,th))

def costeGrad(th,X,y,lambd):
    n = X.shape[0]
    grad = (1/n)*(np.matmul((h(X,th)-y[:,0]).T,X))
    grad =grad.T
    reg =(lambd/n)*th
    reg[0]= 0
    grad = grad+reg
    a = np.matmul(-y.T,np.log(h(X,th)))
    b = (np.matmul((1-y).T,np.log(1-h(X,th))).T)
    coste= (1/n)*np.sum(a-b)
    reg = (lambd/(2*n))* np.sum(th**2)
    coste = coste+reg
    return coste,grad

def oneVsAll(X,y,num_etiquetas,lambd):
    Xaux = np.hstack((np.ones((X.shape[0],1)),X))
    entrenador = np.zeros((num_etiquetas,X.shape[1]+1))
    for i in range(0,num_etiquetas):
        entrenador[i]= opt.fmin_tnc(costeGrad,entrenador[i],args=(Xaux,(y==i)*1,lambd))[0]
        #entrenador[i] = opt.minimize(coste,entrenador[i],args=(X,(y==i)*1,reg), jac=gradiente).x   
    return entrenador
def porcentaje(th,X,y):
    res = h(X,th.T)
    maximo = np.argmax(res, axis = 1)
    comp = (maximo==y[:,0])*1
    g= np.count_nonzero(comp)
    return (g/len(comp))*100
def errorlambda(X,y,Xval,yval):
    lmdb= np.array([0.001,0.003,0.01,0.03,0.1,0.3,1,3,5,10,15,30,70,120])
    n = X.shape[0]
    num_etiquetas = 21
    m=len(lmdb)
    porcent= np.zeros(m)
    porcval = np.zeros(m)
    for i in range(0,m):
        th=oneVsAll(X,y,num_etiquetas,lmdb[i])
        Xaux = np.hstack((np.ones((n,1)),X))
        porcent[i] = porcentaje(th,Xaux,y)
        Xaux = np.hstack((np.ones((Xval.shape[0],1)),Xval))
        porcval[i] = porcentaje(th,Xaux,yval)
    plt.xlabel('lambda')
    plt.ylabel('Error')
    plt.plot(lmdb,porcent,label="Entrenamiento", c='r')
    plt.plot(lmdb,porcval,label="Validacion", c= 'g')
    plt.legend()        
datamat = np.genfromtxt('student-mat-Modificado.csv', delimiter = ';')
datapor = np.genfromtxt('student-por-Modificado.csv', delimiter = ';')

#datos ordenados
entmat = np.vstack((datamat[:244],datamat[349:381]))
entpor = np.vstack((datapor[:296],datapor[423:581]))
ent = np.vstack((entmat,entpor))
valmat = np.vstack((datamat[244:297],datamat[381:387]))
valpor = np.vstack((datapor[296:360],datapor[581:615]))
val = np.vstack((valmat,valpor))
testmat = np.vstack((datamat[297:349],datamat[387:]))
testpor = np.vstack((datapor[360:423],datapor[615:]))
test = np.vstack((testmat,testpor))

Xmat = entmat[:,:32]
Xpor = entpor[:,:32]
X = ent[:,:32]
ymat = entmat[:,32:]
ypor = entpor[:,32:]
y = ent[:,32:]

Xvalmat = valmat[:,:32]
Xvalpor = valpor[:,:32]
Xval = val[:,:32]
yvalmat = valmat[:,32:]
yvalpor = valpor[:,32:]
yval = val[:,32:]

Xtestmat = testmat[:,:32]
Xtestpor = testpor[:,:32]
Xtest = test[:,:32]
ytestmat = testmat[:,32:]
ytestpor = testpor[:,32:]
ytest = test[:,32:]
errorlambda(X,y,Xval,yval)

Xent_val = np.concatenate((X,Xval))
yent_val = np.concatenate((y,yval))
print("Todo")
th = oneVsAll(Xent_val,yent_val,21,0)
Xp = np.concatenate((np.atleast_2d(np.ones(X.shape[0])).T,X),axis=1)
p=porcentaje(th,Xp,y)
print("Porcentaje entrenamiento")
print(p)

Xpval = np.concatenate((np.atleast_2d(np.ones(Xval.shape[0])).T,Xval),axis=1)
pval=porcentaje(th,Xpval,yval)
print("Porcentaje validacion")
print(pval)

Xptest = np.concatenate((np.atleast_2d(np.ones(Xtest.shape[0])).T,Xtest),axis=1)
ptest=porcentaje(th,Xptest,ytest)
print("Porcentaje test")
print(ptest)

Xent_valmat = np.concatenate((Xmat,Xvalmat))
yent_valmat = np.concatenate((ymat,yvalmat))
print("Matematicas")
th = oneVsAll(Xent_valmat,yent_valmat,21,3)
Xpmat = np.concatenate((np.atleast_2d(np.ones(Xmat.shape[0])).T,Xmat),axis=1)
p=porcentaje(th,Xpmat,ymat)
print("Porcentaje entrenamiento")
print(p)

Xpvalmat = np.concatenate((np.atleast_2d(np.ones(Xvalmat.shape[0])).T,Xvalmat),axis=1)
pval=porcentaje(th,Xpvalmat,yvalmat)
print("Porcentaje validacion")
print(pval)

Xptestmat = np.concatenate((np.atleast_2d(np.ones(Xtestmat.shape[0])).T,Xtestmat),axis=1)
ptest=porcentaje(th,Xptestmat,ytestmat)
print("Porcentaje test")
print(ptest)

Xent_valpor = np.concatenate((Xpor,Xvalpor))
yent_valpor = np.concatenate((ypor,yvalpor))

print("Portugues")
th = oneVsAll(Xent_valpor,yent_valpor,21,3)
Xppor = np.concatenate((np.atleast_2d(np.ones(Xpor.shape[0])).T,Xpor),axis=1)
p=porcentaje(th,Xppor,ypor)
print("Porcentaje entrenamiento")
print(p)

Xpvalpor = np.concatenate((np.atleast_2d(np.ones(Xvalpor.shape[0])).T,Xvalpor),axis=1)
pval=porcentaje(th,Xpvalpor,yvalpor)
print("Porcentaje validacion")
print(pval)

Xptestpor = np.concatenate((np.atleast_2d(np.ones(Xtestpor.shape[0])).T,Xtestpor),axis=1)
ptest=porcentaje(th,Xptestpor,ytestpor)
print("Porcentaje test")
print(ptest)

import numpy as np
from sklearn.svm import SVC
#import matplotlib.pyplot as plt

##def pintar(X,y,svm):
# #   neg = np.where(y==0)
#  #  pos = np.where(y==1)
#   # plt.figure()
#    
#    #x1_min,x1_max = X[:,0].min(), X[:,0].max()
#    #x2_min,x2_max = X[:,1].min(), X[:,1].max()
#    xx1,xx2= np.meshgrid(np.linspace(x1_min,x1_max),np.linspace(x2_min,x2_max))
#    Z = svm.predict(np.c_[xx1.ravel(), xx2.ravel()])
#    Z = Z.reshape(xx1.shape)
#    plt.scatter(X[pos,0],X[pos,1],marker ='+',c='k')
#    plt.scatter(X[neg,0],X[neg,1],marker ='o',c='y')
#    plt.contour(xx1,xx2,Z,[0.5],linewidths=1,colors='g')

def supportv(Xent,yent,Xval,yval):
    val = np.array([0.01,0.03,0.05,0.1,0.3,0.5,1,3,5,10,15,30,50,100,150,300])
    maxilin = 0
    Csollin = 0
    for i in range(0,val.shape[0]):
        svm = SVC( kernel='linear', C=val[i])
        svm.fit(Xent,yent)
        w = svm.predict(Xval)        
        t = (w==yval[:,0])
        p = (np.count_nonzero(t)/yval.shape[0])*100
        #text = 'C='+repr(val[i])+'.Porcentaje='+repr(p)
        if(p>maxilin):
            Csollin = val[i]
            maxilin = p
        #print(text)
    textlin = 'Mejor solucion lineal: C = '+ repr(Csollin)+ ' . % = ' +repr(maxilin)
    maxigaus = 0
    Csolgaus = 0
    sigmasolgaus= 0
    for i in range(0,val.shape[0]):
        for j in range(0,val.shape[0]):
            svm = SVC( kernel='rbf', C=val[i], gamma = 1/(2*val[j]**2))
            svm.fit(Xent,yent)
            w = svm.predict(Xval)
            t = (w==yval[:,0])
            p = (np.count_nonzero(t)/yval.shape[0])*100
            #text = 'C='+repr(val[i])+',sigma='+repr(val[j])+' .Porcentaje='+repr(p)
            if(p>maxigaus):
              Csolgaus = val[i]
              sigmasolgaus = val[j]
              maxigaus = p
            #print(text)
    text = 'Mejor solucion gaussiana: C = '+ repr(Csolgaus)+', Sigma = '+repr(sigmasolgaus)+ ' . % = ' +repr(maxigaus)
    print(textlin)
    print(text)
    

datamat = np.genfromtxt('student-mat-Modificado.csv', delimiter = ';')
datapor = np.genfromtxt('student-por-Modificado.csv', delimiter = ';')

entmat = np.vstack((datamat[:244],datamat[349:381]))
entpor = np.vstack((datapor[:296],datapor[423:581]))
ent = np.vstack((entmat,entpor))
valmat = np.vstack((datamat[244:297],datamat[381:387]))
valpor = np.vstack((datapor[296:360],datapor[581:615]))
val = np.vstack((valmat,valpor))
testmat = np.vstack((datamat[297:349],datamat[387:]))
testpor = np.vstack((datapor[360:423],datapor[615:]))
test = np.vstack((testmat,testpor))

print("Solo matematicas")
#lin C= 0.05 51,666%
#Gaus C=50 Sigma = 30 53,333%
X = entmat[:,:32]
y = entmat[:,32:]
Xval = valmat[:,:32]
yval = valmat[:,32:]
Xtest = testmat[:,:32]
ytest = testmat[:,32:]
Xr = np.vstack((X,Xval))
yr = np.vstack((y, yval))
yr = np.reshape(yr,yr.shape[0])
supportv(Xr,yr,Xtest,ytest)

print("Solo matematicas sin G1 y G2")
#lin C =0.05 26,666%
#Gaus C = 15 Sigma = 30 28,333%
X = entmat[:,:30]
Xval = valmat[:,:30]
Xtest = testmat[:,:30]
Xr = np.vstack((X,Xval))
supportv(Xr,yr,Xtest,ytest)

print("Solo portugues")
#lin C =0.1 39,17%
#Gaus C = 300 Sigma = 50 38,14%
X = entpor[:,:32]
y = entpor[:,32:]
Xval = valpor[:,:32]
yval = valpor[:,32:]
Xtest = testpor[:,:32]
ytest = testpor[:,32:]
Xr = np.vstack((X,Xval))
yr = np.vstack((y, yval))
yr = np.reshape(yr,yr.shape[0])
supportv(Xr,yr,Xtest,ytest)

print("Solo portugues sin G1 y G2")
#lin C =0.01 18,556%
#Gaus C = 300 Sigma = 50 20,62%
X = entpor[:,:30]
Xval = valpor[:,:30]
Xtest = testpor[:,:30]
Xr = np.vstack((X,Xval))
supportv(Xr,yr,Xtest,ytest)

print("Todo")
#lin C =10 45,22%
#Gaus C = 300 Sigma = 50 39,49%
X = ent[:,:32]
y = ent[:,32:]
Xval = val[:,:32]
yval = val[:,32:]
Xtest = test[:,:32]
ytest = test[:,32:]
Xr = np.vstack((X,Xval))
yr = np.vstack((y, yval))
yr = np.reshape(yr,yr.shape[0])
supportv(Xr,yr,Xtest,ytest)

print("Todo sin G1 y G2")
#lin C =0.03 15,92%
#Gaus C = 100 Sigma = 5 15,92%
X = ent[:,:30]
Xval = val[:,:30]
Xtest = test[:,:30]
Xr = np.vstack((X,Xval))
supportv(Xr,yr,Xtest,ytest)
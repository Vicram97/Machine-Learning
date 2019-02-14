import numpy as np
from sklearn.svm import SVC
import scipy.io
import matplotlib.pyplot as plt
from process_email import email2TokenList
from get_vocab_dict import getVocabDict
import codecs

def pintar(X,y,svm):
    neg = np.where(y==0)
    pos = np.where(y==1)
    plt.figure()
    
    x1_min,x1_max = X[:,0].min(), X[:,0].max()
    x2_min,x2_max = X[:,1].min(), X[:,1].max()
    xx1,xx2= np.meshgrid(np.linspace(x1_min,x1_max),np.linspace(x2_min,x2_max))
    Z = svm.predict(np.c_[xx1.ravel(), xx2.ravel()])
    Z = Z.reshape(xx1.shape)
    plt.scatter(X[pos,0],X[pos,1],marker ='+',c='k')
    plt.scatter(X[neg,0],X[neg,1],marker ='o',c='y')
    plt.contour(xx1,xx2,Z,[0.5],linewidths=1,colors='g')
    
def primerapartado():
    data = scipy.io.loadmat('ex6data1.mat')
    y = data['y']
    X = data['X']
    y = np.reshape(y,(51))
    svm = SVC( kernel='linear', C=1.0)
    svm.fit(X,y)
    pintar(X,y,svm)
    
    
def segundoapartado():
    data = scipy.io.loadmat('ex6data2.mat')
    
    y = data['y']
    X = data['X']
    y = np.reshape(y,y.shape[0])
    svm = SVC( kernel='rbf', C=1.0, gamma = 1/(2*0.1**2))
    svm.fit(X,y)
    pintar(X,y,svm)
    
def tercerapartado():
    data = scipy.io.loadmat('ex6data3.mat')
    
    y = data['y']
    X = data['X']
    yval = data['yval']
    Xval = data['Xval']
    y = np.reshape(y,y.shape[0])
    a = np.array([0.01,0.03,0.1,0.3,1,3,10,30])
    
    maxi = 0
    Csol = 0
    sigmasol= 0
    for i in range(0,a.shape[0]):
        for j in range(0,a.shape[0]):
            svm = SVC( kernel='rbf', C=a[i], gamma = 1/(2*a[j]**2))
            svm.fit(X,y)
            w = svm.predict(Xval)
            t = (w==yval[:,0])
            p = (np.count_nonzero(t)/yval.shape[0])*100
            text = 'C='+repr(a[i])+',sigma='+repr(a[j])+' .Porcentaje='+repr(p)
            if(p>maxi):
              Csol = a[i]
              sigmasol = a[j]
              maxi = p
            print(text)
    
    text = 'Mejor solucion: C = '+ repr(Csol)+', Sigma = '+repr(sigmasol)+ ' . % = ' +repr(maxi)
    print(text)
    
    svm = SVC(kernel='rbf', C=Csol, gamma = 1/(2*sigmasol**2))
    svm.fit(X,y)
    pintar(X,y,svm)

def cargar(directorio,numcorreos,vocdic,eSpam):
    X = np.empty((numcorreos, 1899))
    if eSpam:
        y = np.ones((numcorreos,1))
    else:
        y = np.zeros((numcorreos,1))
    frozenvoc = frozenset(vocdic)

    for i in range(1,numcorreos):
        email_contents = codecs.open( '{0}/{1:04d}.txt'.format(directorio,i),'r',encoding='utf-8', errors='ignore').read()
        email = email2TokenList(email_contents)
        for j in email:
            if j in frozenvoc:
                X[i,(vocdic.get(j)-1)] = 1
            
    return X,y
def email():
    dic = getVocabDict()
    val = np.array([0.01,0.03,0.1,0.3,1,3,10,30])
    spamX,spamy = cargar('spam',500,dic,1)
    easyX,easyy = cargar('easy_ham',2551,dic,0)
    hardX,hardy = cargar('hard_ham',250,dic,0)
    
    Xent = np.vstack((spamX[:350],easyX[:1786],hardX[:175]))
    yent = np.vstack((spamy[:350],easyy[:1786],hardy[:175]))
    Xval = np.vstack((spamX[350:],easyX[1786:],hardX[175:]))
    yval = np.vstack((spamy[350:],easyy[1786:],hardy[175:]))
    
    maxilin = 0
    Csollin = 0
    for i in range(0,val.shape[0]):
        svm = SVC( kernel='linear', C=val[i])
        svm.fit(Xent,yent)
        w = svm.predict(Xval)        
        t = (w==yval[:,0])
        p = (np.count_nonzero(t)/yval.shape[0])*100
        text = 'C='+repr(val[i])+'.Porcentaje='+repr(p)
        if(p>maxilin):
            Csollin = val[i]
            maxilin = p
        print(text)
    text = 'Mejor solucion lineal: C = '+ repr(Csollin)+ ' . % = ' +repr(maxilin)
    print(text)
    
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
            text = 'C='+repr(val[i])+',sigma='+repr(val[j])+' .Porcentaje='+repr(p)
            if(p>maxigaus):
              Csolgaus = val[i]
              sigmasolgaus = val[j]
              maxigaus = p
            print(text)
    text = 'Mejor solucion gaussiana: C = '+ repr(Csolgaus)+', Sigma = '+repr(sigmasolgaus)+ ' . % = ' +repr(maxigaus)
    print(text)
def main():
    
    #primerapartado()
    #segundoapartado()
    #tercerapartado()
    email()
main()
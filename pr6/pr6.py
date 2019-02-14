# -*- coding: utf-8 -*-
from sklearn.svm import SVC
import scipy.io

def main():
    data = scipy.io.loadmat('ex6data1.mat')
    y = data['y']
    X = data['X']
    svm = SVC(kernel='linear',C=1.0)
    svm.fit(X,y)

main()
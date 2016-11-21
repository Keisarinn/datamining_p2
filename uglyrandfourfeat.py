# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 00:38:59 2016

@author: Siggi
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 16:57:14 2016

@author: Siggi
"""

import numpy as np


lam = 0.0000001
gamm = 100
c = 0
m = 800
n = 400
#eta = 0.001
def transform(X):
    global m
    global n
    intt = int(m/2)
    np.random.seed(500)
    b = np.random.uniform(low=0.0, high=2*np.pi, size=intt)
    np.random.seed(500+intt)
    w = np.random.normal(loc=0.0, scale=1.0, size=n*intt)
    XX = np.mat(X)
    if np.ndim(X) == 1:
        A = np.zeros([m], dtype = 'float')
        for i in range(intt):
            A[i] = np.cos(np.dot(X,w[i*n:(i+1)*n])+b[i])
            A[intt+i] = np.cos(-np.dot(X,w[i*n:(i+1)*n])+b[i])
    else:
        A = np.zeros([np.shape(X)[0],m], dtype= 'float')
        print(np.shape(XX*np.transpose(np.mat(w[0*n:(0+1)*n]))))
        for i in range(intt):
            #print(np.shape(np.dot(X,w[i*n:(i+1)*n])+b[i]))
            A[:,i] = np.cos(np.dot(XX,np.transpose(np.mat(w[i*n:(i+1)*n])))+b[i])
            A[:,intt+i] = np.cos(-np.dot(XX,np.transpose(np.mat(w[i*n:(i+1)*n])))+b[i])
            #print(i)
    # Make sure this function works for both 1D and 2D NumPy arrays.
    #print(A[:,400])
    return 10*(np.sqrt(2/m)*A+0.05)


def mapper(key, value):
    # key: None
    # value: one line of input file
    global lam
    global gamm
    #global eta
    #size = np.shape(value)[1]
    #size = len(value)
    #dat = np.empty((0,0))
    dat = np.zeros([len(value),401], dtype='f')
    count = 0
    for i in value:
        tokens = i.split()
        x = np.asarray(tokens[0:]).astype(np.float)
        dat[count,:] = x
        count = count + 1
    #dat = np.array(value).astype(np.float)
    
    print(np.shape(dat))
    randindex = np.random.permutation(np.shape(dat)[0])
    cnt = 0.0
    #print(value)
    ratio = 0.0
    A = transform(dat[:,1:])
    w = np.zeros([np.shape(A)[1]], dtype='float')

    #count = 0.0
    for i in range(np.shape(dat)[0]):
        #print(i)
        #tokens = i.split()
        #y = float(tokens[0])
        #x = np.asarray(tokens[1:]).astype(np.float)
        y = dat[randindex[i],0]
        x = A[randindex[i],:]
        print(x)
        cnt = cnt + 1.0
        eta = gamm/(1+lam*gamm*np.sqrt(cnt))
        #eta = 1/(lam*(cnt))
        #print(np.shape(w))
        #print(np.shape(x))
        #print(w+(y/np.sqrt(cnt))*x)
        bina = 0.0
        if (y*np.inner(w,x)) < 1:
            bina = 1.0
            print(bina)
            ratio = ratio + 1.0
            #print(y*np.inner(w,x))
        w = (1-lam*eta)*w+bina*(y*eta)*x
            #print(1/(np.sqrt(lam)*np.linalg.norm(wnew)))
            #w = np.minimum(1,1/(np.sqrt(lam)*np.linalg.norm(wnew)))*wnew
            #print(w)
            #print(np.shape(w))
    print((w))
    print(ratio/cnt)
    yield 1,w

def reducer(key, values):
    wfin = 0.0
    count = 0.0
    for i in values:
        wfin = wfin + i
        print(wfin)
        count = count + 1.0
    print(wfin/count)
    yield wfin/count

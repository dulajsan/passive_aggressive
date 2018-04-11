#developer: dulaj sanjaya
#python 3.4

import numpy as np
import sys
from pylab import *

def loadData():
    dt=[]
    for line in open("datafile.csv"):
        csv_row=line.strip().split(',')
        csv_row=csv_row[1:]
        if(csv_row[9]=='2'):
            csv_row[9]= '+1'
        else:
            csv_row[9]= '-1'
        dt.append(csv_row)
                 
    return dt

def method1(loss,c,x_square):
    return loss/x_square


def PA(X,label,C,t):
    n=len(X[0,:])
    w=np.zeros(n)
    label=[int(i) for i in label]
    

    for i in range(t):
        q=0
        for xt in X:
            #w=w.astype(float)
            #xt=xt.astype(float)
            xt=[int(i) for i in xt]
            
            y_dash= np.dot(w,xt)
            yy_dash=label[q]*y_dash

            loss=max([0,1-yy_dash])
            x_square=(np.power(np.linalg.norm(xt, ord=2), 2))

            tor=method1(loss,C,x_square)
            xt=[float(i) for i in xt]
            s=tor*label[q]
            temp=[x*s for x in xt]
            w=w+temp
            
            q=q+1

    return w

def accuracy(correct,predicted):
    count=0
    n=len(correct)
    for i in range(n):
        if correct[i]==predicted[i]:
            count=count+1
    accuracy=(count/n)*100
    return accuracy
    
    
    
    

if __name__ == '__main__':
    C=1
    data= np.array(loadData())
    Y=data[:,-1]
    X=data[:,:9]
    n=int(len(X)*(2/3))

    t=int(input("enter number of iterations: "))

    trainX=X[:n,:] #data for the training
    testX=X[n:,:] ##unseen data for testing

    TY=Y[:n] #defined target value for updating the w
    testY=Y[n:] #defined target values for testing

    W=PA(trainX,TY,C,t) #updated w matrix

    resultlabeld=[] #predicted label for unseen data
    resultlabeld2=[]  #predicted label for previously seen data

    #predicted values for the testing data
    for x in testX:
        x=[int(t) for t in x]
        resultlabeld.append(np.sign(np.dot(W,x)))
    resultlabeld=[int(i) for i in resultlabeld]
    #predicted labeled
    
    testY=[int(i) for i in testY]

    print("testing accuracy: ",accuracy(testY,resultlabeld))
    print('\n')


    ##accuracy testing for previosly seen data
    ##predicted values for training data

    for x in trainX:
        x=[int(t) for t in x]
        resultlabeld2.append(np.sign(np.dot(W,x)))
    resultlabeld2=[int(i) for i in resultlabeld2]

    TY=[int(i) for i in TY]

    print("training accuracy: ",accuracy(TY,resultlabeld2))
    
        

    

   

    
    
    

 
    
    

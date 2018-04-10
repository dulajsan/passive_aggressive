#developer: dulaj sanjaya
#python 3.4

import numpy as np

def loadData():
    dt=[]
    for line in open("datafile.csv"):
        csv_row=line.strip().split(',')
        csv_row=csv_row[1:]
        if(csv_row[9]=='2'):
            csv_row[9]= '1'
        else:
            csv_row[9]= '-1'
        dt.append(csv_row)
                 
    return dt

def method1(loss,c,x_square):
    return loss/x_square

def PA(X,label,C,t):
    n=len(X[0,:])
    w=np.zeros(n)

    for i in range(t):
        q=0
        for xt in X:
            w=w.astype(float)
            xt=xt.astype(float)
          
            
            y_dash= np.sign(np.dot(w,xt))
            yy_dash=float(label[q])*y_dash

            loss=max([0,1-yy_dash])
            x_square=np.sum(xt)

            tor=method1(loss,C,x_square)
            w=w+float(label[q])*tor*xt
            
            q=q+1

    return w
    

if __name__ == '__main__':
    C=1
    data= np.array(loadData())
    Y=data[:,-1]
    X=data[:,:9]
    n=int(len(X)*(2/3))

    trainX=X[:n,:] #data for the training
    testX=X[n:,:] ##unseen data for testing

    TY=Y[:n] #target value for building the w
    testY=Y[n:] #target values for testing

    W=PA(trainX,TY,C,1)
    #updated w matrix

    testinglabeld=[]
    ##label for unseen data

    for x in testX:
        testinglabeld.append(np.sign(np.dot(W,x.astype(float))))
    testinglabeld=[int(i) for i in testinglabeld]
    testY=[int(i) for i in testY]

   

    
    
    

 
    
    

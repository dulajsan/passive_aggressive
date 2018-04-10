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


    
    

if __name__ == '__main__':
    C=1
    data= np.array(loadData())
    Y=data[:,-1]
    X=data[:,:9]
    n=int(len(X)*(2/3))

    trainX=X[:n,:]
    testX=X[n:,:]

    TY=Y[:n]
    

 
    
    

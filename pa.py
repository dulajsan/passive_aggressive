import numpy as np

def loadData():
    dt=[]
    for line in open("datasmall.csv"):
        csv_row=line.strip().split(',')
        csv_row=csv_row[1:]
        if(csv_row[9]=='2'):
            csv_row[9]= '1'
        else:
            csv_row[9]= '-1'
        dt.append(csv_row)
            
        
    return dt
    
    

if __name__ == '__main__':
    data= np.array(loadData())
    print(data)
    print('\n')
    X=data[:,-1]
    Y=data[:,:9]
    print(Y)
    
    

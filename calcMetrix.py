import numpy as np
import torch

def calcMSET(x,y,weight = None):
    x = x.float()
    y = y.float()
    if weight is None : 
        return torch.sqrt(torch.mean((((x - y) ** 2)+.0001)))
    else : 
        return torch.sqrt(torch.mean(((((x - y) ** 2)*weight+.0001))))

    
def calcCORT(data1, data2, weight = None):
    "data1 & data2 should be numpy arrays."
    data1 = data1.float()
    data2 = data2.float()
    
    mean1 = torch.mean(data1)+.0001
    mean2 = torch.mean(data2)+.0001
    std1 = torch.std(data1)+.0001
    std2 = torch.std(data2)+.0001
    
    #print('std1,std2',std1,std2,mean1,mean2)

    if weight is None : 
        corr = (torch.mean((data1-mean1)*(data2-mean2))/(std1*std2) )
    else : 
        corr = (torch.mean(((data1-mean1)*(data2-mean2))*weight)/(std1*std2))
    #corr = ((data1*data2).mean()-mean1*mean2)/(std1*std2)
    return corr


def calcICCT(data1, data2, weight = None):
    "data1 & data2 should be numpy arrays."
    data1 = data1.float()
    data2 = data2.float()
    
    mean1 = torch.mean(data1) +.0001
    mean2 = torch.mean(data2)+.0001
    std1 = torch.std(data1)+.0001
    std2 = torch.std(data2)+.0001

    #icc = 2 * ((data1-mean1)*(data2-mean2)).mean() / (std1*std1+std2*std2)
    if weight is None : 
        icc = ((2*(torch.mean(data1*data2)-mean1*mean2)) /((std1*std1)+(std2*std2)))
    else : 
        icc = ((2*(torch.mean(data1*data2*weight)-mean1*mean2)) /((std1*std1)+(std2*std2)))
    return icc


def calcCCCT(data1, data2,weight = None):
    "data1 & data2 should be numpy arrays."
    data1 = data1.float()
    data2 = data2.float()
    
    mean1 = torch.mean(data1) +.0001
    mean2 = torch.mean(data2)+.0001
    std1 = torch.std(data1)+.0001
    std2 = torch.std(data2)+.0001
    dm = mean1 - mean2

    #print('dm',dm)
    
    #ICC = 2 * ((data1-mean1)*(data2-mean2)).mean() / (std1^2+std2^2)
    #ccc = (2*((data1*data2).mean()-mean1*mean2)) /((std1*std1)+(std2*std2)+(dm*dm))
    if weight is None : 
        ccc = ((2*(torch.mean((data1-mean1)*(data2-mean2)))) /((std1*std1)+(std2*std2)+(dm*dm)))
    else : 
        ccc = ((2*(torch.mean(((data1-mean1)*(data2-mean2))*weight))) /((std1*std1)+(std2*std2)+(dm*dm)))
    return ccc


def calcMSE(x,y):
    return np.sqrt(((x - y) ** 2).mean())
    
def calcCOR(data1, data2):
    "data1 & data2 should be numpy arrays."
    mean1 = data1.mean() 
    mean2 = data2.mean()
    std1 = data1.std()
    std2 = data2.std()
    print('std1,std2',std1,std2,mean1,mean2)

    corr = ((data1-mean1)*(data2-mean2)).mean()/(std1*std2)
    #corr = ((data1*data2).mean()-mean1*mean2)/(std1*std2)
    return corr


def calcICC(data1, data2):
    "data1 & data2 should be numpy arrays."
    mean1 = data1.mean() 
    mean2 = data2.mean()
    std1 = data1.std()
    std2 = data2.std()

    #icc = 2 * ((data1-mean1)*(data2-mean2)).mean() / (std1*std1+std2*std2)
    icc = (2*((data1*data2).mean()-mean1*mean2)) /((std1*std1)+(std2*std2))
    return icc


def calcCCC(data1, data2):
    "data1 & data2 should be numpy arrays."
    mean1 = data1.mean() 
    mean2 = data2.mean()
    std1 = data1.std()
    std2 = data2.std()
    dm = mean1 - mean2
    

    #ICC = 2 * ((data1-mean1)*(data2-mean2)).mean() / (std1^2+std2^2)
    #ccc = (2*((data1*data2).mean()-mean1*mean2)) /((std1*std1)+(std2*std2)+(dm*dm))
    ccc = (2*((data1-mean1)*(data2-mean2)).mean()) /((std1*std1)+(std2*std2)+(dm*dm))
    return ccc


def calcMSE_B(data1,data2):
    res = []
    for x,y in zip(data1,data2): 
        res.append(np.sqrt(((x - y) ** 2).mean()))
    return(np.asarray(res))
    
def calcCOR_B(x, y):
    "data1 & data2 should be numpy arrays."
    res = []
    for data1,data2 in zip(x,y) : 
    
        mean1 = data1.mean() 
        mean2 = data2.mean()
        std1 = data1.std()
        std2 = data2.std()
    
        #corr = ((data1-mean1)*(data2-mean2)).mean()/(std1*std2)
        corr = ((data1*data2).mean()-mean1*mean2)/(std1*std2)
        res.append(corr)
    return (np.asarray(res))

def calcICC_B(x, y):
    "data1 & data2 should be numpy arrays."
    res = []
    for data1, data2 in zip(x,y) : 
        mean1 = data1.mean() 
        mean2 = data2.mean()
        std1 = data1.std()
        std2 = data2.std()
    
        #ICC = 2 * ((data1-mean1)*(data2-mean2)).mean() / (std1^2+std2^2)
        icc = (2*((data1*data2).mean()-mean1*mean2)) /((std1*std1)+(std2+std2))
        res.append(icc)
    return (np.asarray(res))

def main():
    
    
    x = torch.tensor(np.array([1,2,3,4,5]))
    #y = np.array([1,2,3,4,5])
    y = torch.tensor(np.array([5,4,3,2,1]))
    
    w = torch.tensor([0.1,0.5,1,0.5,0.1])
    
    print(x,y)
    print(calcMSET(x, y))
    print(calcCORT(x, y))
    print(calcICCT(x, y))
    print(calcCCCT(x, y))
    
    print('#'*10)
    
    print(x,y)
    print(calcMSET(x, y,w))
    print(calcCORT(x, y,w))
    print(calcICCT(x, y,w))
    print(calcCCCT(x, y,w))
    
    exit(0)
    print('*'*10)
    
    print(calcMSET(x, x))
    print(calcCORT(x, x))
    print(calcICCT(x, x))
    print(calcCCCT(x, x))
    
    
    
    
    x = np.array([1,2,3,4,5])
    #y = np.array([1,2,3,4,5])
    y = np.array([5,4,3,2,1])
    
    print(x,y)
    print(calcMSE(x, y))
    print(calcCOR(x, y))
    print(calcICC(x, y))
    print(calcCCC(x, y))
    
    exit(0)
    
    
    print('*'*5)
    
    print(calcMSE(x, x))
    print(calcCOR(x, x))
    print(calcICC(x, x))
    print(calcCCC(x, x))
    
    print('*'*5)
    
    x_b = np.array([[1,2,3,4,5],[1,2,3,4,5]])
    y_b = np.array([[5,4,3,2,1],[1,2,3,4,5]])
    
    
    print(calcMSE_B(x_b, y_b))
    print(calcCOR_B(x_b, y_b))
    print(calcICC_B(x_b, y_b))
    
    
    print(calcMSE_B(x_b, x_b))
    print(calcCOR_B(x_b, x_b))
    print(calcICC_B(x_b, x_b))

#main()

import numpy as np
 
# y_true = np.array([0, 1, 1, 0, 1, 0])
# y_pred = np.array([1, 1, 1, 0, 0, 1])
 

def getTP(y_true,y_pred):
    #true positive
    TP = np.sum(np.logical_and(np.equal(y_true,1),np.equal(y_pred,1)))
    print(TP)


def getFP(y_true,y_pred):
    #true positive
    #false positive
    FP = np.sum(np.logical_and(np.equal(y_true,0),np.equal(y_pred,1)))
    print(FP)


def getTN(y_true,y_pred):
    #true positive
    #false positive
    TN = np.sum(np.logical_and(np.equal(y_true,1),np.equal(y_pred,0)))
    print(TN)
    return TN
 
def getFN(y_true,y_pred):
    #true positive
    #false positive
    FN =np.sum(np.logical_and(np.equal(y_true,1),np.equal(y_pred,0)))
    print(FN)
    return FN

#false positive
# FP = np.sum(np.logical_and(np.equal(y_true,0),np.equal(y_pred,1)))
# print(FP)
 
# #true negative
# TN = np.sum(np.logical_and(np.equal(y_true,1),np.equal(y_pred,0)))
# print(TN)
 
# #false negative
# # FN = np.sum(np.logical_and(np.equal(y_true,0),np.equal(y_pred,0)))
# FN =np.sum(np.logical_and(np.equal(y_true,1),np.equal(y_pred,0)))
# print(FN)
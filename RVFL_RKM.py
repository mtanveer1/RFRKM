import numpy as np
import time


import numpy as np

def calculate_kA(A, C, mew):
    norm_A = np.linalg.norm(A, axis=1)[:, np.newaxis]
    norm_C = np.linalg.norm(C, axis=1)

    first_term = np.power(norm_A, 2)
    second_term = -2 * np.dot(A, C.T)
    third_term = np.power(norm_C, 2)

    kA = np.exp(-(1 / (2 * mew)) * (first_term + second_term + third_term))
    return kA


def Evaluate(ACTUAL, PREDICTED):
    idx = (ACTUAL == 1)
    p = np.sum(idx)
    n = np.sum(~idx)
    N = p + n
    tp = np.sum(np.logical_and(ACTUAL[idx] == 1, PREDICTED[idx] == 1))
    tn = np.sum(np.logical_and(ACTUAL[~idx] == 0, PREDICTED[~idx] == 0))
    fp = n - tn
    fn = p - tp
    tp_rate = tp / p if p != 0 else 0
    tn_rate = tn / n if n != 0 else 0
    accuracy = 100 * (tp + tn) / N
    sensitivity = 100 * tp_rate
    specificity = 100 * tn_rate
    precision = 100 * tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = sensitivity
    f_measure = 2 * ((precision * sensitivity) / (precision + sensitivity)) if (precision + sensitivity) != 0 else 0
    gmean = 100 * np.sqrt(tp_rate * tn_rate)
    
    EVAL = [accuracy, sensitivity, specificity, precision, recall, f_measure, gmean, tp, tn, fp, fn]
    return EVAL

def one_hot(x, n_class):
    y = np.zeros([len(x), n_class])
    U_dataY_train = np.array([0,1])
    for i in range(n_class):
        idx= (x == U_dataY_train[i])
        y[idx,i]=1
    return y

def RVFL_RKM(train_data, test_data, C,C1,sig):

    start = time.time()
    trainX=train_data[:,:-1]
    trainY=train_data[:,-1]

    nclass = 2
    dataY_train_temp = one_hot(trainY,nclass)

    XX = np.dot(trainX, trainX.T)

    X=calculate_kA(trainX, trainX, sig) + XX

    beta = np.dot(np.linalg.inv((1/C)*X+C1*np.eye(X.shape[1])), dataY_train_temp)
    
    # Test

    X=test_data[:,:-1]
    Y=test_data[:,-1]

    # Test Data

    X2=calculate_kA(X, trainX, sig) + np.dot(X, trainX.T)

    rawScore = (1/C)*np.dot(X2, beta)

    Validation_label = np.argmax(rawScore, axis=1) 

    EVAL_Validation = Evaluate(Y, Validation_label)
    end = time.time()
    Time=end - start
    return EVAL_Validation,Time

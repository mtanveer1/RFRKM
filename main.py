import os
import numpy as np
from RVFL_RKM import RVFL_RKM
from sklearn.model_selection import KFold


directory = '.'
file_list = os.listdir(directory)

for file_name in file_list:
    if file_name.endswith(".csv"):
        file_path = os.path.join(directory, file_name)
        print(directory)
        print(file_name)
        file_data = np.loadtxt(file_path, delimiter=',')
        
        m, n = file_data.shape
        for i in range(m):
            if file_data[i, n-1] == -1:
                file_data[i, n-1] = 0
    
        np.random.seed(0)
        indices = np.random.permutation(m)
        file_data = file_data[indices]
        A_train=file_data[0:int(m*(1-0.30))]
        A_test=file_data[int(m * (1-0.30)):]

        m, n = A_train.shape
        indices = np.random.permutation(m)
        A_train = A_train[indices]

        x1 = A_train[:, 0:n-1]
        y1 = A_train[:, n-1]
        xtest0 = A_test[:,0:n-1]
        ytest0 = A_test[:,n-1]
        # Normalize the data training and testing
        me = np.tile(np.mean(x1, axis=0), (x1.shape[0], 1))
        st = np.tile(np.std(x1, axis=0), (x1.shape[0], 1))
        tme = np.tile(np.mean(x1, axis=0), (xtest0.shape[0], 1))
        tst = np.tile(np.std(x1, axis=0), (xtest0.shape[0], 1))
        x1 = (x1 - me) / st
        xtest0 = (xtest0 - tme) / tst

        A_test = np.hstack((xtest0, ytest0.reshape(ytest0.shape[0], 1)))
        A_train = np.hstack((x1, y1.reshape(y1.shape[0], 1)))
    
        
        C = 10**-5
        c_1 = 10**-5
        sig = 2**2
        Eval, time = RVFL_RKM(A_train, A_test, C, c_1,sig)
                        
        Test_accuracy=Eval[0]
        print("Test accuracy: ", Test_accuracy)

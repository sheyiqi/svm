#!/home/yiqi.she/venv/common/bin/python3
import sys, os
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import KFold

data1 = np.loadtxt("d1", dtype = np.float)
X1 = data1[:,:-1]
y1 = data1[:,-1:]
kfold1 = KFold(9, n_folds=9,shuffle=True)
data2 = np.loadtxt("d2", dtype = np.float)
X2 = data2[:,:-1]
y2 = data2[:,-1:]
data3 = np.loadtxt("d3", dtype = np.float)
X3 = data3[:,:-1]
y3 = data3[:,-1:]
data4 = np.loadtxt("d4", dtype = np.float)
X4 = data4[:,:-1]
y4 = data4[:,-1:]
data5 = np.loadtxt("d5", dtype = np.float)
X5 = data5[:,:-1]
y5 = data5[:,-1:]
data6 = np.loadtxt("d6", dtype = np.float)
X6 = data6[:,:-1]
y6 = data6[:,-1:]
data7 = np.loadtxt("d7", dtype = np.float)
X7 = data7[:,:-1]
y7 = data7[:,-1:]
data8 = np.loadtxt("d8", dtype = np.float)
X8 = data8[:,:-1]
y8 = data8[:,-1:]
data9 = np.loadtxt("d9", dtype = np.float)
X9 = data9[:,:-1]
y9 = data9[:,-1:]
for train, test in kfold1:
        print("TRAIN:", train, "TEST:", test)
        X1_train, X1_test = X1[train], X1[test]
        y1_train, y1_test = y1[train], y1[test]
        X2_train, X2_test = X2[train], X2[test]
        y2_train, y2_test = y2[train], y2[test]
        X3_train, X3_test = X3[train], X3[test]
        y3_train, y3_test = y3[train], y3[test]
        X4_train, X4_test = X4[train], X4[test]
        y4_train, y4_test = y4[train], y4[test]
        X5_train, X5_test = X5[train], X5[test]
        y5_train, y5_test = y5[train], y5[test]
        X6_train, X6_test = X6[train], X6[test]
        y6_train, y6_test = y6[train], y6[test]
        X7_train, X7_test = X7[train], X7[test]
        y7_train, y7_test = y7[train], y7[test]
        X8_train, X8_test = X8[train], X8[test]
        y8_train, y8_test = y8[train], y8[test]
        X9_train, X9_test = X9[train], X9[test]
        y9_train, y9_test = y9[train], y9[test]
        


       # fitdata = clf.predict(X_test)
       # os.system("rm result")
       # os.system("touch result")
       # f = open('result','a+')
       # f.write("real__value"+str(y_test)+'\n')
       # f.write("predict_val"+str(fitdata)+'\n')
       # f.close()
       # os.system("cat result")

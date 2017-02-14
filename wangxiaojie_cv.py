#!/home/yiqi.she/venv/common/bin/python3
import sys, os
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import KFold

data = np.loadtxt("dataset", dtype = np.float)
X = data[:,:-1]
y = data[:,-1:]
y = np.reshape(y,[81,])
label_kfold = KFold(81, n_folds=9,shuffle=False)
for train_index, test_index in label_kfold:
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = SVC()
        clf.fit(X_train, y_train)
        fitdata = clf.predict(X_test)
        os.system("rm result")
        os.system("touch result")
        f = open('result','a+')
        f.write("real__value"+str(y_test)+'\n')
        f.write("predict_val"+str(fitdata)+'\n')
        f.close()
        os.system("cat result")

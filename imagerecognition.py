# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 22:45:02 2018

@author: SARABJEET
"""

import matplotlib.pyplot as pt

from sklearn import datasets
from sklearn import svm

digits=datasets.load_digits()

clf=svm.SVC(gamma=0.001,C=100)
print(len(digits.data))
x,y=digits.data[:-10],digits.target[:-10]
clf.fit(x,y)
print('prediction :',clf.predict(digits.data[[-9]]))

pt.imshow(digits.images[-9],cmap=pt.cm.gray_r,interpolation="nearest")
pt.show()
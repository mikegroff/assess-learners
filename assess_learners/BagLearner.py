"""
A simple wrapper for linear regression.  (c) 2015 Tucker Balch

Copyright 2018, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 4646/7646

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as github and gitlab.  This copyright statement should not be removed
or edited.

We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future
students of CS 7646 is prohibited and subject to being investigated as a
GT honor code violation.

-----do not edit anything above this line---
"""

import numpy as np
from scipy import stats

class BagLearner(object):

    def __init__(self, learner, kwargs , bags, boost = False, verbose = False):
        self.bags = bags
        self.learner = learner
        for kw in kwargs:
            if kw is "leaf_size" :
                self.leaf_size = 1
            else:
                self.leaf_size = 0

    def author(self):
        return 'mgroff3' # replace tb34 with your Georgia Tech username

    def addEvidence(self,dataX,dataY):
        tot = dataX.shape[0]
        self.model = np.ones((4,tot*2))
        for i in range(0,self.bags):
            n = np.random.randint(tot, size =tot)
            xdata = np.take(dataX, n, axis = 0)
            ydata = np.take(dataY, n, axis = 0)
            if self.leaf_size == 1:
                learner = self.learner(self.leaf_size, verbose = False)
            else:
                learner = self.learner(verbose = False)
            learner.addEvidence(xdata,ydata)
            learner.model = np.pad(learner.model, (0,2*tot-learner.model.shape[0]),'constant', constant_values=(0, 0))
            self.model = np.dstack((self.model,learner.model))
        self.model = self.model[1:,:,:]

    def query(self,points):
        sizey = points.shape[0]
        predi = np.ones(sizey)
        for i in range(0,self.bags):
            model = self.model[i,:,:]
            pred = np.ones(1)
            for i in range(0,sizey):
                j = 0
                while( j != -1):
                    check = model[j,:]
                    k = check[0].astype(int)
                    if k == -1:
                        pred = np.append(pred, check[1])
                        j = -1
                        continue
                    if (points[i,k] <= check[1]):
                        j += check[2].astype(int)
                    else:
                        j += check[3].astype(int)
            predi = np.vstack((predi,pred[1:]))
        predi = predi[1:,:]

        return stats.mode(predi.T)



if __name__=="__main__":
    print "BagLearner'"

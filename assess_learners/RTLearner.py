import numpy as np

class RTLearner(object):

    def __init__(self, leaf_size, verbose = False):
        self.leaf_size = leaf_size

    def author(self):
        return 'mgroff3'

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        #print(dataX.shape,dataY.shape)
        dataset = np.column_stack((dataX, dataY))

        def build(data):
            if data.shape[0] == 0:
                return
            if data.shape[0] <= self.leaf_size:
                return np.array([[-1,np.mean(data[:,-1]),-1,-1]])
            truth = np.all(data[:,-1] == data[0,-1])
            if truth:
                return np.array([[-1,data[0,-1],-1,-1]])
            else:
                r = data.shape[1]-1
                i = np.random.randint(r)
                SplitVal =	np.median(data[:,i])
                lefttree  = build(data[data[:,i] <= SplitVal])
                righttree =	build(data[data[:,i] > SplitVal])
                root = np.array([i,	SplitVal, 1, lefttree.shape[0]+1])
                root = np.vstack((root, lefttree,))
                if righttree is None:
                    root = np.array([[-1,np.mean(data[:,-1]),-1,-1]])
                else:
                    root = np.vstack((root, righttree,))
                return root

        self.model = build(dataset)


    def query(self,points):
        pred = np.ones(1)
        for i in range(0,np.size(points,0)):
            j = 0
            while( j != -1):
                check = self.model[j,:]
                k = check[0].astype(int)
                if k == -1:
                    pred = np.append(pred, check[1])
                    j = -1
                    continue
                if (points[i,k] <= check[1]):
                    j += check[2].astype(int)
                else:
                    j += check[3].astype(int)

        return pred[1:]



if __name__=="__main__":
    print "RTLearner"

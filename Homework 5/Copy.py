# EM.py
# --------------
# Licensing Information:  You are free to use or extend this project for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to The Ohio State University, including a link to http://aritter.github.io/courses/5523_fall18.html
#
# Attribution Information: This assignment was developed at The Ohio State University
# by Alan Ritter (ritter.1492@osu.edu).

import random
import math
import sys
import re
import numpy as np
import matplotlib.pyplot as plt

# GLOBALS/Constants
VAR_INIT = 1


def logExpSum(x):
    # TODO: implement logExpSum
    xmax = max(x)
    xi = x - xmax
    return xmax + np.log(sum(np.exp(xi)))


def readTrue(filename='wine-true.data'):
    f = open(filename)
    labels = []
    splitRe = re.compile(r"\s")
    for line in f:
        labels.append(int(splitRe.split(line)[0]))
    return labels


#########################################################################
# Reads and manages data in appropriate format
#########################################################################


class Data:
    def __init__(self, filename):
        self.data = []
        f = open(filename)
        (self.nRows, self.nCols) = [int(x) for x in f.readline().split(" ")]
        for line in f:
            self.data.append([float(x) for x in line.split(" ")])

    # Computers the range of each column (returns a list of min-max tuples)
    def Range(self):
        ranges = []
        for j in range(self.nCols):
            min = self.data[0][j]
            max = self.data[0][j]
            for i in range(1, self.nRows):
                if self.data[i][j] > max:
                    max = self.data[i][j]
                if self.data[i][j] < min:
                    min = self.data[i][j]
            ranges.append((min, max))
        return ranges

    def __getitem__(self, row):
        return self.data[row]

#########################################################################
# Computes EM on a given data set, using the specified number of clusters
# self.parameters is a tuple containing the mean and variance for each gaussian
#########################################################################


class EM:
    def __init__(self, data, nClusters):
        # Initialize parameters randomly...
        random.seed()
        self.parameters = []
        self.priors = []        # Cluster priors
        self.nClusters = nClusters
        self.data = data
        self.resp = np.zeros((data.nRows, nClusters))
        ranges = data.Range()
        self.likelihood = []
        for i in range(nClusters):
            p = []
            initRow = random.randint(0, data.nRows-1)
            for j in range(data.nCols):
                # Randomly initalize variance in range of data
                p.append((random.uniform(ranges[j][0], ranges[j][1]), VAR_INIT*(ranges[j][1] - ranges[j][0])))
            self.parameters.append(p)

        # Initialize priors uniformly
        for c in range(nClusters):
            self.priors.append(1/float(nClusters))

    def LogLikelihood(self, data):
        logLikelihood = 0.0
        # TODO: compute log-likelihood of the data
        p = list()
        for r in range(data.nRows):
            p.clear()
            for c in range(self.nClusters):
                prob = math.log(self.priors[c]) + self.LogProb(r, c, data)
                p.append(prob)
            logLikelihood = logLikelihood + logExpSum(p)
        return logLikelihood

    # Compute marginal distributions of hidden variables
    def Estep(self):
        # TODO: E-step
        for r in range(self.data.nRows):
            for k in range(self.nClusters):
                self.resp[r][k] = np.log(self.priors[k]) + self.LogProb(r, k, self.data)
            denm = logExpSum(self.resp[r, :])
            self.resp[r, :] = np.exp(self.resp[r, :] - denm)
        pass

    # Update the parameter estimates
    def Mstep(self):
        # TODO: M-step
        for c in range(self.nClusters):
            rik = self.resp[:, c]
            rk = np.sum(rik)
            self.priors[c] = rk / self.data.nRows
            for col in range(self.data.nCols):
                x = [self.data[i][col] for i in range(self.data.nRows)]
                mean = self.parameters[c][col][0]
                newMean = np.sum(rik[i] * x[i] for i in range(self.data.nRows)) / rk
                newVar = np.sum(rik[i] * (x[i] - mean) ** 2 for i in range(self.data.nRows)) / rk
                self.parameters[c][col] = (newMean, newVar)
        pass

    # Computes the probability that row was generated by cluster
    def LogProb(self, row, cluster, data):
        # TODO: compute probability row i was generated by cluster k
        x = np.array(data[row])
        params = np.array(self.parameters[cluster])
        mean = np.array(params[:, 0])
        var = np.array(params[:, 1])
        K = -1 * (data.nCols * 0.5) * math.log(2 * math.pi)
        # prob = -0.5 * sum(math.log(var[i]) + (x[i] - mean[i]) ** 2 / var[i] for i in range(data.nCols)) + K
        var = var + 0.0001  # To avoid zero
        cl = np.log(var)
        div = np.divide(((x - mean) ** 2), var)
        prob = -0.5 * np.sum(cl + div) + K  # for i in range(data.nCols)) + K
        return prob

    def Run(self, maxsteps=100, testData=None):
        # TODO: Implement EM algorithm
        trainLikelihood = 0.0
        testLikelihood = 0.0
        oldVal = -1e-6
        for i in range(maxsteps):
            self.Estep()
            self.Mstep()
            oldVal = trainLikelihood
            trainLikelihood = self.LogLikelihood(self.data)
            print('#Training Details#  Iteration: ', i, 'LogLikelihood: ', trainLikelihood)
            if testData is not None:
                testLikelihood = self.LogLikelihood(testData)
                print('#Test Details#  Iteration:', i, 'LogLikelihood: ', testLikelihood)
            self.likelihood.append((trainLikelihood, testLikelihood))
            if math.fabs(trainLikelihood - oldVal) < 0.001:
                break
        return trainLikelihood, testLikelihood


if __name__ == "__main__":
    d = Data('wine.train')
    td = Data('wine.test')
    if len(sys.argv) > 1:
        e = EM(d, int(sys.argv[1]))
    else:
        e = EM(d, 3)
    e.Run(100, td)
    # Plot Training Likelihood vs # of Iteration
    fig = plt.figure()
    plt.title('Iterations vs log likelihood for training set')
    plt.xlabel('No. of Iterations')
    plt.ylabel('Log Likelihood')
    plt.xticks(range(len(e.likelihood)))
    plt.plot(range(len(e.likelihood)), [train for train, test in e.likelihood], 'bo', ms=3, ls='-', lw=1, label='Training set')
    plt.legend()
    plt.savefig('Train_Plot.png')
    plt.close(fig)
    fig2 = plt.figure()
    plt.title('Iterations vs log likelihood for test set')
    plt.xlabel('No. of Iterations')
    plt.ylabel('Log Likelihood')
    plt.xticks(range(len(e.likelihood)))
    plt.plot(range(len(e.likelihood)), [test for train, test in e.likelihood], 'gs', ms=3, ls='--', lw=1, label='Test set')
    plt.legend()
    plt.savefig('Test_Plot.png')
    plt.close(fig2)

    likelihood_by_cluster = list()
    for numClusters in range(1, 11):
        e = EM(d, numClusters)
        (train, test) = e.Run(100, td)
        likelihood_by_cluster.append((train, test))
    fig3 = plt.figure()
    plt.title('Clusters vs Loglikelihood for Training and Test Dataset')
    plt.xlabel('No. of Clusters')
    plt.ylabel('Loglikelihood')
    plt.xticks(range(1, 11))
    plt.plot(range(1, 11), [train for train, test in likelihood_by_cluster], 'bo', ms=3, ls='-', lw=1,
             label='Training set')
    plt.plot(range(1, 11), [test for train, test in likelihood_by_cluster], 'gs', ms=3, ls='--', lw=1,
             label='Test set')
    plt.legend()
    plt.savefig('Cluster_Plot.png')
    plt.close(fig3)

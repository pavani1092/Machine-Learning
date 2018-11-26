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

#GLOBALS/Constants
VAR_INIT = 1


def logExpSum(x):
    xmax = np.max(x)
    val = x - xmax
    val = np.sum(np.exp(val))
    return xmax + np.log(val)


def readTrue(filename='wine-true.data'):
    f = open(filename)
    labels = []
    splitRe = re.compile(r"\s")
    for line in f:
        labels.append(int(splitRe.split(line)[0]))
    return labels

#########################################################################
#Reads and manages data in appropriate format
#########################################################################
class Data:
    def __init__(self, filename):
        self.data = []
        f = open(filename)
        (self.nRows, self.nCols) = [int(x) for x in f.readline().split(" ")]
        for line in f:
            self.data.append([float(x) for x in line.split(" ")])

    #Computers the range of each column (returns a list of min-max tuples)
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
#Computes EM on a given data set, using the specified number of clusters
#self.parameters is a tuple containing the mean and variance for each gaussian
#########################################################################
class EM:
    def __init__(self, data, nClusters):
        # Initialize parameters randomly...
        random.seed()
        self.parameters = []
        self.priors = []        # Cluster priors
        self.nClusters = nClusters
        self.data = data
        ranges = data.Range()
        self.resp = np.zeros((nClusters, self.data.nRows))
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
        loglikelihood = 0.0
        for i in range(0, data.nRows):
            expsum = list()
            for cluster in range(self.nClusters):
                expsum.append(self.priors[cluster] * self.LogProb(i, cluster, data))
            loglikelihood += logExpSum(expsum)
        return loglikelihood

    # Compute marginal distributions of hidden variables
    def Estep(self):
        for i in range(self.nClusters):
            for j in range(self.data.nRows):
                self.resp[i][j] = self.LogProb(j, i, self.data)

        for j in range(self.data.nRows):
            denm = logExpSum(self.resp[:, j])
            for i in range(self.nClusters):
                self.resp[i][j] = math.exp(self.resp[i][j] - denm)

    # Update the parameter estimates
    def Mstep(self):
        # TODO: M-step
        for i in range(self.nClusters):
            total = np.zeros(self.data.nCols)
            total2 = np.zeros(self.data.nCols)
            denom = 0.0
            p = list()
            for j in range(self.data.nRows):
                denom += self.resp[i][j]
                total = np.add(total, np.multiply(self.data[j], self.resp[i][j]))

            if denom == 0:
                """"
                Getting a divide by 0 error on denom
                """
                denom = 0.000001
            total = total / (1.0 * denom)

            for k in range(self.data.nCols):
                p.append((total[k], self.parameters[i][k][1]))
            self.parameters[i] = p

            p = list()
            means = [list(x) for x in zip(*self.parameters[i])][0]
            for j in range(self.data.nRows):
                xArr = np.subtract(self.data[j], means)
                total2 = np.add(total2, (np.multiply(xArr, self.resp[i][j]) * xArr))
            total2 = total2 / (1.0 * denom)

            for k in range(self.data.nCols):
                p.append((self.parameters[i][k][0], total2[k]))
            self.parameters[i] = p
            self.priors[i] = denom / (1.0 * self.data.nRows)

    # Computes the probability that row was generated by cluster
    def LogProb(self, row, cluster, data):
        # TODO: compute probability row i was generated by cluster k
        epsilon = 1e-06
        nCols = data.nCols
        data = data[row]
        prob = -1 * (nCols * 0.5) * math.log(2 * math.pi)

        for i in range(nCols):
            mean = self.parameters[cluster][i][0]
            var = self.parameters[cluster][i][1]

            if var == 0:
                var = epsilon

            prob += (-0.5 * np.log(var)) + (-0.5 * ((data[i] - mean) ** 2) / (var * 1.0))

        return prob + np.log(self.priors[cluster])

    def Run(self, maxsteps=100, testData=None):
        # TODO: Implement EM algorithm
        trainLikelihood = 0.0
        testLikelihood = 0.0
        printstep = 1
        prevtrainLikelihood = trainLikelihood
        for step in range(maxsteps):
            self.Estep()
            self.Mstep()
            trainLikelihood = self.LogLikelihood(self.data)
            p_change = ((trainLikelihood - prevtrainLikelihood)/trainLikelihood) * 100
            if step % printstep == 0:
                print("Iteration: ", step, " LogLikelihood: ", trainLikelihood, " Change", p_change, ' %')
            prevtrainLikelihood = trainLikelihood

        return (trainLikelihood, testLikelihood)


if __name__ == "__main__":
    d = Data('wine.train')
    if len(sys.argv) > 1:
        e = EM(d, int(sys.argv[1]))
    else:
        e = EM(d, 3)
    e.Run(100)

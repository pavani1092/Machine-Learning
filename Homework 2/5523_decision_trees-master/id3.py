# id3.py
# --------------
# Licensing Information:  You are free to use or extend this project for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to The Ohio State University, including a link to http://aritter.github.io/courses/5523_fall18.html
#
# Attribution Information: This assignment was developed at The Ohio State University
# by Alan Ritter (ritter.1492@osu.edu).
# Modified by Soham Mukherjee(mukherjee.126) for CSE 5523: HW#2
from collections import namedtuple
import sys
from Data import *
import numpy as np


DtNode = namedtuple("DtNode", "fVal, nPosNeg, gain, left, right")

POS_CLASS = 'e'

def InformationGain(data, f):
    data1=[]
    data2=[]
    for d in data:
     if d[f.feature]==f.value:
        data1.append(d)
     else:
        data2.append(d)
      
    t0 = H(data)
    t1 = -((len(data1)+1)/float((len(data)+2)))*H(data1) 
    t2 = -((len(data2)+1)/float((len(data)+2)))*H(data2)
    return (t0+t1+t2)

def H(data):
    if(len(data)==0):
        return 0
    (fpos, fneg) = CountVal(data)
    pX1 = fpos/float(len(data))
    pX0 = fneg/float(len(data))
    l = 0
    r = 0
    if(pX1>0):
        l = -pX1*np.log2(pX1)
    if(pX0>0):
        r = -pX0*np.log2(pX0)
    return (l+r)

def CountVal(data):
    (pos,neg)=(0,0)
    for d in data:
        if d[0]==POS_CLASS:
            pos+=1
        else:
            neg+=1
    return (pos,neg)
def Classify(tree, instance):
    if tree.left == None and tree.right == None:
        return tree.nPosNeg[0] > tree.nPosNeg[1]
    elif instance[tree.fVal.feature] == tree.fVal.value:
        return Classify(tree.left, instance)
    else:
        return Classify(tree.right, instance)

def Accuracy(tree, data):
    nCorrect = 0
    for d in data:
        if Classify(tree, d) == (d[0] == POS_CLASS):
            nCorrect += 1
    return float(nCorrect) / len(data)

def PrintTree(node, prefix=''):
    print("%s>%s\t%s\t%s" % (prefix, node.fVal, node.nPosNeg, node.gain))
    if node.left != None:
        PrintTree(node.left, prefix + '-')
    if node.right != None:
        PrintTree(node.right, prefix + '-')        
        
def ID3(data, features, MIN_GAIN):
    (npos,nneg) = CountVal(data)
    if len(features) == 0:
      return DtNode(None,(npos,nneg),0,None,None)
    
    max = -20000
    for f in features:
      i = InformationGain(data,f)
      # print (i)
      if i > max:
        max = i
        fsplit = f         
        #print("Feature chosen %r" % (fsplit,))
        #print("Gain %s " % max)
        
    if max <= MIN_GAIN:
      return DtNode(list(features)[0],(npos,nneg),max,None,None)
        
        
        
    data1 = []
    data2 = []
    (nfpos,nfneg) = CountVal(data)
    for d in data:
      if d[fsplit.feature] == fsplit.value:
       data1.append(d)
      else:
       data2.append(d)


    features.discard(fsplit)          
    lnode = ID3(data1,features,MIN_GAIN)
    rnode = ID3(data2,features,MIN_GAIN)
    return DtNode(fsplit, (nfpos,nfneg), max, lnode,rnode)

if __name__ == "__main__":
    train = MushroomData(sys.argv[1])
    dev = MushroomData(sys.argv[2])
    dTree = ID3(train.data, train.features, MIN_GAIN=float(sys.argv[3]))
    PrintTree(dTree)
    print("Accuracy on test data: %s"%(Accuracy(dTree, dev.data)))

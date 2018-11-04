# Data.py
# --------------
# Licensing Information:  You are free to use or extend this project for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to The Ohio State University, including a link to http://aritter.github.io/courses/5523_fall18.html
#
# Attribution Information: This assignment was developed at The Ohio State University
# by Alan Ritter (ritter.1492@osu.edu).

from collections import namedtuple
import sys


FeatureVal = namedtuple("FeatureVal", "feature, value")        

class MushroomData:
    def __init__(self, fileName):
        self.data = []
        self.features = set()
        for line in open(fileName):
            line = line.strip()            
            attributes = line.split(',')
            for i in range(1,len(attributes)):
                self.features.add(FeatureVal(i, attributes[i]))
            self.data.append(attributes)

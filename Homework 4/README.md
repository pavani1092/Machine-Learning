# CSE 5523 Homework 4: Linear Models on Brain Image Data
## How to run the program:
Use **python 3**. In the terminal type in:\
**> python linear_brain.py**.\
By default it runs SGDLogistic on test data with best parameters.
To run SGDHinge on test data comment `print("Accuracy (Logistic Loss):\t%s" % crossValidation(X, Y, SgdLogistic, maxIter=100, lmda=0.3, learningRate=0.001, sample=range(20,X.shape[0])))`
this line and uncomment `#print("Accuracy (Hinge Loss):\t%s" % crossValidation(X, Y, SgdHinge, maxIter=100, lmda=0.1, learningRate=0.0001, sample=range(20, X.shape[0])))`
To run the program on training data with parameter tuning comment above two lines and uncomment the following lines:
```python
# for learnrate in eta:
    #     for lmda in lbda:
    #         print("eta =  ", learnrate, " lambda =  ", lmda)
    # # Cross validation
    # # Development
    #         #print("Accuracy (Logistic Loss):\t%s" % crossValidation(X, Y, SgdLogistic, maxIter=100, lmda=lmda, learningRate=learnrate, sample=range(20)))
    #         print("Accuracy (Hinge Loss):\t%s" % crossValidation(X, Y, SgdHinge, maxIter=100, lmda=lmda, learningRate= learnrate, sample=range(20)))
```


## Output files:
Output files can be found in two versions. One in .pdf and one in .txt.\
To open .txt file use notepad ++ . Using notepad of Windows omits the newline and output looks messy.

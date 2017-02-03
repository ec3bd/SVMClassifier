import numpy as np
import sklearn.svm as svm
import sklearn.preprocessing as pre
import matplotlib as plt
import sys
import random
from scipy import sparse
import sklearn.model_selection as ms


vectorOffset = {"Federal-gov":0, "Private":1, "Self-emp-not-inc":2, "Self-emp-inc":3, "State-gov":4, "Local-gov":5, "Without-pay":6, "Never-worked":7,
               "Preschool":0, "1st-4th":1, "5th-6th":2, "7th-8th":3, "9th":4, "10th":5, "11th":6, "12th":7, "HS-grad":8, "Some-college":9, "Prof-school":10, "Assoc-voc":11, "Assoc-acdm":12, "Bachelors":13, "Masters":14, "Doctorate":15,
               "Married-AF-spouse":0, "Married-civ-spouse":1, "Never-married":2, "Widowed":3, "Divorced":4, "Married-spouse-absent":5, "Separated":6,
               "Armed-Forces":0, "Machine-op-inspct":1, "Exec-managerial":2, "Craft-repair":3, "Prof-specialty":4, "Transport-moving":5, "Adm-clerical":6, "Other-service":7, "Sales":8, "Tech-support":9, "Farming-fishing":10, "Protective-serv":11, "Handlers-cleaners":12, "Priv-house-serv":13,
               "Other-relative":0, "Husband":1, "Not-in-family":2, "Unmarried":3, "Own-child":4, "Wife":5,
               "Asian-Pac-Islander":0, "Black":1, "White":2, "Other":3, "Amer-Indian-Eskimo":4,
               "Male":0, "Female":1,
                "Ecuador":0, "United-States":1, "Dominican-Republic":2, "Puerto-Rico":3, "Germany":4, "Columbia":5, "Cuba":6, "Japan":7, "El-Salvador":8, "South":9, "Poland":10, "China":11, "Philippines":12, "Iran":13, "Mexico":14, "England":15, "Haiti":16, "Italy":17, "Jamaica":18, "Vietnam":19, "Scotland":20, "Canada":21, "Guatemala":22, "India":23, "Ireland":24, "Cambodia":25, "Thailand":26, "Yugoslavia":27, "Hong":28, "Nicaragua":29, "Taiwan":30, "Portugal":31, "Hungary":32, "France":33, "Greece":34, "Laos":35, "Outlying-US(Guam-USVI-etc)":36, "Peru":37, "Trinadad&Tobago":38, "Honduras":39, "Holand-Netherlands":40}
vectorLength = {0:1, 1:8, 2:1, 3:16, 4:1, 5:7, 6:14, 7:6, 8:5, 9:2, 10:1, 11:1, 12:1, 13:41}



def main():
    clf = SvmIncomeClassifier()
    xMat,yMat = clf.load_data("salary.labeled.csv")
    print("Done loading and preprocessing data")

    #clf.paramSearch(xMat, yMat)  #failed attempt to search the space automatically, exited with a memory error

    modelWithParam = clf.train_and_select_model(xMat, yMat)   #what I use to do cv and pick the best parameters out of those in param_set, returns a model with those params encoded
    modelWithParam = svm.SVC(kernel = 'rbf', C=2, gamma = .01)  #for generating prediction files a bit quicker towards the end
    xMat = sparse.csr_matrix(xMat)
    yMat = np.squeeze(np.asarray(yMat))
    xPredict, other = clf.load_data("salary.2Predict.csv")
    clf.predict(xMat, yMat, modelWithParam, xPredict)  #fits and predicts, then outputs results in predictions.txt

class SvmIncomeClassifier:

    param_set = [
                     #{'kernel': 'rbf', 'C': 1, 'degree': 1},
                    # {'kernel': 'rbf', 'C': 1, 'degree': 3},
                     #{'kernel': 'rbf', 'C': 1, 'degree': 5},
                     #{'kernel': 'rbf', 'C': 1, 'degree': 7},
        # {'kernel': 'rbf', 'C': 1000, 'degree': 1},
        # {'kernel': 'poly', 'C': 500, 'degree': 1},
        # {'kernel': 'poly', 'C': 500, 'degree': 3},
        # {'kernel': 'poly', 'C': 500, 'degree': 5},
        # {'kernel': 'poly', 'C': 500, 'degree': 7},
        # {'kernel': 'rbf', 'C': 500, 'degree': 3, 'gamma': 1},
        # {'kernel': 'rbf', 'C': 500, 'degree': 3, 'gamma': .5},
        # {'kernel': 'rbf', 'C': 500, 'degree': 3, 'gamma': .01},
        # {'kernel': 'rbf', 'C': 500, 'degree': 3, 'gamma': .05},
        # {'kernel': 'rbf', 'C': 500, 'degree': 3, 'gamma': .005},
        # {'kernel': 'rbf', 'C': 500, 'degree': 3, 'gamma': .0005},
        # {'kernel': 'rbf', 'C': 500, 'degree': 3, 'gamma': .00005},
        # {'kernel': 'rbf', 'C': 500, 'degree': 3, 'gamma': .000005},
        #{'kernel': 'rbf', 'C': 200, 'degree': 3, 'gamma': 65},
        #{'kernel': 'rbf', 'C': 200, 'degree': 3, 'gamma': 60},
        # {'kernel': 'rbf', 'C': 200, 'degree': 3, 'gamma': 55},
        {'kernel': 'rbf', 'C': 200, 'degree': 3, 'gamma': 50},
        {'kernel': 'rbf', 'C': 200, 'degree': 3, 'gamma': 35},
        {'kernel': 'rbf', 'C': 150, 'degree': 3, 'gamma': 50},
        {'kernel': 'rbf', 'C': 250, 'degree': 3, 'gamma': 50},
        {'kernel': 'rbf', 'C': 200, 'degree': 3, 'gamma': 45},
        {'kernel': 'rbf', 'C': 100, 'degree': 3, 'gamma': .01},
        {'gamma': 35, 'kernel': 'rbf', 'degree': 3, 'C': 200},
        #{'kernel': 'rbf', 'C': 100, 'degree': 3, 'gamma': 500},
        # {'kernel': 'rbf', 'C': 50, 'degree': 3, 'gamma': 25},
        # {'kernel': 'rbf', 'C': 100, 'degree': 3, 'gamma': 1},
        # {'kernel': 'rbf', 'C': 250, 'degree': 3, 'gamma': 1},
        # {'kernel': 'rbf', 'C': 20, 'degree': 3, 'gamma': 1},
        # {'kernel': 'rbf', 'C': 1, 'degree': 3, 'gamma': 5},
        # {'kernel': 'rbf', 'C': 1, 'degree': 3, 'gamma': 25},
        # {'kernel': 'rbf', 'C': 1000, 'degree': 3, 'gamma': .005},
        # {'kernel': 'linear', 'C': 500, 'degree': 3, 'gamma': 1},
        # {'kernel': 'rbf', 'C': 100, 'degree': 3, 'gamma': .5},
        # {'kernel': 'sigmoid', 'C': 100, 'degree': 3, 'gamma': .5},
        #{'kernel': 'rbf', 'C': 10000, 'degree': 1},
        #{'kernel': 'rbf', 'C': 20000, 'degree': 1},

        ]


    def __init__(self):
        random.seed(0)
        self.minValues, self.maxValues = self.initMaxMin()

    def train_and_select_model(self, xVal, yVal):
        n = xVal.shape[0]
        p = xVal.shape[1]
        augmented = np.concatenate((xVal,yVal),1)
        random.seed(37)
        random.shuffle(augmented)
        augmented = np.asmatrix(augmented)
        xVal = augmented[:, 0:p]
        yVal = augmented[:, p]
        bestParam = 0;
        bestScore = 0;
        bestModel = None
        for param in self.param_set:
            foldScore = []
            svc = svm.SVC(C=param['C'], kernel = param['kernel'], degree = param['degree'], gamma = param['gamma'])
            for i in range(0,3):
                beg = int(i*n/3)
                end = int((i+1)*n/3)
                xTest = xVal[beg:end,:]
                xTrain = xVal[0:beg:1,:]
                xTrain2 = xVal[end:,:]
                xTrain = np.concatenate((xTrain, xTrain2))
                yTest = yVal[beg:end,:]
                yTrain = yVal[0:beg:1,:]
                yTrain2 = yVal[end:,:]
                yTrain = np.concatenate((yTrain, yTrain2))
                xTrain = sparse.csr_matrix(xTrain)
                yTrain = np.squeeze(np.asarray(yTrain))
                svc.fit(xTrain, yTrain)
                del xTrain
                del yTrain # was having memory issues, this may or may not be necessary
                foldScore.append(svc.score(xTest, yTest))
            meanScore = sum(foldScore) / len(foldScore)
            print(repr(param) + ":  " + repr(meanScore))
            if meanScore > bestScore:
                bestParam = param
                bestScore = meanScore
                bestModel = svc
        print(bestParam)
        print(bestScore)

        return svc

    def predict(self, xMat, yMat, model, xPredict):
        model.fit(xMat, yMat)
        hypothesis = model.predict(xPredict)
        print(hypothesis)
        file = open("predictions.txt", 'w')
        for el in hypothesis:
            if el > 0:
                file.write(">50K\n")
            else:
                file.write("<=50K\n")



    #used to find each unique string category so as to map everything correctly
    def unique(self):
        file = open ("salary.2Predict.csv", 'r')
        list = []
        for line in file:
            if(line.rstrip() == ""): continue
            line = line.rstrip()
            linedata = line.split(", ")
            newLine = []
            if linedata[8].rstrip() not in list:
                list.append(linedata[8].rstrip())
        # for i in range (0,len(xMat[0:,column])):
        #     if xMat[i,column] not in list:
        #         list.append(xMat[i,column])
        print(list)

    def load_data(self, filename):
        file = open(filename, 'r')
        xVals = []
        yVals = []
        for line in file:
            if(line.rstrip() == ""): continue
            line = line.rstrip()
            linedata = line.split(", ")
            newLine = []
            for i in range(0,14):
                try:
                    newLine.append(int(linedata[i]))
                except ValueError:
                    if linedata[i] == "?":
                        for j in range(0, vectorLength[i]): newLine.append(0)
                    else:
                        for j in range(0, vectorLength[i]):
                            if j == vectorOffset[linedata[i]]:
                                newLine.append(1)
                            else:
                                newLine.append(0)
            if(len(linedata) == 15):
                if(linedata[14] == "<=50K"): yVals.append(-1)
                else: yVals.append(1)
            xVals.append(tuple(newLine))
        ts = pre.MinMaxScaler()
        ts.data_min_ = self.minValues
        ts.data_max_ = self.maxValues
        xMat = np.asmatrix(xVals)
        xMat = ts.fit_transform(xMat)
        if len(linedata) == 15:
            yMat = np.asmatrix(yVals)
            return xMat,yMat.T
        else:
            return xMat, None

    def paramSearch(self, xMat, yMat):
        C_range = np.logspace(-3, 3, 6)
        gamma_range = np.logspace(-3, 2, 6)
        param_grid = dict(gamma=gamma_range, C=C_range)
        cv = ms.StratifiedShuffleSplit(n_splits=3, test_size=0.33, random_state=42)
        grid = ms.GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
        grid.fit(xMat, yMat)
        print("The best parameters are %s with a score of %0.2f"% (grid.best_params_, grid.best_score_))

    def initMaxMin(self):
        minValues = np.zeros((105,))
        maxValues =np.ones((105,))
        mins = [17, 0, 12285, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        maxes = [90, 1, 1490400, 1, 16, 1, 1, 1, 1, 1, 99999, 4356, 99, 1]
        col = 0
        count = 0
        while col < 105 and count < 14:
            minValues[col] = mins[count]
            maxValues[col] = maxes[count]
            col+=vectorLength[count]
            count+=1
        return minValues, maxValues

if __name__ == "__main__":
    main()
from math import sqrt

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


class Evaluation:

    def __init__(self, orig: list, pred: list):
        self.orig = orig
        self.pred = pred
        self.labels = list(set(self.orig).union(set(self.pred)))
        self.acc, self.c_mat = self.accuracy()
        self.acc, self.c_mat = self.accuracy()
        self.mape = self.mape()
        self.rmse = self.rmse()
        self.mae = self.mae()

    def accuracy(self):
        c_mat = pd.DataFrame(columns=self.labels, index=self.labels)
        c_mat.fillna(0, inplace=True)
        hits = 0.0
        total = 0.0
        for i in range(len(self.orig)):
            total = total + 1
            if self.orig[i] == self.pred[i]:
                hits = hits + 1
            c_mat.loc[self.orig[i]][self.pred[i]] = c_mat.loc[self.orig[i]][self.pred[i]] + 1
        acc = (hits / total) * 100
        return acc, c_mat

    def mape(self):
        percs = []
        for i in range(len(self.orig)):
            per = (abs((float(self.orig[i]) - float(self.pred[i])) / float(self.orig[i]))) * 100
            percs.append(per)
        return sum(percs) / len(percs)

    def rmse(self):

        return sqrt(mean_squared_error(self.orig, self.pred))

    def mae(self):
        return mean_absolute_error(self.orig, self.pred)


if __name__ == "__main__":
    orig = [1, 2, 3, 4, 5]
    pred = [1, 1, 2, 3, 4]
    eval = Evaluation(orig, pred)
    print(eval.mape)
    print(eval.mae)
    print(eval.rmse)
    print(eval.acc)
    print(eval.c_mat)

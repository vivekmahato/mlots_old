import random

from scipy.stats import mode


class KNN:

    def __init__(self, dic: dict, rows: list, cols: list, dist_mat: None, n_neighbors=5, rnd=False):
        self.dic = dic
        self.rows = rows
        self.cols = cols
        self.dist_mat = dist_mat
        self.n_neighbors = n_neighbors
        self.rnd = rnd
        self.orig, self.pred = self.predict()

    def predict(self):
        original = []
        predicted = []
        targets = []
        for row in self.rows:
            dm = self.dist_mat.loc[int(row)].sort_values(ascending=True).index
            if self.rnd:
                neighbors = random.sample(self.cols, 5)
            else:
                neighbors = [int(i) for i in list(dm) if i not in self.rows]
                neighbors = neighbors[:self.n_neighbors]
            for n in neighbors:
                targets.append(self.dic[n].label)
            pred = mode(targets)[0][0]
            predicted.append(pred)
            original.append(self.dic[row].label)

        return original, predicted

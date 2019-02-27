import random

from TimeSeries import TimeSeries
from scipy.stats import mode

from mlots import DistMat


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


if __name__ == "__main__":
    test_dic = {}
    for i in range(10):
        data = [random.randrange(1, 101, 1) for _ in range(10)]
        # data = [["abc", "bcd", "fgh"], ["aac", "bdd", "ffh"], ["acc", "ccd", "ggh"]]
        dic = {"data": data, "label": random.randint(1, 2)}
        ts = TimeSeries(dic)
        test_dic[i] = ts
    keys = list(test_dic.keys())
    d_mat = DistMat(dic=test_dic, train_keys=None, test_keys=None, pool_size=2, metric="EditDistance").dist_mat
    knn = KNN(dic=test_dic, rows=keys[:4], cols=keys[4:], dist_mat=d_mat)
    print(d_mat)
    print(knn.orig)
    print(knn.pred)

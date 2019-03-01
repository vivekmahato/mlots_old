import unittest

from mlots.distmat import DistMat
from mlots.knn import KNN
from mlots.timeseries import TimeSeries


def prep_data():
    data = [[87, 74, 93, 32, 28, 85, 20, 84, 90, 84],
            [19, 48, 11, 57, 21, 95, 45, 21, 50, 35],
            [23, 58, 72, 3, 19, 80, 79, 73, 69, 53],
            [97, 82, 63, 71, 32, 26, 100, 25, 64, 93],
            [43, 8, 49, 86, 13, 76, 58, 91, 69, 34],
            [42, 89, 92, 56, 54, 84, 56, 14, 11, 3],
            [3, 65, 9, 68, 53, 71, 14, 67, 8, 25],
            [66, 73, 32, 75, 18, 99, 40, 97, 28, 29],
            [27, 92, 2, 7, 81, 91, 67, 97, 63, 39],
            [79, 41, 42, 56, 81, 18, 87, 11, 55, 41]]

    labels = [1, 2, 2, 1, 1, 1, 2, 2, 2, 1]

    test_dic = {}
    for i in range(10):
        d = data[i]
        l = labels[i]
        # data = [["abc", "bcd", "fgh"], ["aac", "bdd", "ffh"], ["acc", "ccd", "ggh"]]
        ts = TimeSeries(d, l)
        test_dic[i] = ts
    return test_dic


class TestMetrics(unittest.TestCase):

    def test_knnEuc(self):
        data = prep_data()
        keys = list(data.keys())
        d_mat = DistMat(dic=data, train_keys=None, test_keys=None, pool_size=2, metric="Euclidean").dist_mat
        knn = KNN(dic=data, rows=keys[:4], cols=keys[4:], dist_mat=d_mat)
        self.assertEqual(knn.pred, [1, 1, 2, 1], "kNN Euclidean failed!")

    def test_knnDTW(self):
        data = prep_data()
        keys = list(data.keys())
        d_mat = DistMat(dic=data, train_keys=None, test_keys=None, pool_size=2, metric="DTW").dist_mat
        knn = KNN(dic=data, rows=keys[:4], cols=keys[4:], dist_mat=d_mat)
        self.assertEqual(knn.pred, [2, 2, 2, 2], "kNN DTW Failed!")

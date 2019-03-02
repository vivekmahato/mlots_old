import unittest

from mlots.distmat import DistMat
from mlots.knn import KNN
from mlots.transform_data import TransformData
from mlots.transformation import Transformation


class TestMetrics(unittest.TestCase):

    def test_knnEuc(self):
        data = TransformData(test=True).transformed_data
        d_mat = DistMat(dic=data, train_keys=None, test_keys=None, pool_size=2, metric="Euclidean").dist_mat
        knn = KNN(dic=data, rows=data["Keys"][:4], cols=data["Keys"][4:], dist_mat=d_mat)
        self.assertEqual(knn.pred, [1, 1, 2, 1], "kNN Euclidean failed!")

    def test_knnDTW(self):
        data = TransformData(test=True).transformed_data
        d_mat = DistMat(dic=data, train_keys=None, test_keys=None, pool_size=2, metric="DTW").dist_mat
        knn = KNN(dic=data, rows=data["Keys"][:4], cols=data["Keys"][4:], dist_mat=d_mat)
        self.assertEqual(knn.pred, [2, 2, 2, 2], "kNN DTW Failed!")

    def test_knnSAX(self):
        data = TransformData(test=True).transformed_data
        data = Transformation(data, method="sax", win_size=5, paa_size=6, alphabet_size=5,
                              nr_strategy=True, z_threshold=True).transformed
        d_mat = DistMat(dic=data, train_keys=None, test_keys=None, pool_size=2, metric="EditDistance").dist_mat
        knn = KNN(dic=data, rows=data["Keys"][:4], cols=data["Keys"][4:], dist_mat=d_mat)
        self.assertEqual(knn.pred, [2, 2, 2, 2], "kNN SAX Failed!")
        self.assertEqual(data[0].data, ['eddcaa', 'decaad', 'ecbcda', 'bbddbe', 'adbbde'], "kNN SAX Failed!")

    def test_knnSFA(self):
        data = TransformData(test=True).transformed_data
        data = Transformation(data, histogram_type="EQUI_DEPTH", method="sfa", win_size=5, word_length=6,
                              alphabet_size=5,
                              norm_mean=True, lower_bound=True).transformed
        d_mat = DistMat(dic=data, train_keys=None, test_keys=None, pool_size=2, metric="EditDistance").dist_mat
        print(data[0].data)
        knn = KNN(dic=data, rows=data["Keys"][:4], cols=data["Keys"][4:], dist_mat=d_mat)
        print(knn.pred)
        self.assertEqual(knn.pred, [2, 2, 2, 2], "kNN SFA Failed!")
        self.assertEqual(data[0].data, ['decbbb', 'ebbabb', 'cacbbb', 'bababb', 'baabbb', 'aacabb'], "kNN SFA Failed!")

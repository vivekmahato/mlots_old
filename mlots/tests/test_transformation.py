import unittest

from mlots.transform_data import TransformData
from mlots.transformation import Transformation

data = {0: [87, 74, 93, 32, 28, 85, 20, 84, 90, 84],
        1: [19, 48, 11, 57, 21, 95, 45, 21, 50, 35],
        2: [23, 58, 72, 3, 19, 80, 79, 73, 69, 53],
        3: [97, 82, 63, 71, 32, 26, 100, 25, 64, 93],
        4: [43, 8, 49, 86, 13, 76, 58, 91, 69, 34],
        5: [42, 89, 92, 56, 54, 84, 56, 14, 11, 3],
        6: [3, 65, 9, 68, 53, 71, 14, 67, 8, 25],
        7: [66, 73, 32, 75, 18, 99, 40, 97, 28, 29],
        8: [27, 92, 2, 7, 81, 91, 67, 97, 63, 39],
        9: [79, 41, 42, 56, 81, 18, 87, 11, 55, 41]}
labels = [1, 2, 2, 1, 1, 1, 2, 2, 2, 1]


class TestMetrics(unittest.TestCase):

    def test_sax(self):
        dic = TransformData(data, labels).transformed_data
        self.transformed_data = Transformation(dic, method="sax", win_size=5, paa_size=6, alphabet_size=5,
                                               nr_strategy=True, z_threshold=True).transformed
        self.assertEqual(self.transformed_data[0], ['eddcaa', 'decaad', 'ecbcda', 'bbddbe', 'adbbde'], "SAX Failed!")

    def test_sfa(self):
        dic = TransformData(data, labels).transformed_data
        self.transformed_data = Transformation(dic, histogram_type="EQUI_DEPTH", method="sfa", win_size=5,
                                               word_length=6, alphabet_size=5,
                                               norm_mean=True, lower_bound=True).transformed
        self.assertEqual(self.transformed_data[0], ['decbbb', 'ebbabb', 'cacbbb', 'bababb', 'baabbb', 'aacabb'],
                         "SFA Failed!")

import unittest

from mlots.metrics import Metrics


class TestMetrics(unittest.TestCase):

    def test_euc(self):
        l1 = [1, 2, 3, 4, 5, 6]
        l2 = [2, 3, 4, 5, 6, 7]
        cost = Metrics(l1, l2, "Euclidean").cost
        self.assertEqual(cost, 2.449489742783178, "Euclidean failed!")

    def test_edit(self):
        cost = Metrics("cat", "bat", "EditDistance").cost
        self.assertEqual(cost, 1, "ED failed!")

    def test_wf(self):
        cost = Metrics("coat", "goat", "WagnerFischer").cost
        self.assertEqual(cost, 4, "WF failed!")

    def test_dtw(self):
        l1 = [1, 2, 3, 4, 5, 6]
        l2 = [2, 3, 4, 5, 6, 7]
        cost = Metrics(l1, l2, "DTW").cost
        self.assertEqual(cost, 2, "DTW failed!")

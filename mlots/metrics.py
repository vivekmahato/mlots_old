import editdistance as ed
from scipy.spatial.distance import cosine as cosine
from scipy.spatial.distance import euclidean as euclidean
from scipy.spatial.distance import jaccard as jaccard

from mlots import dtw
from mlots import wagnerfischer


class Metrics:
    def __init__(self, u: list, v: list, metric="EditDistance"):
        self.u = u
        self.v = v
        self.metric = self.switcher(metric)
        try:
            self.cost = self.metric(u, v).cost
        except:
            self.cost = self.metric(u, v)

    @staticmethod
    def switcher(metric):
        switch = {"WagnerFischer": wagnerfischer, "EditDistance": ed.eval, "DTW": dtw, "Euclidean": euclidean,
                  "Cosine": cosine, "Jaccard": jaccard}
        return switch[metric]

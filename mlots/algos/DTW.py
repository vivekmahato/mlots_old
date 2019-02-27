import numpy as np
from scipy.spatial.distance import euclidean


class DTW:

    def __init__(self, ts_a: list, ts_b=list, d=euclidean, max_warping_window=1):
        self.ts_a = ts_a
        self.ts_b = ts_b
        self.d = d
        self.max_warping_window = max_warping_window
        self.cost = self.calc_dist()

    @staticmethod
    def filler(l: list, n=0):
        l.extend([0] * n)
        return l

    def calc_dist(self):
        # Create cost matrix via broadcasting with large int
        max_len = max(len(self.ts_a), len(self.ts_b))
        self.ts_a = self.filler(self.ts_a, max_len - len(self.ts_a))
        self.ts_b = self.filler(self.ts_b, max_len - len(self.ts_b))
        m, n = len(self.ts_a), len(self.ts_b)

        cost = self.d(self.ts_a, self.ts_b) * np.ones((m, n))

        # Initialize the first row and column
        cost[0, 0] = self.d(self.ts_a[0], self.ts_b[0])
        for i in range(1, m):
            cost[i, 0] = cost[i - 1, 0] + self.d(self.ts_a[i], self.ts_b[0])

        for j in range(1, n):
            cost[0, j] = cost[0, j - 1] + self.d(self.ts_a[0], self.ts_b[j])

        # Populate rest of cost matrix within window
        for i in range(1, m):
            for j in range(max(1, i - self.max_warping_window),
                           min(n, i + self.max_warping_window)):
                choices = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
                cost[i, j] = min(choices) + self.d(self.ts_a[i], self.ts_b[j])

        # Return DTW distance given window
        return cost[-1, -1]


if __name__ == "__main__":
    l1 = [1, 2, 3, 4]
    l2 = [1, 1, 2, 3]
    dist = DTW(l1, l2).cost
    print(dist)

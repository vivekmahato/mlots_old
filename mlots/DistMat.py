from multiprocessing import Pool

import numpy as np
import pandas as pd

from mlots import Metrics


class DistMat:
    def __init__(self, dic: dict, train_keys: list, test_keys: list, pool_size=1, metric="WagnerFischer"):
        self.dic = dic
        if train_keys is not None and test_keys is not None:
            self.test_keys = test_keys
            self.train_keys = train_keys
            self.flag = False
        elif train_keys is None:
            self.test_keys = list(self.dic.keys())
            self.train_keys = list(self.dic.keys())
            self.flag = True
        self.pool_size = pool_size
        self.metric = metric
        self.dist_mat = self.dist_matrix()

    def dist_matrix(self):
        pool = Pool(self.pool_size)
        data = pool.map(self.dist_exe, self.test_keys)
        pool.close()
        pool.join()
        d_mat = pd.concat(data)
        if not self.flag:
            m = np.array(d_mat)
            inds = np.triu_indices_from(m, k=1)
            m[(inds[1], inds[0])] = m[inds]
            d_mat = pd.DataFrame(m, columns=self.train_keys, index=self.test_keys)
        return d_mat

    def dist_exe(self, row):
        if self.flag:
            cols = self.train_keys
            dm = pd.DataFrame(columns=cols, index=[row])
        else:
            keys = self.train_keys
            idx = keys.index(row)
            cols = keys[idx + 1:]
            dm = pd.DataFrame(columns=cols, index=[row])
            dm[row][row] = 0
        for col in cols:
            dist = Metrics(self.dic[row].data, self.dic[col].data, self.metric).cost
            dm[col] = dist
        return dm

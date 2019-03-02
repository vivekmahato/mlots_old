from mlots.timeseries import TimeSeries


class TransformData:
    def __init__(self, data=None, labels=None, test=False):
        self.data = data
        self.labels = labels
        if test:
            self.transformed_data = self.dummy_data()
        else:
            self.transformed_data = self.prep_data()

    def prep_data(self):
        t_dic = {}
        keys = list()
        i = 0
        for k, v in self.data.items():
            keys.append(k)
            ts = v
            lbl = self.labels[i]
            ts = TimeSeries(ts, lbl)
            t_dic[i] = ts
            i += 1
        t_dic["Samples"] = i
        t_dic["Size"] = len(t_dic[0].data)
        t_dic["Labels"] = self.labels
        t_dic["Keys"] = keys
        return t_dic

    def dummy_data(self):
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
        t_dic = {}
        keys = list()
        i = 0
        for k, v in data.items():
            keys.append(k)
            ts = v
            lbl = labels[i]
            ts = TimeSeries(ts, lbl)
            t_dic[i] = ts
            i += 1
        t_dic["Samples"] = i
        t_dic["Size"] = len(t_dic[0].data)
        t_dic["Labels"] = labels
        t_dic["Keys"] = keys
        return t_dic

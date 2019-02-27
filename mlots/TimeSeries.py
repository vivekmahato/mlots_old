import matplotlib.pyplot as plt


class TimeSeries:
    def __init__(self, dic_attr: dict):
        for key in dic_attr.keys():
            self.__setattr__(key, dic_attr[key])

    # def __getattr__(self, item):
    #    print(item, " Attribute doesn't exist!")

    def __str__(self):
        attributes = [i for i in dir(self) if not i.startswith('__')]
        s = "TimeSeries has the following attributes : [" + ", ".join(attributes) + "]"
        return s

    def plot(self):
        try:
            plt.title("Time Series")
            plt.plot(self.data, "r")
            plt.show()
        except():
            return None


if __name__ == "__main__":
    data = [0, 0, 2, 0, 3, 0, 0]
    dic = {"data": data, "label": "test"}
    ts = TimeSeries(dic)
    ts.plot()

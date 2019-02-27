import editdistance as ed


class EditDistance:

    def __init__(self, ts_a: list, ts_b: list):
        self.ts_a = ts_a
        self.ts_b = ts_b
        self.cost = self.edr()

    def edr(self):
        dist = ed.eval(self.ts_a, self.ts_b)
        return dist

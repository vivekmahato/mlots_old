from saxpy.sax import sax_via_window


class SAX:

    def __init__(self, ts: list, win_size=5, paa_size=5, alphabet_size=5, nr_strategy='none', z_threshold=0.01):
        self.ts = ts
        self.win_size = win_size
        self.paa_size = paa_size
        self.alphabet_size = alphabet_size
        self.nr_strategy = nr_strategy
        self.z_threshold = z_threshold
        self.sax_keys = self.alphabetize()

    def alphabetize(self):
        sax = sax_via_window(self.ts, win_size=self.win_size, paa_size=self.paa_size, alphabet_size=self.alphabet_size,
                             nr_strategy=self.nr_strategy, z_threshold=self.z_threshold)
        sax = list(sax.keys())
        return sax


if __name__ == "__main__":
    l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 1, 2, 3, 4, 5]
    keys = SAX(l).sax_keys
    print(keys)

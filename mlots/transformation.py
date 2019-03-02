from mlots.sax import SAX
from mlots.sfa import SFA


class Transformation:
    def __init__(self, data=None, method="sax", **kwargs):
        self.data = data
        if method == "sax":
            self.win_size = kwargs['win_size']
            self.paa_size = kwargs['paa_size']
            self.alphabet_size = kwargs['alphabet_size']
            self.nr_strategy = kwargs['nr_strategy']
            self.z_threshold = kwargs['z_threshold']
            self.transformed = self.sax_transformation()
        elif method == "sfa":
            self.histogram = kwargs['histogram_type']
            self.win_size = kwargs['win_size']
            self.word_length = kwargs['word_length']
            self.alphabet_size = kwargs['alphabet_size']
            self.norm_mean = kwargs['norm_mean']
            self.l_bound = kwargs['lower_bound']
            self.transformed = self.sfa_transformation()

    def sax_transformation(self):
        for key in self.data["Keys"]:
            t_ts = SAX(self.data[key].data, self.win_size, self.paa_size, self.alphabet_size, self.nr_strategy,
                       self.z_threshold).sax_keys
            self.data[key].data = t_ts
        return self.data

    def sfa_transformation(self):
        model = SFA(self.histogram)
        model.fitWindowing(self.data, window_size=self.win_size, word_length=self.word_length,
                           symbols=self.alphabet_size, norm_mean=self.norm_mean, lower_bounding=self.l_bound)
        sym_dic = model.alphabetize(self.data)
        return sym_dic

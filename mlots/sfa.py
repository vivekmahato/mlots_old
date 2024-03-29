import pandas as pd

from mlots.mft import *
from mlots.timeseries import *

'''
 Symbolic Fourier Approximation as published in
 Schäfer, P., Högqvist, M.: SFA: a symbolic fourier approximation and
 index for similarity search in high dimensional datasets.
 In: EDBT, ACM (2012)
'''


def sfa2word(word):
    word_string = ""
    alphabet = "abcdefghijklmnopqrstuv"
    for w in word:
        word_string += alphabet[w]
    return word_string


def sfa2wordlist(wordList):
    list_string = []
    for word in wordList:
        list_string.append(sfa2word(word))
    return list_string


class SFA:

    def __init__(self, histogram_type="EQUI_DEPTH", sup=False, lb=True, mb=False):
        self.initialized = False
        self.histogram_type = histogram_type
        self.sup = sup
        self.lower_bounding = lb
        self.muse_bool = mb

    def initialize(self, word_length, symbols, norm_mean):
        self.initialized = True
        self.word_length = word_length
        self.max_word_length = word_length
        self.symbols = symbols
        self.norm_mean = norm_mean
        self.alphabet_size = symbols
        self.transformation = None
        self.order_line = []
        self.bins = pd.DataFrame(np.zeros((word_length, self.alphabet_size))) + math.inf
        self.bins.iloc[:, 0] = -math.inf

    def printBins(self):
        print(self.bins)

    def alphabetize(self, dic: dict):
        for idx in range(dic["Samples"]):
            word_list = self.transform_windowing(dic[idx])
            sym = sfa2wordlist(word_list)
            dic[idx].data = sym
        return dic

    def mv_fitWindowing(self, samples, window_size, word_length, symbols, norm_mean, lower_bounding):
        sa = {}
        index = 0
        for i in range(samples["Samples"]):
            for k in range(len(samples[i].keys())):
                new_list = getDisjointSequences(samples[i][k], window_size, norm_mean)
                for j in range(len(new_list)):
                    sa[index] = new_list[j]
                    index += 1

        sa["Samples"] = index
        self.fitWindowing(sa, window_size, word_length, symbols, norm_mean, lower_bounding)

    def fitWindowing(self, samples, window_size, word_length, symbols, norm_mean, lower_bounding):
        self.transformation = MFT(window_size, norm_mean, lower_bounding, self.muse_bool)
        sa = {}
        index = 0
        for i in range(samples["Samples"]):
            new_list = getDisjointSequences(samples[i], window_size, norm_mean)
            for j in range(len(new_list)):
                sa[index] = new_list[j]
                index += 1
        sa["Samples"] = index
        if self.sup:
            self.fitTransformSupervised(sa, word_length, symbols, norm_mean)
        else:
            self.fit_transform(sa, word_length, symbols, norm_mean)

    def fit_transform(self, samples, word_length, symbols, norm_mean):
        return self.transform(samples, self.fit_transformDouble(samples, word_length, symbols, norm_mean))

    def transform(self, samples, approximate):
        transformed = []
        for i in range(samples["Samples"]):
            transformed.append(self.transform2(samples[i].data, approximate[i]))

        return transformed

    def transform2(self, series, one_approx):
        if one_approx == "null":
            one_approx = self.transformation.transform(series, self.max_word_length)

        if self.sup:
            return self.quantizationSupervised(one_approx)
        return self.quantization(one_approx)

    def transform_windowing(self, series):
        mft = self.transformation.transform_windowing(series, self.max_word_length)

        words = []
        for i in range(len(mft)):
            if self.sup:
                words.append(self.quantizationSupervised(mft[i]))
            else:
                words.append(self.quantization(mft[i]))

        return words

    def transform_windowingInt(self, series, word_length):
        words = self.transform_windowing(series)
        intWords = []
        for i in range(len(words)):
            intWords.append(self.createWord(words[i], word_length, self.int2byte(self.alphabet_size)))
        return intWords

    def fit_transformDouble(self, samples, word_length, symbols, norm_mean):
        if not self.initialized:
            self.initialize(word_length, symbols, norm_mean)

            if self.transformation is None:
                self.transformation = MFT(len(samples[0].data), norm_mean, self.lower_bounding, self.muse_bool)

        transformed_samples = self.fillOrderline(samples, word_length)

        if self.histogram_type == "EQUI_DEPTH":
            self.divideEquiDepthHistogram()
        elif self.histogram_type == "EQUI_FREQUENCY":
            self.divideEquiWidthHistogram()
        elif self.histogram_type == "INFORMATION_GAIN":
            self.divideHistogramInformationGain()

        return transformed_samples

    def fillOrderline(self, samples, word_length):
        self.order_line = [[None for _ in range(samples["Samples"])] for _ in range(word_length)]

        transformedSamples = []
        for i in range(samples["Samples"]):
            transformedSamples_small = self.transformation.transform(samples[i].data, word_length)
            transformedSamples.append(transformedSamples_small)
            for j in range(len(transformedSamples_small)):
                value = float(
                    format(round(transformedSamples_small[j], 2), '.2f')) + 0  # is a bad way of removing values of -0.0
                obj = (value, samples[i].label)
                self.order_line[j][i] = obj

        for l, list in enumerate(self.order_line):
            del_list = list
            new_list = []
            while len(del_list) != 0:
                current_min_value = math.inf
                current_min_location = -1
                label = -math.inf
                for j in range(len(del_list)):
                    # print(type(del_list[j][0]),type(label))
                    # print(del_list[j][0],label)
                    if (del_list[j][0] < current_min_value) | (
                            (del_list[j][0] == current_min_value) & (del_list[j][1] < label)):
                        current_min_value = del_list[j][0]
                        label = del_list[j][1]
                        current_min_location = j
                new_list.append(del_list[current_min_location])
                del del_list[current_min_location]
            self.order_line[l] = new_list

        return transformedSamples

    def quantization(self, one_approx):
        i = 0
        word = [0 for _ in range(len(one_approx))]
        for v in one_approx:
            c = 0
            for C in range(self.bins.shape[1]):
                if v < self.bins.iloc[i, c]:
                    break
                else:
                    c += 1
            word[i] = c - 1
            i += 1
        return word

    def divideEquiDepthHistogram(self):
        for i in range(self.bins.shape[0]):
            depth = len(self.order_line[i]) / self.alphabet_size
            try:
                depth = len(self.order_line[i]) / self.alphabet_size
            except:
                depth = 0

            pos = 0
            count = 0
            try:
                for j in range(len(self.order_line[i])):
                    count += 1
                    condition1 = count > math.ceil(depth * (pos))
                    condition2 = pos == 1
                    condition3 = self.bins.iloc[i, pos - 1] != self.order_line[i][j][0]
                    if (condition1) & (condition2 | condition3):
                        self.bins.iloc[i, pos] = round(self.order_line[i][j][0], 2)
                        pos += 1
            except:
                pass
        self.bins.iloc[:, 0] = -math.inf

    def divideEquiWidthHistogram(self):
        i = 0
        for element in self.order_line:
            if len(element) != 0:
                first = element[0][0]
                last = element[-1][0]
                intervalWidth = (last - first) / self.alphabet_size

                for c in range(self.alphabet_size - 1):
                    self.bins.iloc[i, c] = intervalWidth * (c + 1) + first
            i += 1

        self.bins.iloc[:, 0] = -math.inf

    def divideHistogramInformationGain(self):
        for i, element in enumerate(self.order_line):
            self.splitPoints = []
            self.findBestSplit(element, 0, len(element), self.alphabet_size)
            self.splitPoints.sort()
            for j in range(len(self.splitPoints)):
                self.bins.iloc[i, j + 1] = element[self.splitPoints[j] + 1][0]

    def findBestSplit(self, element, start, end, remainingSymbols):
        bestGain = -1
        bestPos = -1
        total = end - start
        self.cOut = {}
        self.cIn = {}
        for pos in range(start, end):
            label = element[pos][1]
            self.cOut[label] = self.cOut[label] + 1. if label in self.cOut.keys() else 1.
        class_entropy = self.entropy(self.cOut, total)
        i = start
        lastLabel = element[i][1]
        i += self.moveElement(lastLabel)
        for split in range(start + 1, end - 1):
            label = element[i][1]
            i += self.moveElement(label)
            if label != lastLabel:
                gain = self.calculateInformationGain(class_entropy, i, total)
                if gain >= bestGain:
                    bestGain = gain
                    bestPos = split
            lastLabel = label
        if bestPos > -1:
            self.splitPoints.append(bestPos)
            remainingSymbols = remainingSymbols / 2
            if remainingSymbols > 1:
                if (bestPos - start > 2) & (end - bestPos > 2):
                    self.findBestSplit(element, start, bestPos, remainingSymbols)
                    self.findBestSplit(element, bestPos, end, remainingSymbols)
                elif end - bestPos > 4:
                    self.findBestSplit(element, bestPos, int((end - bestPos) / 2), remainingSymbols)
                    self.findBestSplit(element, int((end - bestPos) / 2), end, remainingSymbols)
                elif bestPos - start > 4:
                    self.findBestSplit(element, start, int((bestPos - start) / 2), remainingSymbols)
                    self.findBestSplit(element, int((bestPos - start) / 2), end, remainingSymbols)

    def moveElement(self, label):
        self.cIn[label] = self.cIn[label] + 1. if label in self.cIn.keys() else 1.
        self.cOut[label] = self.cOut[label] - 1. if label in self.cOut.keys() else -1.
        return 1

    def entropy(self, freq, total):
        e = 0
        if total != 0:
            log2 = 1.0 / math.log(2)
            for k in freq.keys():
                p = freq[k] / total
                if p > 0:
                    e += -1 * p * math.log(p) * log2
        else:
            e = math.inf
        return e

    def calculateInformationGain(self, class_entropy, total_c_in, total):
        total_c_out = total - total_c_in
        return class_entropy - total_c_in / total * self.entropy(self.cIn,
                                                                 total_c_in) - total_c_out / total * self.entropy(
            self.cOut, total_c_out)

    def createWord(self, numbers, maxF, bits):
        shortsPerLong = int(round(60 / bits))
        to = min(len(numbers), maxF)
        b = int(0)
        s = 0
        shiftOffset = 1
        for i in range(s, (min(to, shortsPerLong + s))):
            shift = 1
            for j in range(bits):
                if (numbers[i] & shift) != 0:
                    b |= shiftOffset
                shiftOffset <<= 1
                shift <<= 1
        return b

    def int2byte(self, number):
        log = 0
        if (number & 0xffff0000) != 0:
            number >>= 16
            log = 16
        if number >= 256:
            number >>= 8
            log += 8
        if number >= 16:
            number >>= 4
            log += 4
        if number >= 4:
            number >>= 2
            log += 2
        return log + (number >> 1)

    ## Supervised
    def fitTransformSupervised(self, samples, word_length, symbols, norm_mean):
        length = len(samples[0].data)
        transformed_signal = self.fit_transformDouble(samples, length, symbols, norm_mean)

        best = self.calcBestCoefficients(samples, transformed_signal)

        self.best_values = [0 for i in range(min(len(best), word_length))]
        self.max_word_length = 0

        for i in range(len(self.best_values)):
            self.best_values[i] = best[i][0]
            self.max_word_length = max(best[i][0] + 1, self.max_word_length)

        self.max_word_length += self.max_word_length % 2

        return self.transform(samples, transformed_signal)

    def calcBestCoefficients(self, samples, transformed_signal):
        classes = {}
        for i in range(samples["Samples"]):
            if samples[i].label in classes.keys():
                classes[samples[i].label].append(transformed_signal[i])
            else:
                classes[samples[i].label] = [transformed_signal[i]]
        n_samples = len(transformed_signal)
        n_classes = len(classes.keys())
        length = len(transformed_signal[1])
        f = self.getFoneway(length, classes, n_samples, n_classes)
        f_sorted = sorted(f, reverse=True)
        best = []
        inf_index = 0
        for value in f_sorted:
            if value == math.inf:
                index = f.index(value) + inf_index
                inf_index += 1
            else:
                index = f.index(value)
            best.append([index, value])
        return best

    def getFoneway(self, length, classes, n_samples, n_classes):
        ss_alldata = [0. for i in range(length)]
        sums_args = {}
        keys_class = list(classes.keys())
        for key in keys_class:
            allTs = classes[key]
            sums = [0. for i in range(len(ss_alldata))]
            sums_args[key] = sums
            for ts in allTs:
                for i in range(len(ts)):
                    ss_alldata[i] += ts[i] * ts[i]
                    sums[i] += ts[i]
        square_of_sums_alldata = [0. for i in range(len(ss_alldata))]
        square_of_sums_args = {}
        for key in keys_class:
            # square_of_sums_alldata2 = [0. for i in range(len(ss_alldata))]
            sums = sums_args[key]
            for i in range(len(sums)):
                square_of_sums_alldata[i] += sums[i]
            # square_of_sums_alldata += square_of_sums_alldata2
            squares = [0. for i in range(len(sums))]
            square_of_sums_args[key] = squares
            for i in range(len(sums)):
                squares[i] += sums[i] * sums[i]
        for i in range(len(square_of_sums_alldata)):
            square_of_sums_alldata[i] *= square_of_sums_alldata[i]
        sstot = [0. for i in range(len(ss_alldata))]
        for i in range(len(sstot)):
            sstot[i] = ss_alldata[i] - square_of_sums_alldata[i] / n_samples
        ssbn = [0. for i in range(len(ss_alldata))]  ## sum of squares between
        sswn = [0. for i in range(len(ss_alldata))]  ## sum of squares within
        for key in keys_class:
            sums = square_of_sums_args[key]
            n_samples_per_class = len(classes[key])
            for i in range(len(sums)):
                ssbn[i] += sums[i] / n_samples_per_class
        for i in range(len(square_of_sums_alldata)):
            ssbn[i] += -square_of_sums_alldata[i] / n_samples
        dfbn = n_classes - 1  ## degrees of freedom between
        dfwn = n_samples - n_classes  ## degrees of freedom within
        msb = [0. for i in range(len(ss_alldata))]  ## variance (mean square) between classes
        msw = [0. for i in range(len(ss_alldata))]  ## variance (mean square) within samples
        f = [0. for i in range(len(ss_alldata))]  ## f-ratio
        for i in range(len(sswn)):
            sswn[i] = sstot[i] - ssbn[i]
            msb[i] = ssbn[i] / dfbn
            msw[i] = sswn[i] / dfwn
            f[i] = msb[i] / msw[i] if msw[i] != 0. else math.inf
        return f

    def quantizationSupervised(self, one_approx):
        signal = [0 for _ in range(min(len(one_approx), len(self.best_values)))]
        for a in range(len(signal)):
            i = self.best_values[a]
            b = 0
            for beta in range(self.bins.shape[1]):
                if one_approx[i] < self.bins.iloc[i, beta]:
                    break
                else:
                    b += 1
            signal[a] = b - 1
        return signal

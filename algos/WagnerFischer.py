### Reference: https://gist.github.com/kylebgorman/1081951/9b38b7743a3cb5167ab2c6608ac8eea7fc629dca


import collections


# Default cost functions.


def INSERTION(A):
    alphas = "abcdefghijklmnopqrstuvwxyz"
    cost = (alphas.find(A) + 1)
    return cost


def DELETION(A):
    cost = 1
    return cost


def SUBSTITUTION(A, B):
    alphas = "abcdefghijklmnopqrstuvwxyz"
    cost = abs((alphas.find(A) + 1) - (alphas.find(B) + 1))
    return cost


Trace = collections.namedtuple("Trace", ["cost", "ops"])


class WagnerFischer(object):

    def __init__(self, a, b, insertion=INSERTION, deletion=DELETION,
                 substitution=SUBSTITUTION):
        # Stores cost functions in a dictionary for programmatic access.
        self.costs = {"I": insertion, "D": deletion, "S": substitution}

        if not isinstance(a, list) and isinstance(b, list):
            a = [a]

        if not isinstance(b, list) and isinstance(a, list):
            b = [b]

        if isinstance(a, list) and isinstance(b, list):
            a_size = len(a)
            b_size = len(b)
            if a_size == b_size:
                f_cost = 0
                for i in range(len(a)):
                    pair = [a[i], b[i]]
                    pair.sort()
                    f_cost += WagnerFischer(pair[0], pair[1]).cost
                self.cost = f_cost
            else:
                diff = a_size - b_size
                if diff > 0:
                    for i in range(diff):
                        b.append("")
                elif diff < 0:
                    for i in range(abs(diff)):
                        a.append("")
                f_cost = 0
                for i in range(len(a)):
                    pair = [a[i], b[i]]
                    pair.sort()
                    f_cost += WagnerFischer(pair[0], pair[1]).cost
                    self.cost = f_cost

        elif not isinstance(a, list) and not isinstance(b, list):
            # Initializes table.
            pair = [a, b]
            pair.sort()
            a = pair[0]
            b = pair[1]
            self.asz = len(a)
            self.bsz = len(b)
            self._table = [[None for _ in range(self.bsz + 1)] for
                           _ in range(self.asz + 1)]
            # From now on, all indexing done using self.__getitem__.
            # Fills in edges.
            self[0][0] = Trace(0, {"O"})  # Start cell.
            for i in range(1, self.asz + 1):
                self[i][0] = Trace(self[i - 1][0].cost + self.costs["D"](a[i - 1]),
                                   {"D"})
            for j in range(1, self.bsz + 1):
                self[0][j] = Trace(self[0][j - 1].cost + self.costs["I"](b[j - 1]),
                                   {"I"})
            # Fills in rest.
            for i in range(len(a)):
                for j in range(len(b)):
                    # Cleans it up in case there are more than one check for match
                    # first, as it is always the cheapest option.
                    if a[i] == b[j]:
                        self[i + 1][j + 1] = Trace(self[i][j].cost, {"M"})
                    # Checks for other types.
                    else:
                        costD = self[i][j + 1].cost + self.costs["D"](a[i])
                        costI = self[i + 1][j].cost + self.costs["I"](b[j])
                        costS = self[i][j].cost + self.costs["S"](a[i], b[j])
                        min_val = min(costI, costD, costS)
                        trace = Trace(min_val, set())
                        # Adds _all_ operations matching minimum value.
                        if costD == min_val:
                            trace.ops.add("D")
                        if costI == min_val:
                            trace.ops.add("I")
                        if costS == min_val:
                            trace.ops.add("S")
                        self[i + 1][j + 1] = trace
            # Stores optimum cost as a property.
            self.cost = self[-1][-1].cost

    def __repr__(self):
        return self.pprinter.pformat(self._table)

    def __iter__(self):
        for row in self._table:
            yield row

    def __getitem__(self, i):
        """
        Returns the i-th row of the table, which is a list and so
        can be indexed. Therefore, e.g.,  self[2][3] == self._table[2][3]
        """
        return self._table[i]

    # Stuff for generating alignments.

    def _stepback(self, i, j, trace, path_back):
        """
        Given a cell location (i, j) and a Trace object trace, generate
        all traces they point back to in the table
        """
        for op in trace.ops:
            if op == "M":
                yield i - 1, j - 1, self[i - 1][j - 1], path_back + ["M"]
            elif op == "I":
                yield i, j - 1, self[i][j - 1], path_back + ["I"]
            elif op == "D":
                yield i - 1, j, self[i - 1][j], path_back + ["D"]
            elif op == "S":
                yield i - 1, j - 1, self[i - 1][j - 1], path_back + ["S"]
            elif op == "O":
                return  # Origin cell, so we"re done.
            else:
                raise ValueError("Unknown op {!r}".format(op))

    def alignments(self):
        """
        Generate all alignments with optimal-cost via breadth-first
        traversal of the graph of all optimal-cost (reverse) paths
        implicit in the dynamic programming table
        """
        # Each cell of the queue is a tuple of (i, j, trace, path_back)
        # where i, j is the current index, trace is the trace object at
        # this cell, and path_back is a reversed list of edit operations
        # which is initialized as an empty list.
        queue = collections.deque(self._stepback(self.asz, self.bsz,
                                                 self[-1][-1], []))
        while queue:
            (i, j, trace, path_back) = queue.popleft()
            if trace.ops == {"O"}:
                # We have reached the origin, the end of a reverse path, so
                # yield the list of edit operations in reverse.
                yield path_back[::-1]
                continue
            queue.extend(self._stepback(i, j, trace, path_back))

    def IDS(self):
        """
        Estimates insertions, deletions, and substitution _count_ (not
        costs). Non-integer values arise when there are multiple possible
        alignments with the same cost.
        """
        npaths = 0
        opcounts = collections.Counter()
        for alignment in self.alignments():
            # Counts edit types for this path, ignoring "M" (which is free).
            opcounts += collections.Counter(op for op in alignment if op != "M")
            npaths += 1
        # Averages over all paths.
        return collections.Counter({o: c / npaths for (o, c) in opcounts.items()})


if __name__ == "__main__":
    a = "coat"
    b = "goat"
    d = WagnerFischer
    dist = d(a, b).cost
    print(dist)

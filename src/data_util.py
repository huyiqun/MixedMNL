import string
import numpy as np
from scipy.special import logsumexp
from logging_util import ColoredLog
DEFAULT_VERBOSE = 3


class Population(object):
    def __init__(self, data=None, num_type=None, num_feat=None, verbose=DEFAULT_VERBOSE):
        if data:
            self.parse_data(data)
        else:
            self.simulate(num_type, num_feat)

        self.cid = [f"Type {letter}" for letter in self.cluster_id]
        self.logger = ColoredLog(self.__class__.__name__, verbose=verbose)
        self.show_info()

    @classmethod
    def from_data(cls, data, **kwargs):
        return cls(data, **kwargs)

    @classmethod
    def from_sim(cls, num_type, num_feat, **kwargs):
        return cls(None, num_type, num_feat, **kwargs)

    def parse_data(self, data):
        self.cluster_id = []
        self.alpha = []
        self.preference_vec = []
        self.preference_dict = {}
        with open(data, "r") as f:
            for l in f:
                ctype, alpha, *pref_vec = l.strip().split(",")
                pref_vec = list(map(float, pref_vec))

                self.cluster_id.append(ctype)
                self.alpha.append(float(alpha))
                self.preference_vec.append(pref_vec)
                self.preference_dict[ctype] = pref_vec

        self.num_type = len(self.cluster_id)

    def simulate(self, num_type, num_feat):
        self.cluster_id = [letter for letter in string.ascii_uppercase[:num_type]]
        self.num_type = num_type
        rand_num = np.random.choice(np.arange(1, 5, 0.25), num_type, replace=False)
        [x / sum(rand_num) for x in rand_num]
        self.alpha = [x / sum(rand_num) for x in rand_num]
        self.preference_dict = {}
        while True:
            rand_data = np.random.uniform(low=-1, high=1, size=(num_type, num_feat))
            if len(np.unique(rand_data, axis=0)) == self.num_type:
                self.preference_vec = rand_data.astype(float)
                self.preference_dict = {self.cluster_id[i]: rand_data[i] for i in range(num_type)}
                break

    def show_info(self):
        num_feat = len(self.preference_vec[0])
        table_title = ["alpha"] + [f"Feature {j+1}" for j in range(num_feat)]
        alpha_mat = [[a] for a in self.alpha]
        data_mat = np.append(alpha_mat, self.preference_vec, axis=1)
        table = np.vstack((table_title, data_mat))
        self.logger.info(table, caption="Consumer info", index=self.cid)

    def save(self, filename):
        data = []
        for k in range(self.num_type):
            row = f"{self.cluster_id[k]},{self.alpha[k]}," + ",".join(
                list(map(str, self.preference_vec[k]))
            )
            data.append(row)
        with open(filename, "w") as f:
            f.write("\n".join(data))


class Product_Set(object):
    def __init__(
        self,
        data=None,
        num_prod=None,
        num_feat=None,
        verbose=DEFAULT_VERBOSE,
    ):
        if data:
            self.parse_data(data)
        else:
            self.simulate(num_prod, num_feat)

        self.pid = [f"Product {i}" for i in self.prod_id]
        self.pid_off = ["Offset"] + self.pid
        self.logger = ColoredLog(self.__class__.__name__, verbose=verbose)
        self.show_info()

    @classmethod
    def from_data(cls, data, **kwargs):
        return cls(data=data, **kwargs)

    @classmethod
    def from_sim(cls, num_prod, num_feat, **kwargs):
        return cls(num_prod=num_prod, num_feat=num_feat, **kwargs)

    def parse_data(self, data):
        self.prod_id = []
        self.features = []
        with open(data, "r") as f:
            for l in f:
                prod_id, *feature = l.strip().split(",")
                feature = list(map(float, feature))

                self.prod_id.append(prod_id)
                self.features.append(feature)

        self.num_prod = len(self.prod_id)
        self.num_feat = len(self.features[0])

    def simulate(self, num_prod, num_feat):
        self.prod_id = [i + 1 for i in range(num_prod)]
        self.num_prod = num_prod
        self.num_feat = num_feat
        while True:
            rand_data = np.random.uniform(low=-1, high=1, size=(num_prod, num_feat))
            if len(np.unique(rand_data, axis=0)) == self.num_prod:
                self.features = rand_data.astype(float)
                break

    def show_info(self):
        num_feat = len(self.features[0])
        table_title = [f"Feature {j+1}" for j in range(num_feat)]
        self.logger.info(self.features, caption="Product info", header=table_title, index=self.pid)

    def _exp_vec(self, weights, prices):
        return [
            np.dot(f, weights) - p
            for f, p in zip(self.features, prices)
        ]

    def _exp_calc(self, weights, prices):
        t = self._exp_vec(weights, prices)
        expnom = [np.exp(i) for i in t]
        expsum = logsumexp(t)
        return expnom, expsum

    def choice_prob_vec(self, weights, prices):
        v, s = self._exp_calc(weights, prices)
        vec = [1 / (np.exp(s) + 1)] + [t / (np.exp(s) + 1) for t in v]
        return np.array(vec)

    def choice_prob(self, weights, prices, i):
        return self.choice_prob_vec(weights, prices)[i]

    def save(self, filename):
        data = []
        for j in range(self.num_prod):
            row = f"{self.prod_id[j]}," + ",".join(
                list(map(str, self.features[j]))
            )
            data.append(row)
        with open(filename, "w") as f:
            f.write("\n".join(data))


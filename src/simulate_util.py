import string
import numpy as np
from scipy.special import logsumexp
from logging_util import ColoredLog
from settings import VERBOSE


class Population(object):
    def __init__(self, data=None, num_types=None, feature_len=None, verbose=VERBOSE):
        if data:
            self.parse_data(data)
        else:
            self.simulate(num_types, feature_len)

        self.cid = [f"Type {letter}" for letter in self.cluster_id]
        self.logger = ColoredLog(self.__class__.__name__, verbose=verbose)
        self.logger.info(self.info, caption="Consumer info", index=self.cid)

    @classmethod
    def from_data(cls, data, **kwargs):
        return cls(data, **kwargs)

    @classmethod
    def from_sim(cls, num_types, feature_len, **kwargs):
        return cls(None, num_types, feature_len, **kwargs)

    def parse_data(self, data):
        self.cluster_id = []
        self.alpha = []
        self.preference_vec = []
        with open(data, "r") as f:
            for l in f:
                type, alpha, *pref_vec = l.strip().split(",")
                pref_vec = list(map(float, pref_vec))

                self.cluster_id.append(type)
                self.alpha.append(float(alpha))
                self.preference_vec.append(pref_vec)

        self.num_types = len(self.cluster_id)

    def simulate(self, num_types, feature_len):
        self.cluster_id = [letter for letter in string.ascii_uppercase[:num_types]]
        self.num_types = num_types
        rand_num = np.random.choice(range(10, 510, 20), num_types, replace=False)
        self.alpha = [x / sum(rand_num) for x in rand_num]
        while True:
            rand_data = np.random.uniform(low=0, high=2, size=(num_types, feature_len))
            if len(np.unique(rand_data, axis=0)) == self.num_types:
                self.preference_vec = rand_data.astype(float)
                break

    @property
    def info(self):
        feature_len = len(self.preference_vec[0])
        table_title = ["alpha"] + [f"Feature {j+1}" for j in range(feature_len)]
        alpha_mat = [[a] for a in self.alpha]
        data_mat = np.append(alpha_mat, self.preference_vec, axis=1)
        table = np.vstack((table_title, data_mat))
        return table

    def save(self, filename):
        data = []
        for k in range(self.num_types):
            row = f"{self.cluster_id[k]},{self.alpha[k]}," + ",".join(
                list(map(str, self.preference_vec[k]))
            )
            data.append(row)
        with open(filename, "w") as f:
            f.write("\n".join(data))


def default_err_func(price, calc_true, w=None):
    return 0
    #  if w:
    #  return -1 * w * price
    #  else:
    #  return -1 * price


class Product_Set(object):
    def __init__(
        self,
        err=default_err_func,
        data=None,
        num_prod=None,
        feature_len=None,
        verbose=VERBOSE,
    ):
        if data:
            self.parse_data(data)
        else:
            self.simulate(num_prod, feature_len)

        self.err = err
        self.pid = [f"Product {i}" for i in self.prod_id]
        self.pid_off = self.pid + ["Offset"]
        self.logger = ColoredLog(self.__class__.__name__, verbose=verbose)
        self.logger.info(self.info, caption="Product info", index=self.pid)

    @classmethod
    def from_data(cls, data, **kwargs):
        return cls(data=data, **kwargs)

    @classmethod
    def from_sim(cls, num_prod, feature_len, **kwargs):
        return cls(num_prod=num_prod, feature_len=feature_len, **kwargs)

    def parse_data(self, data):
        self.prod_id = []
        self.prices = []
        self.features = []
        with open(data, "r") as f:
            for l in f:
                prod_id, price, *feature = l.strip().split(",")
                feature = list(map(float, feature))

                self.prod_id.append(prod_id)
                self.prices.append(float(price))
                self.features.append(feature)

        self.num_prod = len(self.prod_id)
        self.feature_len = len(self.features[0])

    def simulate(self, num_prod, feature_len):
        self.prod_id = [i + 1 for i in range(num_prod)]
        self.num_prod = num_prod
        self.prices = np.ones(self.num_prod)
        self.feature_len = feature_len
        while True:
            rand_data = np.random.uniform(low=0, high=2, size=(num_prod, feature_len))
            if len(np.unique(rand_data, axis=0)) == self.num_prod:
                self.features = rand_data.astype(float)
                break

    @property
    def info(self):
        feature_len = len(self.features[0])
        table_title = [f"Feature {j+1}" for j in range(feature_len)] + ["Price"]
        price_mat = [[p] for p in self.prices]
        data_mat = np.append(self.features, price_mat, axis=1)
        table = np.vstack((table_title, data_mat))
        return table

    def set_price(self, new_prices):
        self.prices = [p for p in new_prices]

    def _exp_vec(self, weights, prices, calc_true):
        #  if calc_true:
        #  err = lambda x: self.err(x, weights[0])
        #  else:
        #  err = lambda x: np.zeros(1)[0] * x
        return [
            np.dot(f, weights) - p + self.err(p, calc_true=calc_true)
            for f, p in zip(self.features, prices)
        ]

    def _exp_calc(self, weights, prices, calc_true):
        t = self._exp_vec(weights, prices, calc_true)
        #  t = [np.dot(f, weights) - p for f, p in zip(self.features, prices)]
        expnom = [np.exp(i) for i in t]
        expsum = logsumexp(t)
        return expnom, expsum

    def choice_prob_vec(self, weights, prices=None, calc_true=False):
        if prices is not None:
            v, s = self._exp_calc(weights, prices, calc_true)
        else:
            v, s = self._exp_calc(weights, self.prices, calc_true)

        vec = [t / (np.exp(s) + 1) for t in v] + [1 / (np.exp(s) + 1)]
        return np.array(vec)

    def choice_prob(self, weights, i, calc_true=False):
        if i == -1:
            return self.choice_prob_vec(weights, calc_true=calc_true)[-1]
        else:
            return self.choice_prob_vec(weights, calc_true=calc_true)[i - 1]

    def save(self, filename):
        data = []
        for j in range(self.num_prod):
            row = f"{self.prod_id[j]},{self.prices[j]}," + ",".join(
                list(map(str, self.features[j]))
            )
            data.append(row)
        with open(filename, "w") as f:
            f.write("\n".join(data))


if __name__ == "__main__":
    import settings

    if settings.SIMULATE:
        pop = Population.from_sim(settings.NUM_TYPE, settings.NUM_FEATURE)
        ps = Product_Set.from_sim(settings.NUM_PRODUCT, settings.NUM_FEATURE)
    else:
        pop = Population.from_data(settings.CONSUMER_DATA)
        ps = Product_Set.from_data(settings.PRODUCT_DATA)

    print(ps.choice_prob_vec(pop.preference_vec[0], calc_true=True))
    print(ps.choice_prob_vec(pop.preference_vec[1], calc_true=True))
    print(ps.choice_prob_vec(pop.preference_vec[0], calc_true=False))
    print(ps.choice_prob_vec(pop.preference_vec[1], calc_true=False))

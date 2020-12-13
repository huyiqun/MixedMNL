from collections import Counter, defaultdict
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.optimize import minimize, LinearConstraint, Bounds
from logging_util import ColoredLog

DEFAULT_VERBOSE = 3
# old_settings = np.geterr()
# np.seterr(over="raise")
# np.seterr(all="raise")
# np.seterr(**old_settings)

class Simulator(object):
    def __init__(self, product_set, population, num_cons=1000, verbose=DEFAULT_VERBOSE):
        self.logger = ColoredLog(self.__class__.__name__, verbose=verbose)
        self.ps = product_set
        self.pop = population
        self.num_cons = num_cons
        self.simulate_consumer(num_cons)

    def simulate_consumer(self, num_cons):
        self.num_cons = num_cons
        self.type_dict = {}
        self.member = defaultdict(list)

        for i in range(num_cons):
            tp = self.pop.cluster_id[
                int(np.random.choice(self.pop.num_type, 1, p=self.pop.alpha))
            ]
            self.type_dict[i] = tp
            self.member[tp].append(i)

        ct = Counter(self.type_dict.values())
        type_res = [
            [ct[cid], float(ct[cid]) / num_cons]
            for cid in self.pop.cluster_id
        ]
        self.logger.debug(
            type_res,
            caption="Simulated types",
            header=["Number", "Percent"],
            index=self.pop.cid,
        )

    def calculate_groud_truth(self, price):
        K = self.pop.num_type
        choice_prob_dict = {}
        ms = np.zeros(self.ps.num_prod + 1)
        for k in range(K):
            p_vec = self.ps.choice_prob_vec(self.pop.preference_vec[k], price)
            choice_prob_dict[self.pop.cluster_id[k]] = p_vec
            ms += self.pop.alpha[k] * p_vec
        return choice_prob_dict, ms

    def simulate_transaction(self, choice_prob_dict):
        m = self.ps.num_prod + 1
        sim_data = np.zeros((self.num_cons, m))

        for i in range(self.num_cons):
            tp = self.type_dict[i]
            purchase = np.random.choice(
                m, 1, p=choice_prob_dict[tp]
            )
            sim_data[i, purchase] = 1

        return sim_data

    def run_experiments(self, exp_price):
        self.exp_price = exp_price
        self.T = len(exp_price)
        self.data_hist = {}
        self.theoretical_market_share = {}
        self.simulated_market_share = {}
        self.choice_prob_dict = {}

        for t, p in enumerate(self.exp_price):
            choice_prob_dict, ms = self.calculate_groud_truth(p)
            self.choice_prob_dict[t] = choice_prob_dict
            self.theoretical_market_share[t] = ms
            sim_data = self.simulate_transaction(choice_prob_dict)
            self.data_hist[t] = sim_data
            self.simulated_market_share[t] = np.mean(sim_data, axis=0)

            self.logger.info(f"Experiment: {t}; price: {p}")
            self.logger.debug(np.vstack((self.theoretical_market_share[t], self.simulated_market_share[t])), caption=f"Market Share", header = self.ps.pid_off, index=["Thoretical", "Simulated"])

    def get_personal_cdf(self, T):
        cdf = {i: np.cumsum(np.mean([self.data_hist[t][i] for t in range(T)], axis=0)) for i in range(self.num_cons)}
        return cdf


class Sampler(object):
    def __init__(self, cdf_dict, type_dict, member, verbose=DEFAULT_VERBOSE):
        self.cdf_dict = cdf_dict
        self.type_dict = type_dict
        self.member = member
        self.cdf_mat = np.array(list(self.cdf_dict.values()))
        self.score = pairwise_distances(self.cdf_mat, metric=lambda x, y: np.linalg.norm(x-y, ord=np.inf))
        # self.score = [[np.linalg.norm(vi-vj, ord=np.inf) for j, vj in self.cdf_dict.items()] for i, vi in self.cdf_dict.items()]
        self.num_cons = len(self.cdf_dict)
        self.logger = ColoredLog(self.__class__.__name__, verbose=verbose)

    def create_sample(self, seed, num_samp, samp_func):
        samp_prob = [samp_func(sj) for sj in self.score[seed]]
        samp_prob = [s / sum(samp_prob) for s in samp_prob]
        sample = np.random.choice(self.num_cons, size=num_samp, p=samp_prob)
        self.logger.debug(f"{seed}: {self.type_dict[seed]}")
        types = [self.type_dict[i] for i in sample]
        ct = Counter(types)
        purity = ct.most_common()[0][1]/num_samp
        print(ct)
        print(purity)
        return samp_prob, sample, purity


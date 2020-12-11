from collections import Counter, defaultdict, namedtuple, OrderedDict
import pickle
import numpy as np
from sklearn.metrics import pairwise_distances
from tabulate import tabulate
from scipy.optimize import minimize, LinearConstraint, Bounds
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from simulate_util import Population, Product_Set
from logging_util import ColoredLog
from srfw import FrankWolfe, SubRegionFrankWolfe
from plot import d3plot, compare_prob_est
cmap = cm.get_cmap(name="Dark2")

DEFAULT_VERBOSE = 3
old_settings = np.geterr()
np.seterr(over="raise")
# np.seterr(all="raise")
#  np.seterr(**old_settings)


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

        self.personal_cdf = {i: np.cumsum(np.mean([hist[i] for _, hist in self.data_hist.items()], axis=0)) for i in range(self.num_cons)}


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

a = [1,1,2,1,2,1,2]
Counter(a).most_common(1)[0]

def mle_loss(x, sim, ss):
    diff = [
            np.mean(v[ss], axis=0) - sim.ps.choice_prob_vec(x, sim.exp_price[t])
            for t, v in sim.data_hist.items()
           ]
    loss = np.sum([np.linalg.norm(d) ** 2 for d in diff])
    return loss

def closest(true_q, q):
    dist = [np.linalg.norm(tq - q) for tq in true_q]
    return np.argmin(dist)

sim = Simulator(ps, pop, num_cons=N, verbose=3)
exp_price = np.random.uniform(low=0.5, high=1.5, size=(T, M))
seeds = np.random.choice(N, size=50, replace=False)
Counter([sim.type_dict[s] for s in seeds])
rand_price = [round(p, 2) for p in np.random.uniform(low=0.5, high=3, size=ps.num_prod)]
save_res = defaultdict(list)
all_est = {}
purity_res = []

np.save(os.path.join(exp_dir, "purity.npy"), purity_res)
with open(os.path.join(exp_dir, "min_dist.pkl"), "wb") as f:
    pickle.dump(save_res, f)

TT = np.arange(150, 170, 10)
for T in TT:
    run(ps, pop, sim, exp_price[:T], rand_price, seeds, save_res, all_est, purity_res, generate_plot=False)

plt.plot(np.arange(10, 170, 10), save_res["fw"], label="FW")
plt.plot(np.arange(10, 170, 10), save_res["srfw"], label="SRFW")
plt.xlabel("Experiment")
plt.ylabel("Average Dist to Closest True Choice Prob")
plt.legend
plt.savefig(os.path.join(exp_dir, "min_dist.png"))

plt.plot(np.arange(10, 170, 10), purity_res)
plt.xlabel("Experiment")
plt.ylabel("Average Subsample Purity")
plt.savefig(os.path.join(exp_dir, "purity.png"))

plt.plot(range(T), save_res["fw"], label="FW")
plt.plot(range(T), save_res["srfw"], label="SRFW")
plt.xlabel("Experiment")
plt.ylabel("Average Min Dist to Closest True q")
plt.legend()
plt.savefig(os.path.join(exp_dir, "dist.png"))
for t, e in yy.items():
    plt.plot(xx, e, label=f"Type {t}")
    plt.xlabel("Number of Experiments")
    plt.ylabel("MSE for MNL Estimation")
plt.legend()

def run(ps, pop, sim, exp_price, rand_price, seeds, save_res, all_est, purity_res, generate_plot=False):
    sim.run_experiments(exp_price)
    sampler = Sampler(sim.personal_cdf, sim.type_dict, sim.member, verbose=3)

    beta_set = []
    valid_seeds = []
    purity = []
    # for tp, mem in sim.member.items():
        # print(tp)
        # while Counter(seeds)[tp] < 2:
    # while len(np.unique(seeds)) < K:
    for seed in seeds:
        # seed = np.random.choice(sim.member[tp])
        tp = sim.type_dict[seed]
        pp, ss, pure = sampler.create_sample(seed, 200, lambda x: max(1-15*x, 1e-4))
        # pp, ss = sampler.create_sample(np.random.randint(sim.num_cons), 200, lambda x: np.exp(-10*x))
        res = minimize(lambda x: mle_loss(x, sim, ss), np.random.rand(ps.num_feat), tol=1e-4)
        if res.success:
            valid_seeds.append(tp)
            beta_set.append(res.x)
            purity.append(pure)

    estimators = defaultdict(list)
    for tp, b in zip(valid_seeds, beta_set):
        estimators[tp].append(sim.ps.choice_prob_vec(b, rand_price))

    bscopy = np.copy(beta_set)
    all_est[len(exp_price)] = bscopy
    purity_res.append(np.mean(purity))

    fw = FrankWolfe(sim, verbose=3)
    fw.run()

    beta_set = np.copy(bscopy)
    srfw = SubRegionFrankWolfe(sim, verbose=3)
    srfw.run(beta_set)

    if generate_plot:
        # plot 1
        true_q = np.asarray([ps.choice_prob_vec(p, rand_price) for p in pop.preference_vec])
        d3plot(ps, pop, rand_price, beta_set, exp_dir, value_range=20, granularity=10, single=True)
        # plot 2
        fig, ax = plt.subplots(len(estimators), 1, sharex=True, figsize=(6, 2*len(estimators)+1))
        for i, t in enumerate(estimators.keys()):
            ground_truth = ps.choice_prob_vec(pop.preference_dict[t], rand_price)
            compare_prob_est(estimators, t, ground_truth, ps, ax[i])
            plt.tight_layout()
        lgd = plt.legend(bbox_to_anchor=(1,1.1), loc="upper left")
        plt.savefig(os.path.join(exp_dir, "type.png"), bbox_extra_artists=(lgd,), bbox_inches='tight')
        # plot 3
        d3plot_result(ps, pop, rand_price, exp_dir, fw, srfw)

    dist_fw = []
    dist_srfw = []
    for t, p in enumerate(sim.exp_price):
        true_q = [ps.choice_prob_vec(b, p) for b in pop.preference_vec]
        dist = np.mean([np.linalg.norm(true_q[closest(true_q, ps.choice_prob_vec(b, p))]-ps.choice_prob_vec(b, p)) for b in fw.active_beta])
        dist_fw.append(dist)
        dist = np.mean([np.linalg.norm(true_q[closest(true_q, ps.choice_prob_vec(b, p))]-ps.choice_prob_vec(b, p)) for b in srfw.active_beta])
        dist_srfw.append(dist)

    ave_fw = np.mean(dist_fw)
    ave_srfw = np.mean(dist_srfw)
    save_res["fw"].append(ave_fw)
    save_res["srfw"].append(ave_srfw)


# sim = Simulator(ps, pop, verbose=4)
# all_est = {}

# def investigate_num_exp():
    # T = 300
    # for T in np.arange(10, 510, 10):
        # sim.run_experiments(np.random.uniform(low=0.5, high=1.5, size=(T, M)))
        # sampler = Sampler(sim.personal_cdf, sim.type_dict, sim.member, verbose=4)
        # beta_set = []
        # seeds = []
        # estimators = defaultdict(list)

        # for _ in range(20):
            # seed = np.random.randint(sim.num_cons)
            # tp = sim.type_dict[seed]
            # seeds.append(tp)
            # pp, ss = sampler.create_sample(seed, 200, lambda x: max(1-10*x, 1e-4))
            # res = minimize(lambda x: mle_loss(x, sim, ss), np.random.rand(2), tol=1e-6)
            # beta_set.append(res.x)
            # estimators[tp].append(sim.ps.choice_prob_vec(res.x, np.ones(2)))
        # all_est[T] = estimators

    # estimators
    # plt.hist(np.asarray(estimators["B"])[:, 1])

    # xx = []
    # yy = defaultdict(list)
    # for t, dic in all_est.items():
        # xx.append(t)
        # for tp, vec in pop.preference_dict.items():
            # tr = ps.choice_prob_vec(vec, np.ones(2))
            # tse = np.mean([np.linalg.norm(tr - ee) ** 2 for ee in dic[tp]])
            # yy[tp].append(tse)





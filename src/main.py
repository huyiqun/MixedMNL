import os
import sys
from collections import Counter, defaultdict
import numpy as np
from scipy.optimize import minimize
from scipy.stats import entropy
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin, pairwise_distances_argmin_min
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

from logging_util import ColoredLog
from simulator import Sampler
from frankwolfe import FrankWolfe, SubRegionFrankWolfe
from plot import d3plot, compare_prob_est, d3plot_result

cmap = cm.get_cmap(name="Dark2")

def mle_loss(x, sim, ss):
    diff = [
            np.mean(v[ss], axis=0) - sim.ps.choice_prob_vec(x, sim.exp_price[t])
            for t, v in sim.data_hist.items()
           ]
    loss = np.sum([np.linalg.norm(d) ** 2 for d in diff])
    return loss

def run(ps, pop, sim, rand_price, seeds, T, result_dict):
    personal_cdf = sim.get_personal_cdf(T)
    sampler = Sampler(personal_cdf, sim.type_dict, sim.member, verbose=3)

    beta_set = []
    valid_seeds = []
    purity = []
    subsamples = {}
    for seed in seeds:
        tp = sim.type_dict[seed]
        _, ss, pure = sampler.create_sample(seed, 200, lambda x: max(1-15*x, 1e-4))
        # pp, ss = sampler.create_sample(np.random.randint(sim.num_cons), 200, lambda x: np.exp(-10*x))
        res = minimize(lambda x: mle_loss(x, sim, ss), np.random.rand(ps.num_feat), tol=1e-4)
        if res.success:
            valid_seeds.append(tp)
            subsamples[seed] = ss
            beta_set.append(res.x)
            purity.append(pure)

    estimators = defaultdict(list)
    for tp, b in zip(valid_seeds, beta_set):
        estimators[tp].append(sim.ps.choice_prob_vec(b, rand_price))

    bscopy = np.copy(beta_set)
    result_dict["valid_seeds"] = valid_seeds
    result_dict["subsamples"] = subsamples
    result_dict["beta_set"] = bscopy
    result_dict["estimators"] = estimators
    result_dict["purity"] = np.mean(purity)

    fw = FrankWolfe(sim, verbose=3)
    fw.run(max_iter=20, tol=1e-3)
    result_dict["fw"] = fw

    beta_set = np.copy(bscopy)
    srfw = SubRegionFrankWolfe(sim, verbose=3)
    srfw.run(beta_set, max_iter=20, tol=1e-3)
    result_dict["srfw"] = srfw

    md_fw = []
    md_srfw = []
    for _, p in enumerate(sim.exp_price):
        true_q = [ps.choice_prob_vec(b, p) for b in pop.preference_vec]
        fw_q = [ps.choice_prob_vec(b, p) for b in fw.active_beta]
        srfw_q = [ps.choice_prob_vec(b, p) for b in srfw.active_beta]

        md_fw.append(pairwise_distances_argmin_min(fw_q, true_q))
        md_srfw.append(pairwise_distances_argmin_min(srfw_q, true_q))

    result_dict["dist_fw"] = md_fw
    result_dict["dist_srfw"] = md_srfw

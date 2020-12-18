import os
import sys
import argparse
import shlex
import pickle
import time
import warnings
from pathlib import Path
from typing import NamedTuple
from shutil import rmtree
from collections import defaultdict, Counter

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin_min

try:
    project_root = str(Path(__file__).resolve().parent)
except:
    print("Try cd to project root.")
    project_root = str(Path.cwd())

sys.path.append(os.path.join(project_root, "src"))
# import importlib
# importlib.reload(logging_util)
# import logging_util
from src.logging_util import ColoredLog
from src.data_util import Population, Product_Set
from src.simulator import Simulator
from src.main import run
from src.plot import mdsplot, boxplot


parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", type=str, help="experiment save directory")
parser.add_argument("--num_cons", type=int, help="number of consumers")
parser.add_argument("--num_type", type=int, help="number of consumer types")
parser.add_argument("--num_prod", type=int, help="number of products")
parser.add_argument("--num_feat", type=int, help="number of features")
parser.add_argument("--sample_ratio", type=float, help="sample ratio")
parser.add_argument("--default", help="use toy data", action="store_true")
parser.add_argument("--guessK", type=int, help="predetermined K")
parser.add_argument("-v", "--verbose", action="count")

parser.set_defaults(
        num_cons=2000,
        num_type=5,
        num_prod=10,
        num_feat=10,
        sample_ratio=0.2,
        default=False,
        verbose=3,
        )

# arg_str = shlex.split("-v --num_type 5 --num_feat 10 --num_prod 10 --num_cons 2000")
# args = parser.parse_args(arg_str)
args = parser.parse_args()

VERBOSE = args.verbose
logger = ColoredLog(__name__, verbose=VERBOSE)
if args.default:
    exp_dir = os.path.join(project_root, "experiments", "default")
    if os.path.isdir(exp_dir):
        rmtree(exp_dir)
    os.makedirs(exp_dir)
    pop = Population.from_data("data/consumer.csv", verbose=VERBOSE)
    ps = Product_Set.from_data("data/product.csv", verbose=VERBOSE)
else:
    exp_name = f"K-{args.num_type}-M-{args.num_prod}-d-{args.num_feat}"
    exp_dir = os.path.join(project_root, "experiments", exp_name)
    pop = Population.from_data(os.path.join(exp_dir, "consumer.csv"), verbose=VERBOSE)
    ps = Product_Set.from_data(os.path.join(exp_dir, "product.csv"), verbose=VERBOSE)

N = args.num_cons
K = pop.num_type
M = ps.num_prod
d = ps.num_feat

T_max = 300
exp_price = np.ones((T_max, M))
rand_price = np.ones(M)

if not os.path.isdir(os.path.join(exp_dir, "em")):
    os.makedirs(os.path.join(exp_dir, "em"))

if os.path.isfile(os.path.join(exp_dir, "sim.pkl")):
    with open(os.path.join(exp_dir, "sim.pkl"), "rb") as f:
        sim = pickle.load(f)
else:
    sim = Simulator(ps, pop, num_cons=N, verbose=4)
    sim.run_experiments(exp_price)
    with open(os.path.join(exp_dir, "sim.pkl"), "wb") as f:
        pickle.dump(sim, f, pickle.HIGHEST_PROTOCOL)

p = np.ones(M)
b = np.ones(d)

# from sklearn.cluster import KMeans
# cluster_data = [np.asarray(personal_data[i]).flatten() for i in cid]
# km = KMeans(guessK)
# cl = km.fit_predict(cluster_data)

# times = np.arange(5, 300, 5)
# T = 20
cid = range(2000)
guessK = args.guessK
logger.info(f"Running EM for K={guessK}")

if os.path.isfile(os.path.join(exp_dir, "em", f"em_{guessK}.pkl")):
    with open(os.path.join(exp_dir, "em", f"em_{guessK}.pkl"), "rb") as f:
        em_res = pickle.load(f)
    alpha_Ts = em_res[guessK]["alpha_Ts"]
    beta_Ts = em_res[guessK]["beta_Ts"]
    ptime_Ts = em_res[guessK]["ptime_Ts"]
    dist_Ts = em_res[guessK]["dist_Ts"]
else:
    em_res = defaultdict(dict)
    alpha_Ts = {}
    beta_Ts = {}
    ptime_Ts = {}
    dist_Ts = {}

# alpha_Ts
# em_res[3]["alpha_Ts"].keys()
# em_res[3]["dist_Ts"].keys()
# [len(v[0]) for v in em_res[5]["alpha_Ts"].values()]
# [len(v) for v in em_res[5]["alpha_Ts"].values()]
# [len(v[0]) for v in alpha_Ts.values()]
# [len(v) for v in alpha_Ts.values()]
# alpha_Ts.keys()
# guessK = 5
# with open(os.path.join(exp_dir, f"em_{guessK}.pkl"), "rb") as f:
    # em_res = pickle.load(f)


logger.info(f"existing Ts: {em_res[guessK]['alpha_Ts'].keys()}")
TT = [tt for tt in np.arange(5,155,5) if tt not in em_res[guessK]["alpha_Ts"].keys()]
logger.info(f"Will run: {TT}")

# for T in times:
for T in TT:
    print(T)
    proc_time = []
    personal_data = {i: np.asarray([sim.data_hist[t][i] for t in range(T)]) for i in cid}
    decision = np.asarray([[np.argwhere(personal_data[i][t]!=0)[0][0] for t in range(T)] for i in cid])
    # decision.shape
    # true_alpha = [sim.type_dict[kk] for kk in cid]
    # [a/len(list(cid)) for a in Counter(true_alpha).values()]

    def obj(b, data):
        diff = [np.mean(data, axis=0) - ps.choice_prob_vec(b, p)]
        return np.sum([np.linalg.norm(d) ** 2 for d in diff])

    # check_A = [personal_data[i][t] for i in sim.member["A"] for t in range(T)]
    # res = minimize(lambda b: obj(b, check_A), np.random.rand(d), tol=1e-3)
    alpha = [1/guessK] * guessK

    init_cl = {}
    ind = list(cid)
    # np.random.shuffle(ind)
    ending = [int(np.round(len(ind) * a)) for a in np.cumsum(alpha)]

    init_cl[0] = ind[0:ending[0]]
    for j in range(1, guessK):
        init_cl[j] = ind[ending[j-1]:ending[j]]

    ss = time.time()
    beta_est = {}
    for k in range(guessK):
        data_k = [personal_data[i][t] for i in init_cl[k] for t in range(T)]
        res = minimize(lambda b: obj(b, data_k), np.random.rand(d), tol=1e-3)
        beta_est[k] = res.x
    proc_time.append(time.time() - ss)

    # for k in range(guessK):
        # plt.plot(ps.choice_prob_vec(beta_est[k], p), label=f"est {k}")
    # for k in range(K):
        # plt.plot(ps.choice_prob_vec(pop.preference_vec[k], p), '--', label=f"true {k}")
    # plt.legend()

    converged = False
    iter = 0
    max_iter = 20
    data = np.asarray([np.transpose(v) for v in personal_data.values()])
    # data.shape
    alpha_prog = [alpha]
    beta_prog = [list(beta_est.values())]
    while not converged and iter < max_iter:
        ss = time.time()
        h_numerator = [
                [
                    alpha[k] * np.prod([
                        ps.choice_prob(beta_est[k], p, decision[i][t])
                        for t in range(T)
                    ])
                    for k in range(guessK)
                ]
                for i in cid
            ]
        h = [[hh/np.sum(h_numerator[i]) for hh in h_numerator[i]] for i in cid]

        h_trans = np.transpose(h)

        alpha = [np.sum(h_trans[k]) / np.sum(h_trans) for k in range(guessK)]
        alpha_prog.append(alpha)

        beta_est = {}
        for k in range(guessK):
            # check_A=np.argwhere(cl==1).flatten()
            # check_A = [i for i in range(300) if sim.type_dict[i] == "A"]
            # check_A
            # Counter([sim.type_dict[ii] for ii in check_A])
            # h = [[1,0,0,0,0] for i in cid]
            # k = 4

            def func(b, k):
                log_original = -np.log(np.dot(ps.choice_prob_vec(b, p), data))
                vf = np.vectorize(lambda x: min(1e3, x))
                stable_log = vf(log_original)
                stable_log.shape
                obj = h_trans[k][:, np.newaxis] * stable_log
                return np.mean(obj)


            res = minimize(lambda x: func(x, k), np.ones(d), tol=1e-4)

            beta_est[k] = res.x

        beta_prog.append(list(beta_est.values()))
        proc_time.append(time.time() - ss)

        gap = [np.linalg.norm(ps.choice_prob_vec(beta_prog[-1][k], p) - ps.choice_prob_vec(beta_prog[-2][k], p))/(M+1) for k in range(guessK)]
        print(f"iter: {iter}, diff: {np.mean(gap)}")
        if np.mean(gap) < 1e-3 and iter > 1:
            converged = True
        iter += 1

    alpha_Ts[T] = alpha_prog
    beta_Ts[T] = beta_prog
    ptime_Ts[T] = proc_time

    min_dist = np.mean(pairwise_distances_argmin_min([ps.choice_prob_vec(bb, p) for bb in beta_est.values()], [ps.choice_prob_vec(bb, p) for bb in pop.preference_vec])[1])
    dist_Ts[T] = min_dist

    em_res[guessK]["alpha_Ts"] = alpha_Ts
    em_res[guessK]["beta_Ts"] = beta_Ts
    em_res[guessK]["ptime_Ts"] = ptime_Ts
    em_res[guessK]["dist_Ts"] = dist_Ts

    with open(os.path.join(exp_dir, "em", f"em_{guessK}.pkl"), "wb") as f:
        pickle.dump(em_res, f, pickle.HIGHEST_PROTOCOL)

    print([len(v[0]) for v in em_res[guessK]["alpha_Ts"].values()])
    print([len(v) for v in em_res[guessK]["alpha_Ts"].values()])
# with open(os.path.join(exp_dir, "em", f"em_{guessK}.pkl"), "rb") as f:
    # em_res = pickle.load(f)


# for bb in beta_est.values():
    # plt.plot(np.arange(11), ps.choice_prob_vec(bb, p), '--')
# for bb in pop.preference_vec:
    # plt.plot(np.arange(11), ps.choice_prob_vec(bb, p))
# for bb in res["srfw"].active_beta:
    # plt.plot(np.arange(11), ps.choice_prob_vec(bb, p), '-.')

# plt.savefig(os.path.join(exp_dir, "temp.png"))

# len(res["dist_srfw"])
# dist_em_3 = [em_res_all[3]["dist_Ts"][tt] for tt in range(5,150,5)]
# dist_em_4 = [em_res_all[4]["dist_Ts"][tt] for tt in range(5,150,5)]
# # plt.plot(np.arange(5,150,5), min_dist_fw[:29], label="FW")
# plt.plot(np.arange(5,150,5), min_dist_srfw[:29], label="SRFW")
# plt.plot(np.arange(5,150,5), dist_em_3, label="EM (K=3)")
# plt.plot(np.arange(5,150,5), dist_em_4, label="EM (K=4)")
# plt.legend()
# plt.savefig(os.path.join(exp_dir, "temp.png"))


# func = lambda b: np.mean([
        # [
            # h[i][k] * min(1e3, -np.log(ps.choice_prob(b, p, decision[i][t])))
            # for t in range(T)
        # ]
        # for i in cid
    # ])

# jac = lambda b: np.mean([
    # np.sum(
        # (personal_data[i][t]-ps.choice_prob_vec(b, p))[1:, np.newaxis] * np.asarray(ps.features) * (-1),
        # axis=0)
    # for i in check_A for t in range(T)
    # ], axis=0)

# def jac(b):
    # orig = np.mean([
        # np.sum(
            # (personal_data[i][t]-ps.choice_prob_vec(b, p))[1:, np.newaxis] * np.asarray(ps.features),
            # axis=0)
        # for i in cid for t in range(T)
        # ], axis=0)
    # return orig
    # while np.max(orig) > 10:
        # orig *= 1/2
    # return [bb/sum(orig) for bb in orig]

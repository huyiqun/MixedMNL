import os
import sys
import argparse
import shlex
import pickle
from pathlib import Path
from typing import NamedTuple
from shutil import rmtree
from collections import defaultdict, Counter

import numpy as np
import matplotlib.pyplot as plt

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
parser.add_argument("-v", "--verbose", action="count")

parser.set_defaults(
        exp_dir="K-2-M-2-d-1",
        num_cons=2000,
        num_type=2,
        num_prod=2,
        num_feat=1,
        sample_ratio=0.2,
        default=False,
        verbose=3,
        )

arg_str = shlex.split("-v --num_type 5 --num_feat 10 --num_prod 10 --num_cons 2000")
args = parser.parse_args(arg_str)
# args = parser.parse_args()

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

    if os.path.isdir(exp_dir):
        pop = Population.from_data(os.path.join(exp_dir, "consumer.csv"), verbose=VERBOSE)
        ps = Product_Set.from_data(os.path.join(exp_dir, "product.csv"), verbose=VERBOSE)
    else:
        pop = Population.from_sim(num_type=args.num_type, num_feat=args.num_feat, verbose=VERBOSE)
        min_alpha = min(pop.alpha)
        tries = 0
        while min_alpha < 0.1 and tries < 50:
            pop = Population.from_sim(num_type=args.num_type, num_feat=args.num_feat, verbose=VERBOSE)
            tries += 1
        # min_alpha
        if min_alpha < 0.1:
            logger.warning("Pop simulation not successful, try again.")
            sys.exit()

        ps = Product_Set.from_sim(num_prod=args.num_prod, num_feat=args.num_feat, verbose=VERBOSE)
        min_prob = np.min([ps.choice_prob_vec(p, np.ones(args.num_prod)) for p in pop.preference_vec])
        tries = 0
        while min_prob < 5e-3 and tries < 50:
            ps = Product_Set.from_sim(num_prod=args.num_prod, num_feat=args.num_feat, verbose=VERBOSE)
            min_prob = np.min([ps.choice_prob_vec(p, np.ones(args.num_prod)) for p in pop.preference_vec])
            tries += 1
        # min_prob
        if min_prob < 5e-3:
            logger.warning("PS simulation not successful, try again.")
            sys.exit()

        os.makedirs(exp_dir)
        pop.save(os.path.join(exp_dir, "consumer.csv"))
        ps.save(os.path.join(exp_dir, "product.csv"))


N = args.num_cons
K = pop.num_type
M = ps.num_prod
d = ps.num_feat

T_max = 300
exp_price = np.ones((T_max, M))
rand_price = np.ones(M)
# exp_price = np.random.uniform(low=0.5, high=1.5, size=(T, M))
# rand_price = [round(p, 2) for p in np.random.uniform(low=0.5, high=3, size=ps.num_prod)]

if os.path.isfile(os.path.join(exp_dir, "sim.pkl")):
    with open(os.path.join(exp_dir, "sim.pkl"), "rb") as f:
        sim = pickle.load(f)
else:
    sim = Simulator(ps, pop, num_cons=N, verbose=4)
    sim.run_experiments(exp_price)
    with open(os.path.join(exp_dir, "sim.pkl"), "wb") as f:
        pickle.dump(sim, f, pickle.HIGHEST_PROTOCOL)

if os.path.isfile(os.path.join(exp_dir, "seeds.pkl")):
    with open(os.path.join(exp_dir, "seeds.pkl"), "rb") as f:
        seeds = pickle.load(f)
    print(Counter([sim.type_dict[s] for s in seeds]))
else:
    seeds = np.random.choice(N, size=75, replace=False)
    print(Counter([sim.type_dict[s] for s in seeds]))

    with open(os.path.join(exp_dir, "seeds.pkl"), "wb") as f:
        pickle.dump(seeds, f, pickle.HIGHEST_PROTOCOL)


# result_dict = defaultdict(dict)

TT = np.arange(280, 350, 5)
for T in TT:
    result_dict = {}
    run(ps, pop, sim, rand_price, seeds, T, result_dict)
    with open(os.path.join(exp_dir, f"result_{T}.pkl"), "wb") as f:
        pickle.dump(result_dict, f, pickle.HIGHEST_PROTOCOL)

# analysis
times = []

min_dist_fw = []
min_dist_srfw = []
opt_time = []
for T in np.arange(5, 300, 5):
    if os.path.isfile(os.path.join(exp_dir, f"result_{T}.pkl")):
        times.append(T)
        with open(os.path.join(exp_dir, f"result_{T}.pkl"), "rb") as f:
            res = pickle.load(f)
        dist_fw = np.mean([np.mean(md[1]) for md in res["dist_fw"]])
        # len(res["dist_fw"])
        dist_srfw = np.mean([np.mean(md[1]) for md in res["dist_srfw"]])

        min_dist_fw.append(dist_fw)
        min_dist_srfw.append(dist_srfw)

        opt_time.append((np.mean(res["fw"].q_opt_time), np.mean(res["srfw"].q_opt_time)))

plt.plot(times, min_dist_fw)
plt.plot(times, min_dist_srfw)

import seaborn as sns
sns.lineplot(x=times, y=min_dist_fw)
sns.lineplot(x=times, y=min_dist_srfw)

T = 300
with open(os.path.join(exp_dir, f"result_{T}.pkl"), "rb") as f:
    res = pickle.load(f)
true_q = [ps.choice_prob_vec(p, rand_price) for p in pop.preference_vec]
fw_q = [ps.choice_prob_vec(p, rand_price) for p in res["fw"].active_beta]
srfw_q = [ps.choice_prob_vec(p, rand_price) for p in res["srfw"].active_beta]

boxplot(res["estimators"], pop, ps, rand_price, Counter(res["valid_seeds"]), ctype="A")
boxplot(res["estimators"], pop, ps, rand_price, Counter(res["valid_seeds"]), ctype="all")
mdsplot(true_q, fw_q, srfw_q)

# plot 1
# d3plot(ps, pop, rand_price, beta_set, exp_dir, value_range=20, granularity=10, single=True)

# plot 2
# fig, ax = plt.subplots(len(estimators), 1, sharex=True, figsize=(6, 2*len(estimators)+1))
# for i, t in enumerate(estimators.keys()):
    # ground_truth = ps.choice_prob_vec(pop.preference_dict[t], rand_price)
    # compare_prob_est(estimators, t, ground_truth, ps, ax[i])
    # plt.tight_layout()
# lgd = plt.legend(bbox_to_anchor=(1,1.1), loc="upper left")
# plt.savefig(os.path.join(exp_dir, "type.png"), bbox_extra_artists=(lgd,), bbox_inches='tight')

# plot 3
# d3plot_result(true_q, fw_q, srfw_q, rand_price, exp_dir)




# pop.show_info()
# np.average(true_q, axis=0, weights=pop.alpha)
# np.average([ps.choice_prob_vec(b, rand_price) for b in srfw.active_beta], axis=0, weights=srfw.active_alpha)
# np.average([ps.choice_prob_vec(b, rand_price) for b in fw.active_beta], axis=0, weights=fw.active_alpha)
# logger.info(true_q, header=np.arange(len(true_q[0])))
# logger.info([ps.choice_prob_vec(b, rand_price) for b in srfw.active_beta], header=np.arange(len(true_q[0])))
# result_dict[T]["fw"].q_opt_time
# result_dict[T]["dist_fw"]





# plt.plot(np.arange(10, 170, 10), save_res["fw"], label="FW")
# plt.plot(np.arange(10, 170, 10), save_res["srfw"], label="SRFW")
# plt.xlabel("Experiment")
# plt.ylabel("Average Dist to Closest True Choice Prob")
# plt.legend
# plt.savefig(os.path.join(exp_dir, "min_dist.png"))

# plt.plot(np.arange(10, 170, 10), purity_res)
# plt.xlabel("Experiment")
# plt.ylabel("Average Subsample Purity")
# plt.savefig(os.path.join(exp_dir, "purity.png"))

# plt.plot(range(T), save_res["fw"], label="FW")
# plt.plot(range(T), save_res["srfw"], label="SRFW")
# plt.xlabel("Experiment")
# plt.ylabel("Average Min Dist to Closest True q")
# plt.legend()
# plt.savefig(os.path.join(exp_dir, "dist.png"))
# for t, e in yy.items():
    # plt.plot(xx, e, label=f"Type {t}")
    # plt.xlabel("Number of Experiments")
    # plt.ylabel("MSE for MNL Estimation")
# plt.legend()


# import yaml
# with open(args.param_file, "r") as f:
    # params = yaml.safe_load(f)

# class Env(NamedTuple):
    # N: int
    # K: int
    # M: int
    # d: int
    # simulate: bool
    # consumer_data: str
    # product_data: str
    # save_data: bool

# env = Env(**params)

# parser.add_argument(
    # "--exp_per_cyc",
    # type=int,
    # help="number of independent experiments per cycle",
    # default=5,
# )
# parser.add_argument(
    # "--sim_per_exp", type=int, help="number of simulations per experiment", default=10
# )
# parser.add_argument(
    # "--trans_per_exp",
    # type=int,
    # help="number of transactions per experiment",
    # default=1000,
# )

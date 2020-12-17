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



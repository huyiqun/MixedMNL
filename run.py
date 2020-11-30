import os
import sys
import argparse
import shlex
from pathlib import Path
from typing import NamedTuple
from shutil import rmtree

import matplotlib.pyplot as plt

try:
    project_root = str(Path(__file__).resolve().parent)
except:
    print("Try cd to project root.")
    project_root = str(Path.cwd())

sys.path.append(os.path.join(project_root, "src"))
# import importlib
# importlib.reload(simulate_util)
# import simulate_util
from logging_util import ColoredLog
from simulate_util import Population, Product_Set
from srfw import SubRegionFrankWolfe

parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", type=str, help="experiment save directory")
parser.add_argument("--num_cons", type=int, help="number of consumers")
parser.add_argument("--num_type", type=int, help="number of consumer types")
parser.add_argument("--num_prod", type=int, help="number of products")
parser.add_argument("--num_feat", type=int, help="number of features")
parser.add_argument("--sample_ratio", type=float, help="sample ratio")
parser.add_argument("--product_data", type=str, help="product data")
parser.add_argument("--consumer_data", type=str, help="consumer data")
parser.add_argument("--simulate", help="simulate data", action="store_true")
parser.add_argument("--save_data", help="save sim data", action="store_true")
parser.add_argument("-v", "--verbose", action="count")

parser.set_defaults(
        exp_dir="default",
        num_cons=1000,
        num_type=2,
        num_prod=2,
        num_feat=1,
        sample_ratio=0.2,
        product_data="data/product.csv",
        consumer_data="data/consumer.csv",
        simulate=False,
        save_data=False,
        verbose=3,
        )

# args = parser.parse_args()
arg_str = shlex.split("-v")
args = parser.parse_args(arg_str)

VERBOSE = args.verbose
if args.simulate:
    N = args.num_cons
    K = args.num_type
    M = args.num_prod
    d = args.num_feat
    exp_name = f"K-{K}-M-{M}-d-{d}"
    exp_dir = os.path.join(project_root, "experiments", exp_name)
    pop = Population.from_sim(num_type=K, num_feat=d, verbose=VERBOSE)
    ps = Product_Set.from_sim(num_prod=M, num_feat=d, verbose=VERBOSE)
    if args.save_data:
        if os.path.isdir(exp_dir):
            rmtree(exp_dir)
        os.mkdir(exp_dir)
        pop.save(os.path.join(exp_dir, "consumer.csv"))
        ps.save(os.path.join(exp_dir, "product.csv"))
else:
    exp_dir = os.path.join(project_root, "experiments", args.exp_dir)
    if os.path.isdir(exp_dir):
        rmtree(exp_dir)
    os.mkdir(exp_dir)
    pop = Population.from_data(args.consumer_data, verbose=VERBOSE)
    ps = Product_Set.from_data(args.product_data, verbose=VERBOSE)
    N = args.num_cons
    K = pop.num_type
    M = ps.num_prod
    d = ps.num_feat

pop.__dict__
vars(args)
res = SubRegionFrankWolfe()



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

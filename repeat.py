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


# parser = argparse.ArgumentParser()
# parser.add_argument("--exp_dir", type=str, help="experiment save directory")
# parser.add_argument("--num_cons", type=int, help="number of consumers")
# parser.add_argument("--num_type", type=int, help="number of consumer types")
# parser.add_argument("--num_prod", type=int, help="number of products")
# parser.add_argument("--num_feat", type=int, help="number of features")
# parser.add_argument("--sample_ratio", type=float, help="sample ratio")
# parser.add_argument("--default", help="use toy data", action="store_true")
# parser.add_argument("-v", "--verbose", action="count")

# parser.set_defaults(
        # exp_dir="K-2-M-2-d-1",
        # num_cons=2000,
        # num_type=2,
        # num_prod=2,
        # num_feat=1,
        # sample_ratio=0.2,
        # default=False,
        # verbose=3,
        # )

exp_dir = os.path.join(project_root, "experiments", "compare")
# exp_name_list = []
# K = 4
# T_max = 50
# for i in range(30, 60):
    # if i % 10 == 0:
        # K += 1
    # j = i % 10
    # print(i,j,K)

    # arg_str = shlex.split(f"-v --num_type {K} --num_feat 10 --num_prod 10 --num_cons 1000")
    # args = parser.parse_args(arg_str)

    # N = args.num_cons
    # M = args.num_prod
    # d = args.num_feat
    # VERBOSE = args.verbose

    # exp_price = np.ones((T_max, M))

    # exp_dir = os.path.join(project_root, "experiments", "compare")
    # exp_name = f"K-{args.num_type}"
    # # exp_name_list.append(exp_name + f"-{j}")

    # pop = Population.from_sim(num_type=args.num_type, num_feat=args.num_feat, verbose=VERBOSE)
    # print(min(pop.alpha))

    # ps = Product_Set.from_sim(num_prod=args.num_prod, num_feat=args.num_feat, verbose=VERBOSE)
    # min_prob = np.min([ps.choice_prob_vec(p, np.ones(args.num_prod)) for p in pop.preference_vec])
    # tries = 0
    # while min_prob < 1e-2:
        # ps = Product_Set.from_sim(num_prod=args.num_prod, num_feat=args.num_feat, verbose=VERBOSE)
        # min_prob = np.min([ps.choice_prob_vec(p, np.ones(args.num_prod)) for p in pop.preference_vec])
        # tries += 1
        # if tries % 50 == 0:
            # print("pop tries: ", tries)
    # print(min_prob)

    # pop.save(os.path.join(exp_dir, exp_name + f"-cons-{j}.csv"))
    # ps.save(os.path.join(exp_dir, exp_name + f"-prod-{j}.csv"))

    # sim = Simulator(ps, pop, num_cons=N, verbose=3)
    # sim.run_experiments(exp_price)
    # with open(os.path.join(exp_dir, exp_name + f"-sim-{j}.pkl"), "wb") as f:
        # pickle.dump(sim, f, pickle.HIGHEST_PROTOCOL)

    # seeds = np.random.choice(N, size=75, replace=False)
    # with open(os.path.join(exp_dir, exp_name + f"-seeds-{j}.pkl"), "wb") as f:
        # pickle.dump(seeds, f, pickle.HIGHEST_PROTOCOL)

    # exp_name_list.append((exp_name, f"{j}"))

# with open(os.path.join(exp_dir, "exp_name_list.txt"), "w") as f:
    # f.write("\n".join(list(map(",".join, exp_name_list))))

from em_util import em_run, log_likelihood

with open(os.path.join(exp_dir, "exp_name_list.txt"), "r") as f:
    exp_name_list = list(map(lambda x:x.strip(), f.readlines()))

parser = argparse.ArgumentParser()
parser.add_argument("--targetK", type=int, help="target K")
args = parser.parse_args()


for s in exp_name_list:
    # s = exp_name_list[0]
    f1, f2 = s.split(",")
    targetK = args.targetK
    # targetK = 5
    res_path = os.path.join(exp_dir, f1, "result.pkl")
    if int(f1.split('-')[1]) == targetK:
        if os.path.exists(res_path):
            with open(res_path, "rb") as f:
                res_dict = pickle.load(f)
        else:
            res_dict = defaultdict(dict)

        print(f1, f2)
        exp_folder = os.path.join(exp_dir, f1, f2)
        ps = Product_Set.from_data(os.path.join(exp_folder, "product.csv"))
        pop = Population.from_data(os.path.join(exp_folder, "consumer.csv"))
        with open(os.path.join(exp_folder, "sim.pkl"), "rb") as f:
            sim = pickle.load(f)
        with open(os.path.join(exp_folder, "seeds.pkl"), "rb") as f:
            seeds = pickle.load(f)
        print(Counter([sim.type_dict[s] for s in seeds]))
        T = 50
        d = ps.num_feat
        N = sim.num_cons

        rand_price = sim.exp_price[0]
        exp_name = "-".join([f1, f2])
        if exp_name + "-alg" not in res_dict:
            run(ps, pop, sim, rand_price, seeds, T, res_dict[exp_name + "-alg"])
            with open(res_path, "wb") as f:
                pickle.dump(res_dict, f, pickle.HIGHEST_PROTOCOL)
        else:
            print(exp_name + f"-alg already exists in result, skipping")

        # for guessK in range(max(2, targetK-4), min(11, targetK+4)):
        for guessK in range(1, 13):
            if exp_name + f"-em-{guessK}" not in res_dict:
                alpha, beta, ptime, dist = em_run(guessK, T, sim)
                res_dict[exp_name + f"-em-{guessK}"]["alpha"] = alpha
                res_dict[exp_name + f"-em-{guessK}"]["beta"] = beta
                res_dict[exp_name + f"-em-{guessK}"]["ptime"] = ptime
                res_dict[exp_name + f"-em-{guessK}"]["dist"] = dist

                LL = log_likelihood(beta[-1], alpha[-1], sim)
                num_var_k = (d + 1) * guessK - 1
                num_obs = T * N
                aic = 2 * num_var_k - 2 * LL
                bic = np.log(num_obs) * num_var_k - 2 * LL

                res_dict[exp_name + f"-em-{guessK}"]["aic"] = aic
                res_dict[exp_name + f"-em-{guessK}"]["bic"] = bic
                print(targetK, guessK, bic)

                with open(res_path, "wb") as f:
                    pickle.dump(res_dict, f, pickle.HIGHEST_PROTOCOL)
            else:
                print(exp_name + f"-em-{guessK} already exists in result, skipping")

# aic_dict = defaultdict(list)
# bic_dict = defaultdict(list)
# for s in exp_name_list[:-10]:
    # # s = exp_name_list[20]
    # f1, f2 = s.split(",")
    # # targetK = args.targetK
    # targetK = 5
    # currentK = int(f1.split('-')[1])
    # exp_folder = os.path.join(exp_dir, f1, f2)
    # exp_name = "-".join([f1, f2])
    # with open(os.path.join(exp_folder, "sim.pkl"), "rb") as f:
        # sim = pickle.load(f)
    # ps = sim.ps
    # pop = sim.pop

    # res_path = os.path.join(exp_dir, f1, "result.pkl")
    # with open(res_path, "rb") as f:
        # res_dict = pickle.load(f)
    # print(res_dict.keys())
    # for k, v in res_dict.items():
        # if "em" in k and f2==k[4]:
            # gK = int(k.split('-')[-1])
            # aic_dict[(currentK, int(f2), gK)] = v["aic"]
            # bic_dict[(currentK, int(f2), gK)] = v["bic"]

    # res_dict.keys()
    # guessK=5
    # beta = res_dict[exp_name+f"-em-{guessK}"]["beta"][-1]
    # alpha = res_dict[exp_name+f"-em-{guessK}"]["alpha"][-1]
    # np.asarray(beta).shape
    # np.asarray(alpha).shape

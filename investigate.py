import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.style.available
# analysis
times = []

min_dist_fw = []
min_dist_srfw = []
opt_time = []
purity = []
T = 10
for T in np.arange(5, 155, 5):
    if os.path.isfile(os.path.join(exp_dir, f"result_{T}.pkl")):
        times.append(T)
        with open(os.path.join(exp_dir, f"result_{T}.pkl"), "rb") as f:
            res = pickle.load(f)
        dist_fw = np.mean([np.mean(md[1]) for md in res["dist_fw"]])
        # len(res["dist_fw"])
        dist_info = pairwise_distances_argmin_min([ps.choice_prob_vec(bb, p) for bb in res["srfw"].active_beta], [ps.choice_prob_vec(bb, p) for bb in pop.preference_vec])
        dist_srfw = np.average(dist_info[1], weights=res["srfw"].active_alpha)
        # res["srfw"].active_beta
        # res["srfw"].active_alpha
        # dist_srfw = np.mean([np.mean(md[1]) for md in res["dist_srfw"]])

        min_dist_fw.append(dist_fw)
        min_dist_srfw.append(dist_srfw)

        opt_time.append((np.mean(res["fw"].q_opt_time), np.mean(res["srfw"].q_opt_time)))
        purity.append(res["purity"])


from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances, pairwise_distances_argmin
guessK = 5
em_dict = {}
for guessK in [3,4,5,6]:
    if os.path.isfile(os.path.join(exp_dir, "em", f"em_{guessK}.pkl")):
        with open(os.path.join(exp_dir, "em", f"em_{guessK}.pkl"), "rb") as f:
            em_res = pickle.load(f)

    time_range = list(em_res.keys())
    rec = []
    for tt in time_range:
        tt = 90
        min_dist = np.average(pairwise_distances_argmin_min([ps.choice_prob_vec(bb, p) for bb in em_res[tt]["beta"][0][-1]], [ps.choice_prob_vec(bb, p) for bb in pop.preference_vec])[1], weights=em_res[tt]["alpha"][0][-1])
        rec.append(min_dist)
    # rec = [np.mean(em_res[tt]["dist"]) for tt in time_range]
    em_dict[guessK] = (time_range, rec)
    pop.alpha

with plt.style.context("seaborn-paper"):
    for k,v in em_dict.items():
        plt.plot(v[0], v[1], label=f"EM (K={k})")

    # plt.plot(times, min_dist_fw, label="FW")
    plt.plot(times, min_dist_srfw, label="SRFW")
    plt.xlabel("Number of transactions (T)")
    plt.ylabel(r"Average distance to closest true $q$")
    plt.legend()

plt.show()
plt.clf()

np.mean([t[0] for t in opt_time])
np.mean([t[1] for t in opt_time])

plt.plot(times, purity)
plt.xlabel("Number of transactions (T)")
plt.ylabel(r"Average Subsample Purity")

import seaborn as sns
import string


T = 20
T = 50
T = 300
with open(os.path.join(exp_dir, f"result_{T}.pkl"), "rb") as f:
    res = pickle.load(f)
true_q = [ps.choice_prob_vec(p, rand_price) for p in pop.preference_vec]
fw_q = [ps.choice_prob_vec(p, rand_price) for p in res["fw"].active_beta]
srfw_q = [ps.choice_prob_vec(p, rand_price) for p in res["srfw"].active_beta]

i=5
[pairwise_distances_argmin_min([em_q[i]], em_q[:i]+em_q[i+1:]) for i in range(6)]

boxplot(res["estimators"], pop, ps, rand_price, Counter(res["valid_seeds"]), ctype="A")
boxplot(res["estimators"], pop, ps, rand_price, Counter(res["valid_seeds"]), ctype="all")
em_q = [ps.choice_prob_vec(bb, p) for bb in em_res[150]["beta"][0][-1]]
mdsplot(true_q, em_q, srfw_q)

# ecdf plots
fig = plt.figure(figsize=(16,12))
# len(res["srfw"].active_beta)
for i in range(min(12, len(res["srfw"].active_beta))):
    ax = fig.add_subplot(3, 4, i+1)
    ind = res["dist_srfw"][0][0][i]
    sns.ecdfplot(np.cumsum(true_q[ind]), alpha=0.25, ax=ax)
    sns.ecdfplot(np.cumsum(srfw_q[i]), alpha=0.95, ax=ax)
    ax.set_title(f"Cloest type: {string.ascii_uppercase[ind]}")
    ax.set_xticks(np.arange(0,1,0.1))
    ax.set_xticklabels(ps.pid_off, rotation=45)
plt.tight_layout()

fig = plt.figure(figsize=(16,12))
# len(res["srfw"].active_beta)
for i in range(9):
    ax = fig.add_subplot(3, 4, i+1)
    ind = res["dist_fw"][0][0][i]
    sns.ecdfplot(np.cumsum(true_q[ind]), alpha=0.25, ax=ax)
    sns.ecdfplot(np.cumsum(fw_q[i]), alpha=0.95, ax=ax)
    ax.set_title(f"Cloest type: {string.ascii_uppercase[ind]}")
    ax.set_xticks(np.arange(0,1,0.1))
    ax.set_xticklabels(ps.pid_off, rotation=45)
plt.tight_layout()

# simulate data assumption feasibility
comp = []
alpha1 = np.arange(0.01, 0.20, 0.02)
for a1 in alpha1:
    new = True
    while new:
        init = np.sort(np.random.uniform(0,1,5))
        alphas = init/sum(init)
        diff = alphas[0] - a1
        alphas[0] = a1
        alphas[1:] += diff / 4
        if min(alphas) == a1:
            new = False
            comp.append(alphas)

compcumsum = [np.hstack((0, np.cumsum(c))) for c in comp]
exp = defaultdict(lambda: defaultdict(list))
for j, a1 in enumerate(alpha1):
    for NN in np.arange(10, 500, 10):
        for _ in range(100):
            # j=0
            # a1=0.05
            # NN=10
            picked = [[compcumsum[j][i] <= x < compcumsum[j][i+1] for i in range(5)] for x in np.random.rand(NN)]
            agged = [v.any() for  v in np.transpose(picked)]

            if np.asarray(agged).all():
                exp[a1][NN].append(1)
            else:
                exp[a1][NN].append(0)
s = [[np.mean(vv) for vv in v.values()] for a, v in exp.items()]
for i, ss in enumerate(s):
    plt.plot(np.hstack((0, np.arange(10,500,10))), [0] + ss, '--', label=f"{round(alpha1[i], 2)}")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title=r"mmp $\alpha_1$")
plt.ylim(0, 1.1)
plt.xlim(0, 500)
plt.xlabel("Number of Consumers Present in Data")
plt.ylabel("\"Cover All\" Simulated Probability")


# plot cover all's with breakage in axis
aa1 = np.arange(0.01, 0.20, 0.02)
K = 5
x = np.arange(1, 1500)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios":[10,1]})
fig.subplots_adjust(hspace=0.05)
# fig, ax = plt.subplots(1, 1)
for a1 in aa1:
    y1 = [max(0, 1- K * np.exp(-2*a1 ** 2*xx)) for xx in x]
    ax1.plot(x, y1, '--', label=f"{a1:.2f}")
    # ax2.plot(x, y1, '--', label=f"{a1:.2f}")
ax1.set_ylim((0.45, 1.05))
ax2.set_ylim((0, 0))
ax1.spines["bottom"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax1.tick_params(labeltop=False, bottom=False)
ax2.xaxis.tick_bottom()
ax1.set_yticks(np.arange(0.5, 1.01, 0.1))
ax1.set_yticklabels([str(round(v, 1)) for v in np.arange(0.5, 1.01, 0.1)])
ax2.set_yticks([0])
ax2.set_yticklabels([0])
ax1.set_ylabel("\" Cover All\" Lower Bound")
ax2.set_xlabel("Numer of Consumers Present in Data")
ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title=r"mmp $\alpha_1$")
d = .5
kw = dict(marker=[(-1,-d), (1,d)], markersize=12, linestyle="none", color="k", mec="k", mew=1, clip_on=False)
ax1.plot([0,1],[0, 0], transform=ax1.transAxes, **kw)
ax2.plot([0,1],[0.95,0.95], transform=ax2.transAxes, **kw)

aa1 = np.arange(0.05, 0.20, 0.02)
K = 5
x = np.arange(1, 100)
for a1 in aa1:
    y2 = [max(0, 1-(1-a1) ** xx) for xx in x]
    plt.plot(x, y2, '-.', label=f"{a1:.2f}")
plt.ylim((0,1.1))
plt.ylabel("\" Cover All\" Lower Bound")
plt.xlabel("Numer of Subsample Seeds")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title=r"mmp $\alpha_1$")

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios":[10,1]})
fig.subplots_adjust(hspace=0.05)
for a1 in aa1:
    y2 = [max(0, 1-(1-a1) ** xx) for xx in x]
    ax1.plot(x, y2, '-.', label=f"{a1:.2f}")
    # ax2.plot(x, y2, '-.', label=f"{a1:.2f}")
ax1.set_ylim((0.45, 1.05))
ax2.set_ylim((0, 0))
ax1.spines["bottom"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax1.tick_params(labeltop=False, bottom=False)
ax2.xaxis.tick_bottom()
ax1.set_yticks(np.arange(0.5, 1.01, 0.1))
ax1.set_yticklabels([str(round(v, 1)) for v in np.arange(0.5, 1.01, 0.1)])
ax2.set_yticks([0])
ax2.set_yticklabels([0])
ax1.set_ylabel("\" Cover All\" Lower Bound")
ax2.set_xlabel("Numer of Subsample Seeds")
ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title=r"mmp $\alpha_1$")
d = .5
kw = dict(marker=[(-1,-d), (1,d)], markersize=12, linestyle="none", color="k", mec="k", mew=1, clip_on=False)
ax1.plot([0,1],[0, 0], transform=ax1.transAxes, **kw)
ax2.plot([0,1],[0.95,0.95], transform=ax2.transAxes, **kw)


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


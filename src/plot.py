import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.ticker import PercentFormatter
from sklearn.manifold import MDS
cmap = cm.get_cmap(name="Dark2")

def mdsplot(true_q, fw_q, srfw_q, exp_dir=None):
    X = np.vstack((true_q, fw_q, srfw_q, np.ones(len(true_q[0]))))
    mds = MDS(n_components=2)
    y = mds.fit_transform(X)
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    sp1 = len(true_q)
    sp2 = len(fw_q) + sp1
    sp3 = len(srfw_q) + sp2
    ax.scatter(y[:sp1, 0], y[0:sp1, 1], marker="X", alpha=0.75, c=cmap(0), label="Ground Truth")
    ax.scatter(y[sp1:sp2, 0], y[sp1:sp2, 1], marker="o", alpha=0.75, c=cmap(1), label="FW")
    ax.scatter(y[sp2:sp3, 0], y[sp2:sp3, 1], marker="^", alpha=0.75, c=cmap(2), label="SRFW")
    ax.set_title("MDS Mapped To 2-dim")
    ax.legend()
    if exp_dir:
        plt.savefig(os.path.join(exp_dir, "mds.png"))

def boxplot(estimators, pop, ps, price, seeds, ctype="all", exp_dir=None):
    bcmap = cm.get_cmap(name="tab20")
    def plot(ax, est_prob, ground_truth, tp, ct):
        box = ax.boxplot(est_prob, labels=ps.pid_off, patch_artist=True)
        ax.set_xlabel("Purchase Option")
        ax.set_ylabel("Choice Probability Value")

        linecolor = [bcmap(i, alpha=0.9) for i in range(len(ground_truth))]
        facecolor = [bcmap(i, alpha=0.2) for i in range(len(ground_truth))]
        for i in range(len(ground_truth)):
            box["boxes"][i].set(edgecolor=linecolor[i], facecolor=facecolor[i])
            box["medians"][i].set(color=linecolor[i])
            box["fliers"][i].set(markeredgecolor=linecolor[i])
            box["caps"][2*i].set(color=linecolor[i], alpha=0.5)
            box["caps"][2*i+1].set(color=linecolor[i], alpha=0.5)
            box["whiskers"][2*i].set(color=linecolor[i], alpha=0.5)
            box["whiskers"][2*i+1].set(color=linecolor[i], alpha=0.5)
        ax.scatter(np.arange(1,len(ground_truth)+1), ground_truth, marker="X", alpha=1, color="black", label="Ground Truth")
        ax.set_xticklabels(ps.pid_off, rotation=45)
        ax.set_title(f"Type {tp} (count: {ct})")
        ax.legend()

    if ctype == "all":
        k = len(estimators)
        ncol = int(np.ceil(np.sqrt(k)))
        nrow = k // ncol + 1
        fig = plt.figure(figsize=(5*ncol,5*nrow))
        i = 1
        for tp in pop.cluster_id:
            est_prob = np.asarray(estimators[tp])
            ground_truth = ps.choice_prob_vec(pop.preference_dict[tp], price)
            ax = fig.add_subplot(nrow, ncol, i)
            plot(ax, est_prob, ground_truth, tp, seeds[tp])
            i += 1
        plt.tight_layout()
    else:
        est_prob = np.asarray(estimators[ctype])
        ground_truth = ps.choice_prob_vec(pop.preference_dict[ctype], price)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        plot(ax, est_prob, ground_truth, ctype, seeds[ctype])

    if exp_dir:
        plt.savefig(os.path.join(exp_dir, f"boxplot-{ctype}.png"))

def d3setup():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection = "3d")
    dd = 1
    ax.view_init(elev=45.0, azim=dd * 45)
    ax.plot_surface(
        np.array([[0, 1], [0, 0]]),
        np.array([[0, 0], [1, 1]]),
        np.array([[1, 0], [0, 0]]),
        alpha=0.2,
    )
    ax.text(0,0,1, "(0,0,1)")
    ax.text(0,1,0, "(0,1,0)")
    ax.text(1,0,0, "(1,0,0)")
    ax.set_xlabel("Prob No-Purchase (x)")
    ax.set_ylabel("Prob Purchase Prod 1 (y)")
    ax.set_zlabel("Prob Purchase Prod 2 (z)")
    return fig, ax


def d3plot_result(true_q, fw_q, srfw_q, rand_price, exp_dir):
    # agg = np.average([ps.choice_prob_vec(p, rand_price) for p in pop.preference_vec], axis=0, weights=pop.alpha)

    fig, ax = d3setup()
    # ax.scatter([agg[0]], [agg[1]], [agg[2]], color=cmap(3), marker="*", s=150, alpha=0.5, label="Aggregated")
    ax.scatter(true_q[:, 0], true_q[:, 1], true_q[:, 2], color=cmap(1), marker="X", s=50, alpha=0.5, label="Ground Truth")
    ax.scatter(srfw_q[:, 0], srfw_q[:, 1], srfw_q[:, 2], color=cmap(2), marker="^", s=50, alpha=1, label="SRFW")
    ax.scatter(fw_q[:, 0], fw_q[:, 1], fw_q[:, 2], color=cmap(0), marker="o", s=50, alpha=0.75, label="FW")
    ax.set_title(f"Choice Prob Under A Random Price {rand_price}", y=1.1)
    ax.grid(False)
    plt.legend()
    plt.savefig(os.path.join(exp_dir, "d3result.png"))

def d3plot(ps, pop, price, beta_set, save_dir, value_range=20, granularity=2, single=True):
    preference_grid = np.meshgrid(*([np.arange(-value_range, value_range, granularity)] * ps.num_feat))
    gridlist = list(map(np.ravel, preference_grid))
    print(len(gridlist))
    poss_q = []
    for i in range(len(gridlist[0])):
        p_vec = [x[i] for x in gridlist]
        q = ps.choice_prob_vec(p_vec, price)
        poss_q.append(q)

    poss_q = np.asarray(poss_q)
    est_q = np.asarray([ps.choice_prob_vec(b, price) for b in beta_set])
    true_q = np.asarray([ps.choice_prob_vec(p, price) for p in pop.preference_vec])

    fig = plt.figure(figsize=(10, 8))
    def plot(ax, dd):
        # fig = plt.figure(figsize=(10, 8))
        # ax = fig.add_subplot(1, 1, 1, projection = "3d")
        # dd = 1
        ax.view_init(elev=45.0, azim=dd * 45)
        ax.plot_surface(
            np.array([[0, 1], [0, 0]]),
            np.array([[0, 0], [1, 1]]),
            np.array([[1, 0], [0, 0]]),
            alpha=0.2,
        )
        ax.scatter(poss_q[:, 0], poss_q[:, 1], poss_q[:, 2], color=cmap(2), alpha=0.20, s=1, label=r"Non-restricted $q$'s")
        ax.scatter(true_q[:, 0], true_q[:, 1], true_q[:, 2], color=cmap(1), marker="X", s=150, alpha=0.75, label="Ground Truth")
        ax.scatter(est_q[:, 0], est_q[:, 1], est_q[:, 2], color=cmap(0), marker="^", s=50, alpha=0.75, label=r"Estimated $q$'s")
        ax.text(0,0,1, "(0,0,1)")
        ax.text(0,1,0, "(0,1,0)")
        ax.text(1,0,0, "(1,0,0)")
        ax.set_xlabel("Prob No-Purchase (x)")
        ax.set_ylabel("Prob Purchase Prod 1 (y)")
        ax.set_zlabel("Prob Purchase Prod 2 (z)")
        ax.set_title(f"MNL Choice Prob Under A Random Price {price}", y=1.1)

    if single:
        ax = fig.add_subplot(1, 1, 1, projection = "3d")
        plot(ax, 1)
        ax.legend()
        plt.savefig(os.path.join(save_dir, "d3init.png"))
    else:
        for dd in range(4):
            ax = fig.add_subplot(2, 2, dd + 1, projection="3d")
            plot(ax, dd)
        plt.legend(bbox_to_anchor=(1,1.25), loc="upper left")

def compare_prob_est(estimators, cons_type, ground_truth, ps, ax):
    m = len(ground_truth)
    prod = [[x[j] for x in estimators[cons_type]] for j in range(m)]
    # fig = plt.figure(figsize=(6,2))
    # ax = fig.add_subplot(ord, 1, ord)
    for i, v in enumerate(ground_truth):
        label_est = f"Prod {i} (est)" if i != 0 else "Offset (est)"
        label_true = f"Prod{i} (true)" if i != 0 else "Offset (true)"
        ax.scatter(prod[i], [i] * len(prod[i]), color=cmap(i), alpha=0.5, s=1, label=label_est)
        ax.scatter([v], [i], color=cmap(i), marker="x", s=50, label=label_true)

    # ax.legend(bbox_to_anchor=(1,1), loc="upper left")
    M = ps.num_prod
    ax.set_yticks(np.arange(M+1))
    ax.set_ylim((-0.5, M+0.5))
    # plt.gca().yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_yticklabels(ps.pid_off)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # ax.spines["bottom"].set_position(("outward", 10))
    ax.tick_params(length=0)
    ax.set_xlabel("Choice Probability Value")
    ax.set_title(f"Estimated Choice Probability Value Comparison (Type {cons_type})")


import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
# from matplotlib.ticker import PercentFormatter, Multiplelocator
cmap = cm.get_cmap(name="Dark2")

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


def d3plot_result(ps, pop, rand_price, exp_dir, fw, srfw):
    agg = np.average([ps.choice_prob_vec(p, rand_price) for p in pop.preference_vec], axis=0, weights=pop.alpha)
    srfw_q = np.asarray([ps.choice_prob_vec(b, rand_price) for b in srfw.active_beta])
    fw_q = np.asarray([ps.choice_prob_vec(b, rand_price) for b in fw.active_beta])
    true_q = np.asarray([ps.choice_prob_vec(p, rand_price) for p in pop.preference_vec])

    fig, ax = d3setup()
    ax.scatter([agg[0]], [agg[1]], [agg[2]], color=cmap(3), marker="*", s=150, alpha=0.5, label="Aggregated")
    ax.scatter(true_q[:, 0], true_q[:, 1], true_q[:, 2], color=cmap(1), marker="X", s=50, alpha=0.5, label="True Q's")
    ax.scatter(srfw_q[:, 0], srfw_q[:, 1], srfw_q[:, 2], color=cmap(2), marker="^", s=50, alpha=1, label="Estimated Q's (SRFW)")
    ax.scatter(fw_q[:, 0], fw_q[:, 1], fw_q[:, 2], color=cmap(0), marker="o", s=50, alpha=0.75, label="Estimated Q's (FW)")
    ax.set_title(f"Choice Prob Under A Random Price {rand_price}", y=1.1)
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
        ax.scatter(poss_q[:, 0], poss_q[:, 1], poss_q[:, 2], color=cmap(2), alpha=0.20, s=1, label="Non-restricted Q's")
        ax.scatter(true_q[:, 0], true_q[:, 1], true_q[:, 2], color=cmap(1), marker="X", s=150, alpha=0.75, label="True Q's")
        ax.scatter(est_q[:, 0], est_q[:, 1], est_q[:, 2], color=cmap(0), marker="^", s=50, alpha=0.75, label="Estimated Q's")
        ax.text(0,0,1, "(0,0,1)")
        ax.text(0,1,0, "(0,1,0)")
        ax.text(1,0,0, "(1,0,0)")
        ax.set_xlabel("Prob No-Purchase (x)")
        ax.set_ylabel("Prob Purchase Prod 1 (y)")
        ax.set_zlabel("Prob Purchase Prod 2 (z)")
        ax.set_title(f"MNL Choice Prob Under A Random Price {rand_price}", y=1.1)

    if single:
        ax = fig.add_subplot(1, 1, 1, projection = "3d")
        plot(ax, 1)
        ax.legend()
        plt.savefig(os.path.join(save_dir, "d3old.png"))
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


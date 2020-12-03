import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
# from matplotlib.ticker import PercentFormatter, Multiplelocator
cmap = cm.get_cmap(name="Dark2")

def d3plot(ps, pop, price, beta_set, single=True):
    preference_grid = np.meshgrid(*([np.arange(-10, 10, 0.1)] * ps.num_feat))
    gridlist = list(map(np.ravel, preference_grid))
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
        ax.view_init(elev=45.0, azim=dd * 45)
        ax.plot_surface(
            np.array([[0, 1], [0, 0]]),
            np.array([[0, 0], [1, 1]]),
            np.array([[1, 0], [0, 0]]),
            alpha=0.2,
        )
        ax.scatter(poss_q[:, 0], poss_q[:, 1], poss_q[:, 2], color=cmap(2), alpha=0.25, s=1, label="Possible Q's")
        ax.scatter(est_q[:, 0], est_q[:, 1], est_q[:, 2], color=cmap(0), marker="^", s=50, alpha=0.25, label="Estimated Q's")
        ax.scatter(true_q[:, 0], true_q[:, 1], true_q[:, 2], color=cmap(1), marker="X", s=150, label="True Q's")
        ax.legend()

    if single:
        ax = fig.add_subplot(1, 1, 1, projection = "3d")
        plot(ax, 1)
    else:
        for dd in range(4):
            ax = fig.add_subplot(2, 2, dd + 1, projection="3d")
            plot(ax, dd)

price = np.ones(2)
d3plot(ps, pop, price, beta_set, single=True)


def compare_prob_est(estimators, cons_type, ground_truth):
    m = len(ground_truth)
    prod = [[x[j] for x in estimators[cons_type]] for j in range(m)]
    fig = plt.figure(figsize=(6,2))
    ax = fig.add_subplot(111)
    for i, v in enumerate(ground_truth):
        label_est = f"Prod {i} (est)" if i != 0 else "Offset (est)"
        label_true = f"Prod{i} (true)" if i != 0 else "Offset (true)"
        ax.scatter(prod[i], [i] * len(prod[i]), color=cmap(i), alpha=0.5, s=1, label=label_est)
        ax.scatter([v], [i], color=cmap(i), marker="x", s=50, label=label_true)

    ax.legend(bbox_to_anchor=(1,1), loc="upper left")
    ax.set_yticks(np.arange(M+1))
    ax.set_ylim((-0.5,M+0.5))
    # plt.gca().yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_yticklabels(ps.pid_off)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # ax.spines["bottom"].set_position(("outward", 10))
    ax.tick_params(length=0)
    ax.set_xlabel("Choice Probability Value")
    ax.set_title(f"Estimated Choice Probability Value Comparison (Type {cons_type})")

tp = "A"
ground_truth = ps.choice_prob_vec(pop.preference_dict[tp], np.ones(2))
compare_prob_est(estimators, tp, ground_truth)
tp = "B"
ground_truth = ps.choice_prob_vec(pop.preference_dict[tp], np.ones(2))
compare_prob_est(estimators, tp, ground_truth)

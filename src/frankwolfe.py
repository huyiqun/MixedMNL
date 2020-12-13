import time
import random
random.seed(123)
import string
from collections import namedtuple, OrderedDict

import numpy as np
from scipy.optimize import minimize, linprog
from logging_util import ColoredLog

ITER_SEP = "=" * 80 + "\n"
DEFAULT_VERBOSE = 3


class FrankWolfe(object):
    def __init__(self, sim, verbose=DEFAULT_VERBOSE):
        self.sim = sim
        self.iter = 0
        self.logger = ColoredLog(self.__class__.__name__, verbose=verbose)
        self.converged = False
        self.loss_value = []

    @property
    def active_alpha(self):
        return self.learned_alpha[self.active_index]

    @property
    def active_beta(self):
        return self.learned_beta[self.active_index]

    def q_optimization(self):
        """support finding"""
        # print(ITER_SEP)
        self.logger.info(f"Support Finding Step: optimizing v (iter: {self.iter})")

        n = self.sim.ps.num_feat
        if self.iter == 0:
            self.learned_beta = np.random.rand(1, n)
            self.learned_alpha = np.ones(1, dtype=float)
            self.active_index = np.ones(1, dtype=bool)

        self.logger.debug(
                np.hstack((self.learned_alpha[:, np.newaxis], self.learned_beta)),
                header=["alpha"] + [f"beta {i+1}" for i in range(n)],
                index=['active' if self.active_index[i] else 'inactive'
                    for i in range(len(self.learned_beta))],
                caption="Current learned params",
                )

        def func(beta):
            total = 0
            for t, p in enumerate(self.sim.exp_price):
                qs = [self.sim.ps.choice_prob_vec(b, p) for b in self.active_beta]
                calculated_market_share = np.average(qs, axis=0, weights=self.active_alpha)
                g_y_diff = calculated_market_share - self.sim.simulated_market_share[t]
                total += np.dot(g_y_diff, self.sim.ps.choice_prob_vec(beta, p))
            return total

        while True:
            res = minimize(func, np.random.rand(n))
            if res.success:
                break
        # print(res)
        self.logger.info(f"New moving direction: {res.x}")

        self.learned_beta = np.vstack((self.learned_beta, res.x))
        self.learned_alpha = np.hstack((self.learned_alpha, 0))
        self.active_index = np.hstack((self.active_index, True))
        # return res.x

    def alpha_optimization(self, bounds=None, constraints=None):
        """proportion updating"""
        self.logger.info(
            f"Proportion Updating Step: optimizing alpha (iter: {self.iter})"
        )

        n = self.sim.ps.num_feat
        self.logger.debug(
                np.hstack((self.learned_alpha[:, np.newaxis], self.learned_beta)),
                header=["alpha"] + [f"beta {i+1}" for i in range(n)],
                index=['new' if i == len(self.learned_alpha)-1 else 'active' if self.active_index[i] else 'inactive'
                    for i in range(len(self.learned_beta))],
                caption="Current learned params",
                )

        bounds = tuple([(0, 1)] * len(self.learned_alpha))
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        func = lambda alpha: self.total_loss(alpha, self.learned_beta)

        while True:
            res = minimize(
                func,
                self.learned_alpha,
                bounds=bounds,
                constraints=constraints,
                #  method="SLSQP",
            )
            if res.success:
                break
                # print(res)

        self.learned_alpha = res.x
        self.active_index[self.learned_alpha < 1e-3] = False
        self.active_index[self.learned_alpha >= 1e-3] = True

        self.iter += 1
        new_loss = self.total_loss(self.learned_alpha, self.learned_beta)
        self.loss_value.append(new_loss)
        # print(ITER_SEP)
        self.logger.info(
                np.hstack((self.learned_alpha[:, np.newaxis], self.learned_beta)),
                header=["alpha"] + [f"beta {i+1}" for i in range(n)],
                index=['active' if self.active_index[i] else 'inactive'
                    for i in range(len(self.learned_beta))],
                caption=f"New loss: {new_loss}",
                )

    def total_loss(self, alpha, beta):
        loss = []
        total = 0
        for t, p in enumerate(self.sim.exp_price):
            diff = self.sim.simulated_market_share[t] - np.sum([a * self.sim.ps.choice_prob_vec(b, p) for a, b in zip(alpha, beta)], axis=0)
            square_norm = np.linalg.norm(diff) ** 2
            loss.append(square_norm)
            total += square_norm

        return total

    def run(self, max_iter=10, tol=1e-6):
        self.q_opt_time = []
        while self.iter <= max_iter and not self.converged:
            st = time.time()
            self.q_optimization()
            ed = time.time()
            self.q_opt_time.append(ed-st)

            self.alpha_optimization()
            print(self.loss_value)
            if len(self.loss_value) > 1 and np.linalg.norm(self.loss_value[-2]-self.loss_value[-1]) < tol:
                self.converged = True

        n = self.sim.ps.num_feat
        self.logger.info(
                np.hstack((self.learned_alpha[:, np.newaxis], self.learned_beta)),
                header=["alpha"] + [f"beta {i+1}" for i in range(n)],
                index=['active' if self.active_index[i] else 'inactive'
                    for i in range(len(self.learned_beta))],
                caption="Final learned params",
                )
        print(f"Loss: {self.loss_value}")


class SubRegionFrankWolfe(FrankWolfe):
    """docstring for FrankWolfe"""

    def __init__(self, sim, verbose=DEFAULT_VERBOSE):
        super().__init__(sim, verbose=verbose)
        self.logger = ColoredLog(self.__class__.__name__, verbose=verbose)

    def q_optimization(self, beta_set):
        """support finding"""
        self.logger.info(f"Support Finding Step: optimizing v (iter: {self.iter})")


        if self.iter == 0:
            init = np.random.choice(len(beta_set))
            self.learned_beta = beta_set[init][np.newaxis]
            self.learned_alpha = np.ones(1)
            self.active_index = np.ones(1, dtype=bool)
            beta_set = np.delete(beta_set, init, axis=0)

        m, n = beta_set.shape
        self.logger.debug(
                np.hstack((self.learned_alpha[:, np.newaxis], self.learned_beta)),
                header=["alpha"] + [f"beta {i+1}" for i in range(n)],
                index=['active' if self.active_index[i] else 'inactive'
                    for i in range(len(self.learned_beta))],
                caption="Current learned params",
                )

        coef = []
        for b in beta_set:
            gradient = 0
            for t, p in enumerate(self.sim.exp_price):
                # calculate current market share based on selected alpha & beta
                qs = [self.sim.ps.choice_prob_vec(bb, p) for bb in self.active_beta]
                calculated_market_share = np.average(qs, axis=0, weights=self.active_alpha)
                # gradient for price at time t
                g_y_diff = calculated_market_share - self.sim.simulated_market_share[t]
                gradient += np.dot(g_y_diff, self.sim.ps.choice_prob_vec(b, p))
            coef.append(gradient)

            # gradient = np.sum([np.dot(sim.simulated_market_share[t] - np.average([ps.choice_prob_vec(bb, p) for bb in current_beta], axis=0, weights=alpha), ps.choice_prob_vec(b, p)) for t,p in enumerate(sim.exp_price)])

        A_eq = np.ones((1, m))
        b_eq = [1]

        while True:
            res = linprog(coef, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1))
            if res.success:
                break

        new_dir = np.argmax(res.x)
        # print("res: ", new_dir, beta_set[new_dir])
        self.logger.info(f"picked direction {beta_set[new_dir]}")

        self.learned_beta = np.vstack((self.learned_beta, beta_set[new_dir]))
        self.learned_alpha = np.hstack((self.learned_alpha, 0))
        self.active_index = np.hstack((self.active_index, True))
        beta_set = np.delete(beta_set, new_dir, axis=0)

        return beta_set

    def run(self, beta_set, max_iter=10, tol=1e-6):
        self.q_opt_time = []
        while self.iter <= max_iter and not self.converged:
            st = time.time()
            beta_set = self.q_optimization(beta_set)
            ed = time.time()
            self.q_opt_time.append(ed-st)

            self.alpha_optimization()
            print(self.loss_value)
            if len(self.loss_value) > 1 and np.linalg.norm(self.loss_value[-2]-self.loss_value[-1]) < tol:
                self.converged = True

        _, n = beta_set.shape
        self.logger.info(
                np.hstack((self.learned_alpha[:, np.newaxis], self.learned_beta)),
                header=["alpha"] + [f"beta {i+1}" for i in range(n)],
                index=['active' if self.active_index[i] else 'inactive'
                    for i in range(len(self.learned_beta))],
                caption="Final learned params",
                )
        print(f"Loss: {self.loss_value}")

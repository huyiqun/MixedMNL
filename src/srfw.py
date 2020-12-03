import random
random.seed(123)
from collections import namedtuple, OrderedDict

import numpy as np
from scipy.optimize import minimize, linprog

from experiment import Simulator
from learning import OnlineLearning, Experiment, Params
from logging_util import ColoredLog
from simulate_util import Product_Set, Population
from learning import OnlineLearning, Cycle, Experiment, Params
from pricing_MIP import optimization_MIP_mixedMNL_pulp
from pricing_MIP import cost, Demand, Point


CYCLE_SEP = "=" * 80 + "\n"
EXP_SEP = "~" * 80 + "\n"
DEFAULT_VERBOSE = 3
class FrankWolfe(object):
    def __init__(self, params, verbose=DEFAULT_VERBOSE):
        self.logger = ColoredLog(self.__class__.name, verbose=verbose)
        self.alpha = params.alpha
        self.exp = exp
        self.logger = ColoredLog(self.__class__.__name__, verbose=verbose)
        self.iter = 1

        self.active_set = OrderedDict()
        for i in range(len(self.alpha)):
            self.active_set[i] = params.choice[i]

        self.index_mask = np.ones(len(self.alpha))
        self.choice_prob_mat = np.array(list(self.active_set.values()))
        self.calculated_market_share_g = self.calculate_market_share()
        self.converged = False

    def calculate_market_share(self):
        return np.dot(np.multiply(self.index_mask, self.alpha), self.choice_prob_mat)


    @property
    def alpha_masked(self):
        return np.array([a for a, m in zip(self.alpha, self.index_mask) if m == 1])

    @property
    def choice_prob_mat_masked(self):
        return np.array(
            [q for q, m in zip(self.choice_prob_mat, self.index_mask) if m == 1]
        )

    def q_optimization(self, boundary_set):
        """support finding"""
        self.logger.warning(f"Support Finding Step: optimizing v (iter: {self.iter})")

        if self.iter > 1:
            b_mat = np.transpose(list(self.choice_prob_mat) + boundary_set)
        else:
            b_mat = np.transpose(list(boundary_set))

        chk = np.transpose(b_mat)
        self.logger.info(chk, header=exp.ps.pid_off, caption="Conv(Q) plus q(t-1)")
        m = len(chk)
        g_y_diff = self.calculated_market_share_g - self.exp.simulated_market_share
        c = np.dot(g_y_diff, b_mat)
        A_eq = [np.ones(m)]
        b_eq = [1]

        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1))
        if self.logger.level <= 10:
            print(res)
        q = np.dot(b_mat, res.x)

        self.active_set[len(self.active_set)] = q
        self.choice_prob_mat = np.append(self.choice_prob_mat, q[np.newaxis, :], axis=0)
        self.index_mask = np.append(self.index_mask, [1])
        self.alpha = np.append(self.alpha, [0])
        #  self.index_mask.append(1)
        #  self.alpha.append(0)

        return q

    def alpha_optimization(self, bounds=None, constraints=None):
        """proportion updating"""
        self.logger.warning(
            f"Proportion Updating Step: optimizing alpha (iter: {self.iter})"
        )
        self.logger.debug(f"active set length: {len(self.active_set)}")
        if not bounds:
            bounds = tuple(
                [
                    (0, 0)
                    if (self.iter == 1 and i < len(self.alpha) - 1)
                    or self.index_mask[i] == 0
                    else (0, 1)
                    for i in range(len(self.index_mask))
                ]
            )
            self.logger.info(f"alpha bounds: {bounds}")

        if not constraints:
            constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}

        def obj_proportion(alpha):
            coef = np.multiply(self.index_mask, alpha)
            v = np.dot(coef, self.choice_prob_mat)
            d = np.linalg.norm(v - self.exp.simulated_market_share)
            return d ** 2

        res = minimize(
            obj_proportion,
            self.alpha,
            bounds=bounds,
            constraints=constraints,
            #  method="SLSQP",
        )
        if self.logger.level <= 10:
            print(res)

        self.iter += 1
        self.alpha = res.x
        self.calculated_market_share_g = self.calculate_market_share()
        self.index_mask[self.alpha < 1e-3] = 0
        self.logger.info(f"alpha optimized result: {self.alpha}")
        self.logger.info(f"alpha optimized result masked: {self.alpha_masked}")

        if res.x[-1] > 1e-3:
            self.logger.info(
                f"market share g afer alpha opt: {self.calculated_market_share_g}"
            )
        else:
            logger.info(f"last direction not included")
            #  self.index_mask[-1] = 0

        return self.alpha_masked


class SubRegionFrankWolfe(object):
    """docstring for FrankWolfe"""

    def __init__(self, exp, params, verbose=DEFAULT_VERBOSE):



        rewards = []

        learning = OnlineLearning(ps, pop)
        new_params = None
        max_T_allowed = 15
        for t in range(max_T_allowed):
            old_price = np.array([p for p in learning.ps.prices])
            logger.critical(f"optimal price from last cycle: {old_price}")
            cyc = learning.cycles[t]

            exp = Experiment(cyc.ps, cyc.pop)
            exp.simulate()
            #  logger.info(exp.theoretical_market_share)

            if new_params is not None:
                current_params = new_params
            else:
                current_params = Params(
                    alpha=learning.params.alpha,
                    choice=random.sample(cyc.boundary_set, len(learning.params.alpha)),
                )
            logger.info(current_params.table, caption="Input Parameters")

            fw = FrankWolfe(exp, current_params)
            g_old = fw.calculated_market_share_g

            while not fw.converged:
                q_best = fw.q_optimization(cyc.boundary_set)
                logger.info(f"Moving direction q: {q_best}")
                alpha_new = fw.alpha_optimization()
                #  logger.info(f"new alpha: {alpha_new}")

                g = fw.calculated_market_share_g
                logger.info(f"observed g: {exp.simulated_market_share}")
                logger.info(f"old_g: {g_old}")
                logger.info(f"updated_g: {g}")

                g_gap = np.linalg.norm(g - g_old)
                logger.warning(f"g_gap: {g_gap}")
                g_y_distance = np.linalg.norm(g - exp.simulated_market_share)
                logger.warning(f"norm |g-y_vec|:{g_y_distance}\n")
                duality_gap = (-1) * np.dot(exp.simulated_market_share - g, g - q_best)
                logger.warning(f"duality gap: {duality_gap}\n")

                g_old = g
                pts = np.hstack((fw.alpha_masked[:, np.newaxis], fw.choice_prob_mat_masked))
                logger.info(
                    pts,
                    header=["alpha"] + exp.ps.pid_off,
                    index=False,
                    caption=f"After FW at iteration {fw.iter-1}",
                )

                if len(fw.alpha) - len(current_params.alpha) > len(cyc.boundary_set):
                    if g_gap > 1e-3:
                        logger.warning("g not converged, all boundary pt included")
                    print(EXP_SEP)
                    print(f"finish FrankWolfe iteration {fw.iter-1}")
                    print(EXP_SEP)
                    break

                if g_gap < 1e-3:
                    fw.converged = True
                    logger.warning("g converged")

                    if g_y_distance < 1e-3:
                        logger.warning("also converged to y")
                    else:
                        logger.warning("stopped, not converged to y")
                        logger.warning(f"g_y_gap: {g_y_distance}")
                    print(EXP_SEP)
                    print(f"finish FrankWolfe iteration {fw.iter-1}")
                    print(EXP_SEP)

            learned_alpha = fw.alpha_masked
            learned_choice_prob = fw.choice_prob_mat_masked

            tmp = Params(alpha=learned_alpha, choice=learned_choice_prob)
            logger.debug(tmp.table, caption="Parameters print after FW finished:")
            learned_pop_demand = np.dot(learned_alpha, learned_choice_prob)
            logger.warning(f"learned population demand: {learned_pop_demand}")

            # calculate valuation matrix under old price
            learned_valuation = [
                np.array(exp.ps.prices) + np.log(q[:-1]) - np.log(q[-1])
                for q in learned_choice_prob
            ]
            logger.info(
                learned_valuation,
                header=exp.ps.pid_off,
                index=False,
                caption="Learned Valuation Matrix",
            )

            #  f = ps.features
            #  print(f)
            #  XTX = np.dot(np.transpose(f), f)
            #  print(XTX)
            #  XTXinv = np.linalg.inv(XTX)
            #  print(XTXinv)
            #  XTXinvXT = np.dot(XTXinv, np.transpose(f))
            #  print(XTXinvXT)
            #  XTXinvXTY = np.dot(XTXinvXT, learned_valuation[0])
            #  print(XTXinvXTY)
            #  learned_beta = [
            #  np.dot(
            #  np.dot(np.linalg.inv(np.dot(np.transpose(f), f)), np.transpose(f)), v
            #  )
            #  for v in learned_valuation
            #  ]
            #  print(learned_beta)
            #  beta_based_valuation = np.array([np.dot(ps.features, b) for b in learned_beta])

            #  printt(
            #  beta_based_valuation,
            #  header=exp.ps.pid + ["Offset"],
            #  index=False,
            #  caption="Beta Based Valuation",
            #  )

            _, optimal_price, _ = optimization_MIP_mixedMNL_pulp(
                cost,
                Demand,
                Point,
                learned_alpha[:, np.newaxis],
                np.transpose(learned_valuation),
            )

            valuation = np.array(learned_valuation)
            price = np.array([max(0, p) for p in optimal_price])
            logger.info(f"optimal price = {optimal_price}\noptimal price fixed = {price}")

            #  print(price)
            #  print("diff", valuation - price)
            #  print("exp", np.exp(cluster_valuation - optimal_price))
            #  print("exp_sum", np.sum(np.exp(cluster_valuation - optimal_price), axis=1))

            exp.ps.set_price(price)
            #  exp.ps.info
            true_demand_under_new_price = [
                exp.ps.choice_prob_vec(b) for b in pop.preference_vec
            ]
            logger.info(
                np.hstack(
                    (np.array(pop.alpha)[:, np.newaxis], true_demand_under_new_price)
                ),
                header=["true alpha"] + exp.ps.pid_off,
                index=False,
                caption="True demand under new optimal price",
            )

            demand = np.dot(np.array(pop.alpha), true_demand_under_new_price)
            logger.warning(f"true population demand under new optimal price: {demand}")
            reward = np.dot(price, demand[:-1])

            logger.warning(f"reward: {reward}")
            rewards.append(reward)
            #  learning.update_price(np.array(optimalprice_vec))

            #  print(price)
            #  print(old_price)
            #  print(price - old_price)
            logger.warning(
                f"price: {price}\nold price: {old_price}\nprice diff: {np.linalg.norm(price - old_price)}"
            )

            print(CYCLE_SEP)
            print(f"finish cycle {t+1}")
            print(CYCLE_SEP)

            if np.linalg.norm(price - old_price) < 1e-2 * ps.num_prod:
                print("Price converged; Learning Terminated.")
                break
            elif t >= max_T_allowed - 1:
                print("Price not converged; Max T reached; Learning Terminated.")
                break
            else:
                logger.warning("Preparing to add new Cycle...")

                #  calculate q0(offset prob using ESTIMATED valuation and NEW price)
                q0 = 1 / (np.sum(np.exp(valuation - price), axis=1) + 1)
                #  calculate purchase prob for other probability needs coefficient with formula q_i = exp(V_i-p_i)*q0
                coef_mat = []
                for i in range(ps.num_prod):
                    # note v is for each product, transpose to get it for each cluster
                    v = np.exp(valuation[:, i] - price[i])
                    coef_mat.append(v)

                coef_mat = np.transpose(coef_mat)
                q_under_new_price = [
                    np.append(np.multiply(x, y), np.array(x)) for x, y in zip(q0, coef_mat)
                ]

                logger.info(
                    np.hstack((learned_alpha[:, np.newaxis], q_under_new_price)),
                    header=["alpha"] + ps.pid_off,
                    index=False,
                    caption="New Choice Prob Under Optimal Price:",
                )
                #  q_under_new_price should have the same size as number of clusters
                #  logger.debug(np.sum(q_under_new_price, axis=1))
                new_params = Params(alpha=learned_alpha, choice=q_under_new_price)
                #  new_params.print

                learning.add_learning_cycle(price, new_params)

        print(rewards)

        learning.get_current_learning_info()
        logger.warning("Truth under optimal price")
        true_alpha = np.array(pop.alpha)[:, np.newaxis]
        pop.info
        #  cv = [[2, 4], [1, 2]]
        cv = np.dot(pop.preference_vec, np.transpose(ps.features))
        logger.info(cv, header=ps.pid, index=pop.cid, caption="True valuation, i.e. beta*z")

        _, optimal_price_true, _ = optimization_MIP_mixedMNL_pulp(
            cost, Demand, Point, true_alpha, np.transpose(cv),
        )
        logger.info(f"True optimal price = {optimal_price_true}")
        #  logger.info(f"optx_true = {optx_true}")
        ps.set_price(optimal_price_true)
        pop.logger.info(pop.info, caption="Consumer info", index=pop.cid)
        #  market_share_true = (
        #  ps.choice_prob_vec(pop.preference_vec[0]) * 0.8
        #  + ps.choice_prob_vec(pop.preference_vec[1]) * 0.2
        #  )
        #  logger.info(market_share_true)
        demand = [ps.choice_prob_vec(pop.preference_vec[k]) for k in range(len(pop.alpha))]
        logger.info(
            demand,
            header=ps.pid_off,
            index=pop.cid,
            caption="Ground truch choice prob under true optimal price:",
        )
        pop_demand = np.dot(pop.alpha, demand)
        logger.debug(f"population demand: {pop_demand}")
        max_reward = np.dot(optimal_price_true, pop_demand[:-1])
        logger.warning(f"Max reward: {max_reward}")
        regret = np.array(max_reward - rewards)
        ave_regret = [np.sum(regret[: i + 1]) / (i + 1) for i in range(len(regret))]
        rr = np.vstack(([rewards], [regret], [ave_regret]))
        logger.warning(
            np.transpose(rr),
            index=map(str, np.array(list(range(t + 1))) + 1),
            header=["cycle", "rewards", "regret", "average regret"],
            caption="Rewards & Regret",
        )

        init_p = [1] * ps.num_prod
        init_q = np.dot(
            pop.alpha,
            [
                ps.choice_prob_vec(pop.preference_vec[k], init_p)
                for k in range(pop.num_types)
            ],
        )
        init_r = np.dot(init_p, init_q[:-1])
        #  print(init_p)
        #  print(init_q)
        #  print(init_r)

        cum_rew = np.cumsum([init_r] + rewards[:-1])
        print(cum_rew)
        ave_rew = [cum_rew[i] / (i + 1) for i in range(len(cum_rew))]
        print(ave_rew)

        m = len(learning.price_trace) * (settings.EXP_PER_CYC + 1)
        #  print(learning.price_trace)

        rr = [[r] * (settings.EXP_PER_CYC + 1) for r in ave_rew]
        print(rr)
        y = [r for l in rr for r in l]
        print(len(y))

        plt.hlines(max_reward, 0, m - 1, linestyles="dashed", label="Max Reward")
        plt.plot(range(m), y, label="Learning Reward", alpha=0.5)

        st = settings.EXP_PER_CYC + 1
        x = [np.sqrt(i) for i in range(st, len(y))]
        print(x)
        #  a, *b = np.polyfit(x, regret[st:], 2)
        #  print(a)
        fit = np.poly1d(np.polyfit(x, y[st:], 2))
        plt.plot(range(st, len(y)), fit(x), label=r"$\sqrt{T}$ fitting", ls="-.")
        plt.xlabel("Number of Experiments (m)")
        plt.ylabel("Reward")
        plt.legend(loc="lower right")

        plt.savefig("multitype.png")

        #  logger.warning(f"Learned price: {price}")
        #  logger.warning(f"True optimal price: {optimal_price_true}")
        #  logger.warning(f"price gap: {np.linalg.norm(optimal_price_true-price)}")
        #  logger.info(f"Now approximate using single consumer type...")

        #  logger.info(f"regrets: {regret}")

        #  p1 = optimalprice_vec_true[0]
        #  p2 = optimalprice_vec_true[1]

        #  p1ms = (np.exp(2 - p1)) / (np.exp(2 - p1) + np.exp(4 - p2) + 1) * 0.8 + (
        #  np.exp(1 - p1)
        #  ) / (np.exp(1 - p1) + np.exp(2 - p2) + 1) * 0.2
        #  p2ms = (np.exp(4 - p2)) / (np.exp(2 - p1) + np.exp(4 - p2) + 1) * 0.8 + (
        #  ) / (np.exp(1 - p1) + np.exp(2 - p2) + 1) * 0.2
        #  print(p1ms)
        #  print(p2ms)

from collections import Counter, defaultdict, namedtuple, OrderedDict
import numpy as np
from tabulate import tabulate
from scipy.optimize import minimize, LinearConstraint, Bounds
from simulate_util import Population, Product_Set
from logging_util import ColoredLog
from settings import VERBOSE, EXP_PER_CYC, SIM_PER_EXP, TRANS_PER_EXP, SAMPLE_RATIO

Params = namedtuple("Params", ["alpha", "beta"])
old_settings = np.geterr()
np.seterr(all="raise")
#  np.seterr(**old_settings)


class Params(namedtuple("Params", ["alpha", "beta", "choice"], defaults=[None, None])):
    __slots__ = ()

    @property
    def table(self):
        #  k = len(self.alpha)
        mat = self.alpha[:, np.newaxis]
        header = ["alpha"]
        if self.beta is not None:
            n = len(self.beta[0])
            mat = np.concatenate((mat, self.beta), axis=1)
            header.extend([f"beta {i+1}" for i in range(n)])
        if self.choice is not None:
            m = len(self.choice[0])
            mat = np.concatenate((mat, self.choice), axis=1)
            header.extend([f"product {j+1}" for j in range(m - 1)] + ["Offset"])

        return np.vstack((header, mat))


class Learning(object):
    def __init__(self, product_set, population, verbose):
        self.pop = population
        self.ps = product_set

        self.logger = ColoredLog(self.__class__.__name__, verbose=verbose)
        self.logger.info("Creating Learning Process...")
        #  self.logger.debug(self.ps.info)
        #  self.logger.debug(self.pop.info)


class OfflineLearning(Learning):
    def __init__(self, product_set, population, num_exp, estimate_k=2, verbose=VERBOSE):
        super().__init__(product_set, population, verbose=verbose)
        self.converged = False
        self.price_trace = [[p for p in self.ps.prices]]
        self.trace_index = ["Initialization"]

        if self.pop.num_types > 1:
            k = estimate_k
            self.params = Params(alpha=np.array([float(1) / k] * k))
        else:
            self.params = Params(alpha=np.array([1]))

        self.logger.debug(self.params.table, caption="Init Params")

        self.price_trace = []
        self.simulated_market_share = []
        self.tms = []
        exp = Experiment(self.ps, self.pop)
        max_T_allowed = num_exp
        for t in range(max_T_allowed):
            p = np.random.uniform(low=0.0, high=1.0, size=self.ps.num_prod)
            self.logger.debug(f"Experiment {t} with price: {p}")
            self.price_trace.append(p)
            exp.ps.set_price(p)
            exp.calculate_groud_truth()
            exp.simulate()
            exp.print_experiment_info()
            self.simulated_market_share.append(exp.simulated_market_share)
            self.tms.append(exp.theoretical_market_share)


class OnlineLearning(Learning):
    def __init__(self, product_set, population, estimate_k=2, verbose=VERBOSE):
        super().__init__(product_set, population, verbose=verbose)
        self.converged = False
        self.price_trace = [[p for p in self.ps.prices]]
        self.trace_index = ["Initialization"]

        if self.pop.num_types > 1:
            k = estimate_k
            self.params = Params(alpha=np.array([float(1) / k] * k))
        else:
            self.params = Params(alpha=np.array([1]))

        self.logger.debug(self.params.table, caption="Init Params")

        self.logger.info("Adding Cycle 1...")
        cycle = Cycle(self.ps, self.pop, self.params)
        self.cycles = [cycle]
        self.historical_data = [cycle.sim_data]

    def add_learning_cycle(self, opt_price, params, num_exp=EXP_PER_CYC):
        self.update(opt_price, params)

        self.logger.info(f"Adding Cycle {len(self.cycles) + 1}...")
        self.ps.set_price(opt_price)
        cycle = Cycle(self.ps, self.pop, params, num_exp=num_exp)
        self.cycles.append(cycle)
        self.historical_data.append(cycle.sim_data)

        return cycle

    def update(self, new_price, params):
        """This function is only called after pricing optimization finished"""
        self.price_trace.append([p for p in new_price])
        self.trace_index.append(f"Cycle {len(self.price_trace) - 1} Opt P")
        self.params = params
        self.get_current_learning_info()

    def get_current_learning_info(self):
        num_finished_cycle = len(self.trace_index) - 1
        self.logger.info(f"Current phase: finished cycles: {num_finished_cycle}")
        self.logger.info(
            self.price_trace,
            caption="Price Trace:",
            header=self.ps.pid,
            index=self.trace_index,
        )
        self.logger.info(self.params.table, caption="end of previous cycle params")


class Cycle(object):
    def __init__(
        self, product_set, population, params, num_exp=EXP_PER_CYC, verbose=VERBOSE
    ):
        self.ps = product_set
        self.pop = population
        self.current_price = [p for p in self.ps.prices]
        self.params = params

        self.logger = ColoredLog(self.__class__.__name__, verbose=verbose)
        self.logger.info(f"Setting up {num_exp} independent experiments...")
        #  exp = Experiment(self.ps, self.pop)
        #  exp.experiment_info

        # experiment data
        self.exp_price = OrderedDict()
        self.experiments = OrderedDict()
        self.sim_ms = []
        self.sim_data = OrderedDict()
        self.trans_mat = OrderedDict()

        # estimation data
        #  self.estimated_beta = OrderedDict()
        #  self.estimated_alpha = OrderedDict()
        self.estimated_beta = []
        self.estimated_alpha = []
        self.estimate_keep = []

        for u in range(num_exp):
            exp = self.add_experiment()
            self.experiments[u] = exp
            self.exp_price[u] = exp.ps.prices
            self.sim_ms.append(exp.simulated_market_share)
            self.sim_data[u] = exp.sim_data
            self.trans_mat[u] = exp.transaction_data_matrix
            for l in range(SIM_PER_EXP):
                alpha_hat, beta_hat = self.alpha_beta_estimate(
                    exp.transaction_data_matrix
                )
                for a, b in zip(alpha_hat, beta_hat):
                    self.estimated_alpha.append(a)
                    self.estimated_beta.append(tuple(b))
                    if a > 0.95 and np.min(b) > 0.1:
                        self.estimate_keep.append(True)
                    else:
                        self.estimate_keep.append(False)

        chk = [
            [self.estimated_alpha[i], self.estimated_beta[i], self.estimate_keep[i]]
            for i in range(len(self.estimated_alpha))
        ]
        self.logger.debug(chk, header=["a", "b", "k"], caption="estimated")

        #  self.list_exp_price()
        self.reset_price()
        self.construct_beta_set()
        self.construct_boundary_set()

    def add_experiment(self, prange=0.05):
        delta = 0.1
        random_perturb = np.random.uniform(
            low=-1 * delta, high=delta, size=self.ps.num_prod
        )
        perturbed_price = self.current_price + random_perturb
        self.ps.set_price(perturbed_price)
        exp = Experiment(self.ps, self.pop)
        exp.simulate()
        return exp

    def list_exp_price(self):
        self.logger.info(
            self.exp_price.values(),
            caption="Experiment prices",
            header=self.ps.pid,
            index=[f"Experiment {k}" for k in range(len(self.exp_price))],
        )

    def check_exp_info(self, u):
        k = len(self.params.alpha)
        alpha_hat = self.estimated_alpha.get(u, [None] * k)
        beta_hat = self.estimated_beta[u]

        self.logger.info(
            [self.exp_price[u]],
            caption=f"Experiment {u} experiment price",
            header=self.ps.pid,
        )
        tmp_params = Params(alpha_hat, beta_hat)
        self.logger.info(tmp_params)

        return exp

    def alpha_beta_estimate(self, trans_data):
        num_types = len(self.params.alpha)
        num_features = len(self.pop.preference_vec[0])

        num_trans = len(trans_data)
        n = num_trans * SAMPLE_RATIO
        sub_trans = trans_data[np.random.choice(num_trans, int(n), replace=True)]
        sub_ms = np.mean(sub_trans, axis=0)
        #  print(sub_ms)
        #  self.experiment_info

        def obj(beta):
            y = np.reshape(beta, (num_types, (num_features + 1)))
            alpha = y[:, 0][:, np.newaxis]
            x = y[:, 1:]
            q = np.array([self.ps.choice_prob_vec(i) for i in x])
            calculated_market_share = np.sum(alpha * q, axis=0)
            diff = sub_ms - calculated_market_share
            #  diff = self.simulated_market_share - calculated_market_share
            return np.linalg.norm(diff) ** 2

        flen = num_types * (num_features + 1)
        x0 = map(float, np.random.randint(-5, 6, size=(flen,)))
        lb = np.array([-np.inf] * flen)
        ub = np.array([np.inf] * flen)
        A = np.zeros(flen)
        for i in range(num_types):
            lb[i * (num_features + 1)] = 0
            ub[i * (num_features + 1)] = 1
            A[i * (num_features + 1)] = 1

        bnds = Bounds(lb, ub)
        cons = LinearConstraint(A, 1, 1)

        init = np.array(list(x0))
        np.seterr(all="ignore")
        while True:
            res = minimize(obj, init, bounds=bnds, constraints=cons, tol=1e-6)
            if res.success:
                break
        np.seterr(all="raise")
        #  self.logger.info(beta_hat)
        ans = np.reshape(res.x, (num_types, (num_features + 1)))
        alpha_hat, beta_hat = ans[:, 0], ans[:, 1:]

        return alpha_hat, beta_hat

    def construct_beta_set(self):
        betas = np.array(self.estimated_beta)
        beta_mask = np.array(self.estimate_keep)
        self.beta_set = betas[beta_mask]

    def construct_boundary_set(self):
        self.boundary_set = []
        for b in self.beta_set:
            self.boundary_set.append(self.ps.choice_prob_vec(b))

    def reset_price(self):
        self.ps.set_price(self.current_price)


class Experiment(object):
    def __init__(self, product_set, population, verbose=VERBOSE):
        self.ps = product_set
        self.pop = population
        self.calculate_groud_truth()
        self.logger = ColoredLog(self.__class__.__name__, verbose=verbose)
        self.print_experiment_info()

    def calculate_groud_truth(self):
        n = self.ps.num_prod + 1
        K = self.pop.num_types
        self.choice_prob_dict = {}
        self.theoretical_market_share = np.zeros(n)
        for k in range(K):
            p_vec = self.ps.choice_prob_vec(self.pop.preference_vec[k], calc_true=True)
            self.choice_prob_dict[self.pop.cluster_id[k]] = p_vec
            self.theoretical_market_share += self.pop.alpha[k] * p_vec

    def simulate(self, num_transaction=TRANS_PER_EXP):
        self.num_transaction = num_transaction
        n = self.ps.num_prod + 1
        self.sim_data = defaultdict(lambda: np.zeros(n))
        self.type_dict = {}

        for i in range(num_transaction):
            t = self.pop.cluster_id[
                int(np.random.choice(self.pop.num_types, 1, p=self.pop.alpha))
            ]
            self.type_dict[i] = t

            purchase_decision = np.random.choice(
                self.ps.num_prod + 1, 1, p=self.choice_prob_dict[t]
            )
            self.sim_data[i][purchase_decision] = 1

        self.transaction_data_matrix = np.array(list(self.sim_data.values()))
        self.simulated_market_share = np.mean(self.transaction_data_matrix, axis=0)
        self.print_simulate_info()

    def print_simulate_info(self):
        self.logger.debug(f"Simulated transactions: {self.num_transaction}")
        ct = Counter(self.type_dict.values())
        type_mat = [
            [ct[cid], float(ct[cid]) / self.num_transaction]
            for cid in self.pop.cluster_id
        ]

        self.logger.debug(
            type_mat,
            caption="Simulated types",
            header=["Number", "Percent"],
            index=self.pop.cid,
        )
        self.logger.debug(
            [self.simulated_market_share],
            header=self.ps.pid + ["Offset"],
            index=["Simulated Market Share"],
        )

    def print_experiment_info(self):
        self.logger.debug(
            [self.ps.prices], header=self.ps.pid, index=["Experiment Price"]
        )

        self.logger.debug(
            [self.theoretical_market_share],
            header=self.ps.pid_off,
            index=["Theoretical Market Share"],
        )
        self.logger.debug(self.choice_prob_dict, header="keys", caption="Ground truth")

    def alpha_beta_estimate(self, alpha, sample_percent=SAMPLE_RATIO):
        num_types = len(alpha)
        num_features = len(self.pop.preference_vec[0])
        n = self.num_transaction * sample_percent

        sub_trans = self.transaction_data_matrix[
            np.random.choice(self.num_transaction, int(n), replace=True)
        ]
        sub_ms = np.mean(sub_trans, axis=0)
        #  print(sub_ms)
        #  self.experiment_info

        def obj(beta):
            y = np.reshape(beta, (num_types, (num_features + 1)))
            alpha = y[:, 0][:, np.newaxis]
            x = y[:, 1:]
            q = np.array([self.ps.choice_prob_vec(i) for i in x])
            calculated_market_share = np.sum(alpha * q, axis=0)
            diff = sub_ms - calculated_market_share
            #  diff = self.simulated_market_share - calculated_market_share
            return np.linalg.norm(diff) ** 2

        flen = num_types * (num_features + 1)
        x0 = map(float, np.random.randint(-5, 6, size=(flen,)))
        lb = np.array([-np.inf] * flen)
        ub = np.array([np.inf] * flen)
        A = np.zeros(flen)
        for i in range(num_types):
            lb[i * (num_features + 1)] = 0
            ub[i * (num_features + 1)] = 1
            A[i * (num_features + 1)] = 1

        bnds = Bounds(lb, ub)
        cons = LinearConstraint(A, 1, 1)

        init = np.array(list(x0))
        np.seterr(all="ignore")
        res = minimize(obj, init, bounds=bnds, constraints=cons, tol=1e-6)
        np.seterr(all="raise")
        #  self.logger.info(beta_hat)
        ans = np.reshape(res.x, (num_types, (num_features + 1)))
        alpha_hat, beta_hat = ans[:, 0], ans[:, 1:]

        return alpha_hat, beta_hat


if __name__ == "__main__":
    cons_data = "consumer.csv"
    prod_data = "product.csv"
    pop = Population.from_data(cons_data)
    ps = Product_Set.from_data(prod_data)

    exp = Experiment(ps, pop)
    exp.simulate()

    #  def beta_estimate(self, alpha):
    #  alpha = np.array(alpha)[:, np.newaxis]
    #  num_types = len(alpha)
    #  num_features = len(self.pop.preference_vec[0])

    #  def obj(beta):
    #  x = np.reshape(beta, (num_types, num_features))
    #  q = np.array([self.ps.choice_prob_vec(i) for i in x])
    #  calculated_market_share = np.sum(alpha * q, axis=0)
    #  diff = self.simulated_market_share - calculated_market_share
    #  return np.linalg.norm(diff) ** 2

    #  x0 = map(float, np.random.randint(-5, 6, size=(num_types * num_features,)))
    #  init = np.array(list(x0))
    #  beta_hat = minimize(obj, init, tol=1e-3)
    #  self.logger.info(beta_hat)

    #  return np.reshape(beta_hat.x, (num_types, num_features))

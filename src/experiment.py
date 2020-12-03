from collections import Counter, defaultdict, namedtuple, OrderedDict
import numpy as np
from sklearn.metrics import pairwise_distances
from tabulate import tabulate
from scipy.optimize import minimize, LinearConstraint, Bounds
from simulate_util import Population, Product_Set
from logging_util import ColoredLog
from settings import VERBOSE, EXP_PER_CYC, SIM_PER_EXP, TRANS_PER_EXP, SAMPLE_RATIO
DEFAULT_VERBOSE = 3

Params = namedtuple("Params", ["alpha", "beta"])
old_settings = np.geterr()
np.seterr(all="raise")
#  np.seterr(**old_settings)


class Params(namedtuple("Params", ["alpha", "beta", "choice"], defaults=[None, None])):
    __slots__ = ()

    @property
    def table(self):
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

        if self.pop.num_type > 1:
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

        if self.pop.num_type > 1:
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


class Simulator(object):
    def __init__(self, product_set, population, num_cons=1000, verbose=DEFAULT_VERBOSE):
        self.logger = ColoredLog(self.__class__.__name__, verbose=verbose)
        self.ps = product_set
        self.pop = population
        self.num_cons = num_cons
        self.simulate_consumer(num_cons)

    def simulate_consumer(self, num_cons):
        self.num_cons = num_cons
        self.type_dict = {}
        self.member = defaultdict(list)

        for i in range(num_cons):
            tp = self.pop.cluster_id[
                int(np.random.choice(self.pop.num_type, 1, p=self.pop.alpha))
            ]
            self.type_dict[i] = tp
            self.member[tp].append(i)

        ct = Counter(self.type_dict.values())
        type_res = [
            [ct[cid], float(ct[cid]) / num_cons]
            for cid in self.pop.cluster_id
        ]
        self.logger.debug(
            type_res,
            caption="Simulated types",
            header=["Number", "Percent"],
            index=self.pop.cid,
        )

    def calculate_groud_truth(self, price):
        K = self.pop.num_type
        choice_prob_dict = {}
        ms = np.zeros(self.ps.num_prod + 1)
        for k in range(K):
            p_vec = self.ps.choice_prob_vec(self.pop.preference_vec[k], price)
            choice_prob_dict[self.pop.cluster_id[k]] = p_vec
            ms += self.pop.alpha[k] * p_vec
        return choice_prob_dict, ms

    def simulate_transaction(self, choice_prob_dict):
        m = self.ps.num_prod + 1
        sim_data = np.zeros((self.num_cons, m))

        for i in range(self.num_cons):
            tp = self.type_dict[i]
            purchase = np.random.choice(
                m, 1, p=choice_prob_dict[tp]
            )
            sim_data[i, purchase] = 1

        return sim_data

    def run_experiments(self, exp_price):
        self.exp_price = exp_price
        self.T = len(exp_price)
        self.data_hist = {}
        self.theoretical_market_share = {}
        self.simulated_market_share = {}
        self.choice_prob_dict = {}

        for t, p in enumerate(self.exp_price):
            choice_prob_dict, ms = self.calculate_groud_truth(p)
            self.choice_prob_dict[t] = choice_prob_dict
            self.theoretical_market_share[t] = ms
            sim_data = self.simulate_transaction(choice_prob_dict)
            self.data_hist[t] = sim_data
            self.simulated_market_share[t] = np.mean(sim_data, axis=0)

            self.logger.info(f"Experiment: {t}; price: {p}")
            self.logger.debug(np.vstack((self.theoretical_market_share[t], self.simulated_market_share[t])), caption=f"Market Share", header = self.ps.pid_off, index=["Thoretical", "Simulated"])

        self.personal_cdf = {i: np.cumsum(np.mean([hist[i] for _, hist in self.data_hist.items()], axis=0)) for i in range(self.num_cons)}


class Sampler(object):

    def __init__(self, cdf_dict, type_dict, member, verbose=DEFAULT_VERBOSE):
        self.cdf_dict = cdf_dict
        self.type_dict = type_dict
        self.member = member
        self.cdf_mat = np.array(list(self.cdf_dict.values()))
        self.score = pairwise_distances(self.cdf_mat, metric=lambda x, y: np.linalg.norm(x-y, ord=np.inf))
        # self.score = [[np.linalg.norm(vi-vj, ord=np.inf) for j, vj in self.cdf_dict.items()] for i, vi in self.cdf_dict.items()]
        self.num_cons = len(self.cdf_dict)
        self.logger = ColoredLog(self.__class__.__name__, verbose=verbose)

    def create_sample(self, seed, num_samp, samp_func):
        samp_prob = [samp_func(sj) for sj in self.score[seed]]
        samp_prob = [s / sum(samp_prob) for s in samp_prob]
        sample = np.random.choice(self.num_cons, size=num_samp, p=samp_prob)
        self.logger.debug(f"{seed}: {self.type_dict[seed]}")
        types = [self.type_dict[i] for i in sample]
        ct = Counter(types)
        print(ct)
        return samp_prob, sample


def loss(x, sim, ss):
    diff = [
            np.mean(v[ss], axis=0) - sim.ps.choice_prob_vec(x, sim.exp_price[t])
            for t, v in sim.data_hist.items()
           ]
    loss = np.sum([np.linalg.norm(d) ** 2 for d in diff])
    return loss


def main(ps, pop, N, K, M, d):
    sim = Simulator(ps, pop, verbose=4)
    sim.type_dict[0]
    sim.run_experiments(np.random.uniform(low=0.5, high=1.5, size=(500, M)))
    sim.run_experiments([[1,1]]*100)
    sim.choice_prob_dict
    sim.theoretical_market_share
    sim.simulated_market_share
    sim.personal_cdf
    np.linalg.norm(sim.personal_cdf[998] - sim.personal_cdf[999])

sampler = Sampler(sim.personal_cdf, sim.type_dict, sim.member, verbose=4)
beta_set = []
seeds = []
estimators = defaultdict(list)

for _ in range(20):
    seed = np.random.randint(sim.num_cons)
    tp = sim.type_dict[seed]
    seeds.append(tp)
    pp, ss = sampler.create_sample(seed, 200, lambda x: max(1-10*x, 1e-4))
    # pp, ss = sampler.create_sample(np.random.randint(sim.num_cons), 200, lambda x: np.exp(-10*x))
    res = minimize(lambda x: loss(x, sim, ss), np.random.rand(2), tol=1e-6)
    print(res)
    beta_set.append(res.x)
    estimators[tp].append(sim.ps.choice_prob_vec(res.x, np.ones(2)))

estimators["A"]
estimators["B"]

sim = Simulator(ps, pop, verbose=4)
all_est = {}

T = 300
for T in np.arange(10, 510, 10):
    sim.run_experiments(np.random.uniform(low=0.5, high=1.5, size=(T, M)))
    sampler = Sampler(sim.personal_cdf, sim.type_dict, sim.member, verbose=4)
    beta_set = []
    seeds = []
    estimators = defaultdict(list)

    for _ in range(20):
        seed = np.random.randint(sim.num_cons)
        tp = sim.type_dict[seed]
        seeds.append(tp)
        pp, ss = sampler.create_sample(seed, 200, lambda x: max(1-10*x, 1e-4))
        res = minimize(lambda x: loss(x, sim, ss), np.random.rand(2), tol=1e-6)
        beta_set.append(res.x)
        estimators[tp].append(sim.ps.choice_prob_vec(res.x, np.ones(2)))
    all_est[T] = estimators

estimators
plt.hist(np.asarray(estimators["A"])[:, 1])

xx = []
yy = defaultdict(list)
for t, dic in all_est.items():
    xx.append(t)
    for tp, vec in pop.preference_dict.items():
        tr = ps.choice_prob_vec(vec, np.ones(2))
        tse = np.mean([np.linalg.norm(tr - ee) ** 2 for ee in dic[tp]])
        yy[tp].append(tse)


for t, e in yy.items():
    plt.plot(xx, e, label=f"Type {t}")
    plt.xlabel("Number of Experiments")
    plt.ylabel("MSE for MNL Estimation")
plt.legend()


def f():
    return



    def print_simulate_info(self):
        self.logger.debug(f"Simulated transactions: {self.num_transaction}")
        self.logger.debug(
            [self.simulated_market_share],
            header=self.ps.pid + ["Offset"],
            index=["Simulated Market Share"],
        )

            # self.personal_cdf[i][purchase_decision]
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
        num_type = len(alpha)
        num_features = len(self.pop.preference_vec[0])
        n = self.num_transaction * sample_percent

        sub_trans = self.transaction_data_matrix[
            np.random.choice(self.num_transaction, int(n), replace=True)
        ]
        sub_ms = np.mean(sub_trans, axis=0)
        #  print(sub_ms)
        #  self.experiment_info

        def obj(beta):
            y = np.reshape(beta, (num_type, (num_features + 1)))
            alpha = y[:, 0][:, np.newaxis]
            x = y[:, 1:]
            q = np.array([self.ps.choice_prob_vec(i) for i in x])
            calculated_market_share = np.sum(alpha * q, axis=0)
            diff = sub_ms - calculated_market_share
            #  diff = self.simulated_market_share - calculated_market_share
            return np.linalg.norm(diff) ** 2

        flen = num_type * (num_features + 1)
        x0 = map(float, np.random.randint(-5, 6, size=(flen,)))
        lb = np.array([-np.inf] * flen)
        ub = np.array([np.inf] * flen)
        A = np.zeros(flen)
        for i in range(num_type):
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
        ans = np.reshape(res.x, (num_type, (num_features + 1)))
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
    #  num_type = len(alpha)
    #  num_features = len(self.pop.preference_vec[0])

    #  def obj(beta):
    #  x = np.reshape(beta, (num_type, num_features))
    #  q = np.array([self.ps.choice_prob_vec(i) for i in x])
    #  calculated_market_share = np.sum(alpha * q, axis=0)
    #  diff = self.simulated_market_share - calculated_market_share
    #  return np.linalg.norm(diff) ** 2

    #  x0 = map(float, np.random.randint(-5, 6, size=(num_type * num_features,)))
    #  init = np.array(list(x0))
    #  beta_hat = minimize(obj, init, tol=1e-3)
    #  self.logger.info(beta_hat)

    #  return np.reshape(beta_hat.x, (num_type, num_features))

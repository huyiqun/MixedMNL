import os
import time
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import pairwise_distances_argmin_min

def log_likelihood(beta, alpha, sim):
    cid = list(sim.type_dict.keys())
    ps = sim.ps
    pop = sim.pop
    exp_price = sim.exp_price[0]
    T = len(sim.exp_price)

    personal_data = {i: np.asarray([sim.data_hist[t][i] for t in range(T)]) for i in cid}
    data = np.asarray([np.transpose(v) for v in personal_data.values()])
    # data.shape

    prob = np.asarray([ps.choice_prob_vec(bb, exp_price) for bb in beta])
    # prob.shape
    Lnt = np.dot(prob, data)
    # Lnt.shape
    logLnt = np.log(Lnt)
    # logLnt.shape
    logKnt = np.sum(logLnt, axis=2)
    # logKnt.shape
    per_class = np.tensordot(np.asarray(alpha), logKnt, axes=(0,0))
    total = np.sum(per_class)
    return total

def em_run(guessK, T, sim):
    print(f"Current Run: guessK={guessK}, T={T}")
    cid = list(sim.type_dict.keys())
    ps = sim.ps
    pop = sim.pop
    p = sim.exp_price[0]
    d = ps.num_feat
    M = ps.num_prod

    init_cl = {}
    np.random.shuffle(cid)
    prop = np.linspace(0, 1, guessK+1)
    ending = [int(np.round(len(cid) * a)) for a in prop]
    for j in range(1, guessK+1):
        init_cl[j-1] = cid[ending[j-1]:ending[j]]

    proc_time = []
    personal_data = {i: np.asarray([sim.data_hist[t][i] for t in range(T)]) for i in cid}
    decision = np.asarray([[np.argwhere(personal_data[i][t]!=0)[0][0] for t in range(T)] for i in cid])

    def obj(b, data):
        diff = [np.mean(data, axis=0) - ps.choice_prob_vec(b, p)]
        return np.sum([np.linalg.norm(d) ** 2 for d in diff])

    alpha = [1/guessK] * guessK
    ss = time.time()
    beta_est = {}
    for k in range(guessK):
        data_k = [personal_data[i][t] for i in init_cl[k] for t in range(T)]
        res = minimize(lambda b: obj(b, data_k), np.random.rand(d), tol=1e-3)
        beta_est[k] = res.x
    proc_time.append(time.time() - ss)


    converged = False
    iter = 0
    max_iter = 20
    data = np.asarray([np.transpose(v) for v in personal_data.values()])
    data.shape
    alpha_prog = [alpha]
    beta_prog = [list(beta_est.values())]
    while not converged and iter < max_iter:
        ss = time.time()
        h_numerator = [
                [
                    alpha[k] * np.prod([
                        ps.choice_prob(beta_est[k], p, decision[i][t])
                        for t in range(T)
                    ])
                    for k in range(guessK)
                ]
                for i in cid
            ]
        h = [[hh/np.sum(h_numerator[i]) for hh in h_numerator[i]] for i in cid]

        h_trans = np.transpose(h)

        alpha = [np.sum(h_trans[k]) / np.sum(h_trans) for k in range(guessK)]
        alpha_prog.append(alpha)

        beta_est = {}
        for k in range(guessK):

            def func(b, k):
                log_original = -np.log(np.dot(ps.choice_prob_vec(b, p), data))
                vf = np.vectorize(lambda x: min(1e3, x))
                stable_log = vf(log_original)
                stable_log.shape
                obj = h_trans[k][:, np.newaxis] * stable_log
                return np.mean(obj)


            res = minimize(lambda x: func(x, k), np.ones(d), tol=1e-4)

            beta_est[k] = res.x

        beta_prog.append(list(beta_est.values()))
        proc_time.append(time.time() - ss)
        print(f"iteration time: {time.time() - ss}")

        gap = [np.linalg.norm(ps.choice_prob_vec(beta_prog[-1][k], p) - ps.choice_prob_vec(beta_prog[-2][k], p))/(M+1) for k in range(guessK)]
        print(f"iter: {iter}, diff: {np.mean(gap)}")
        if np.mean(gap) < 1e-3 and iter > 1:
            converged = True
        iter += 1

    min_dist = np.mean(pairwise_distances_argmin_min([ps.choice_prob_vec(bb, p) for bb in beta_est.values()], [ps.choice_prob_vec(bb, p) for bb in pop.preference_vec])[1])

    return alpha_prog, beta_prog, proc_time, min_dist

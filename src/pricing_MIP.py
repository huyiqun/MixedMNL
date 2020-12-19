from pulp import *
from math import sqrt
from math import log
from math import exp
from math import fabs
import numpy as np
import xlrd
import random  #########to generate random number
# from settings import NUM_PRODUCT

n = args.num_prod
MaxIter = 100
cost = np.zeros((n, 1))
Demand = np.zeros((MaxIter, n + 1))
for j in range(n + 1):
    for i in range(MaxIter):
        Demand[i, j] = (i + 0.5) / 100
Point = np.zeros((MaxIter, n + 1))
for j in range(n):
    for i in range(MaxIter):
        Point[i, j] = -Demand[i, j] * np.log(Demand[i, j])
for i in range(MaxIter):
    Point[i, n] = (1 - Demand[i, n]) * np.log(Demand[i, n])


def optimization_MIP_mixedMNL_pulp(cost, demand, point, alphav, VC):
    # point = pointEst
    MaxIter, n = point.shape
    ########number of customer types
    K, l = alphav.shape
    n = n - 1
    M = MaxIter
    Sample = range(M)  #######return a vector from 0 to M -1, M is sample size
    Market = range(n + 1)  ######return a vector from 0 to n, include outside option
    Product = range(n)  ######return a vector from 0 to n-1, n is the number of products
    Type = range(K)
    ProductEXPERIMENT = [(i, j, k) for i in Product for j in Sample for k in Type]
    MarketEXPERIMENT = [(i, j, k) for i in Market for j in Sample for k in Type]
    MarketEXPERIMENTZ = [(i, j, k) for i in Market for j in range(M - 1) for k in Type]
    MarketType = [(i, k) for i in Market for k in Type]
    # Preapring an Optimization Model
    optimization_MIP_model = LpProblem("optimizationMIPMDM", LpMaximize)
    ############## Defining decision variables
    #####demo: X = LpVariable.dicts('X',range(2), lowBound = 0, upBound = 1, cat = pulp.LpInteger)
    # x[1:(n+1)]>=0
    x = LpVariable.dicts("x", MarketType, lowBound=0)
    # lam[1:(n+1),1:M]>=0
    lam = LpVariable.dicts(
        "lam", MarketEXPERIMENT, lowBound=0
    )  #####auxiliaray variable lambda
    # z[1:(n+1),1:(M-1)],Bin
    z = LpVariable.dicts("z", MarketEXPERIMENTZ, lowBound=0, upBound=1, cat="Integer")
    # FI[1:(n+1)]
    FI = LpVariable.dicts("FI", MarketType, None)
    # delta[1:(n+1)]
    delta = LpVariable.dicts("delta", MarketType, None)
    # optimalpricce[1:(n)]
    optimalprice = LpVariable.dicts("optimalprice", Product, None)
    ################## Setting the objective
    optimization_MIP_model += lpSum(
        alphav[k] * delta[j, k] + alphav[k] * (VC[j, k] - cost[j]) * x[j, k]
        for j in Product
        for k in range(K)
    ) + lpSum(alphav[k] * delta[n, k] for k in range(K))
    # optimization_MIP_model +=  lpSum(alphav[k]*delta[n,k] for k in range(K))

    # Adding constraints
    ###########adding constraints on lambda and z
    for k in range(K):
        for i in range(n + 1):
            optimization_MIP_model += lam[(i, 1, k)] - z[(i, 1, k)] <= 0
            optimization_MIP_model += lam[(i, M - 1, k)] - z[(i, M - 2, k)] <= 0
            for j in range(1, M - 1):
                optimization_MIP_model += (
                    lam[(i, j, k)] - (z[(i, j, k)] + z[(i, j - 1, k)]) <= 0
                )
    #############second type of constraint, adding the relationship between demand and price
    for k in range(K):
        for i in range(n):
            optimization_MIP_model += lpSum(lam[(i, j, k)] for j in range(M)) == 1
            optimization_MIP_model += lpSum(z[(i, j, k)] for j in range(M - 1)) == 1
            optimization_MIP_model += (
                lpSum(lam[(i, j, k)] * demand[(j, i)] for j in range(M)) - x[i, k] == 0
            )
            optimization_MIP_model += (
                lpSum(lam[(i, j, k)] * (point[(j, i)] / demand[j, i]) for j in range(M))
                - FI[i, k]
                == 0
            )
            optimization_MIP_model += (
                lpSum(lam[(i, j, k)] * point[(j, i)] for j in range(M)) - delta[i, k]
                == 0
            )
        optimization_MIP_model += lpSum(lam[(n, j, k)] for j in range(M)) == 1
        optimization_MIP_model += lpSum(z[(n, j, k)] for j in range(M - 1)) == 1
        optimization_MIP_model += (
            lpSum(lam[(n, j, k)] * demand[(j, n)] for j in range(M)) - x[n, k] == 0
        )
        optimization_MIP_model += (
            lpSum(
                lam[(n, j, k)] * (point[(j, n)] / (1 - demand[j, n])) for j in range(M)
            )
            - FI[n, k]
            == 0
        )
        optimization_MIP_model += (
            lpSum(lam[(n, j, k)] * point[j, n] for j in range(M)) - delta[n, k] == 0
        )
        optimization_MIP_model += lpSum(x[j, k] for j in range(n + 1)) == 1
        ################ constraints: bound x
        for j in range(n + 1):
            optimization_MIP_model += x[j, k] <= demand[M - 1, j]
            optimization_MIP_model += x[j, k] >= demand[1, j]
        ################ constraints: get optimal price
        for j in range(n):
            optimization_MIP_model += optimalprice[j] == VC[j, k] + FI[j, k] + FI[n, k]
    # Solving the optimization problem
    optimization_MIP_model.solve()
    # Printing the optimal solutions obtained
    optx = np.zeros((n + 1))
    optdelta = np.zeros((n + 1))
    optimalprice_vec = np.zeros((n))
    for j in range(n + 1):
        for k in range(K):
            optx[j] = optx[j] + alphav[k] * x[j, k].varValue

    for j in range(n):
        optimalprice_vec[j] = optimalprice[j].varValue
    # Get objective value
    opt_obj = optimization_MIP_model.objective
    # print("\nOptimal value: \t%g" % optimization_MIP_model.objVal)
    return optx, optimalprice_vec, opt_obj

#!/usr/bin/env python

import numpy as np
import gurobipy as grb


def risk_LP():
    #     +----+----+----+
    #     | X3 | X4 | X5 |
    #     +----+----+----+
    #     | X0 | X1 | X2 |
    #     +----+----+----+
    # X0: initial position, X4: obstacle, X5: target position

    S_num = 6  # number of states not in T
    action_num = 4  # number of actions
    # P = [[[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1],
    #       [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], # action: up
    #      [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
    #       [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]],  # action: down
    #      [[0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0],
    #       [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0]],  # action: right
    #      [[0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0],
    #       [0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0]],  # action: left
    #       ] # deterministic transition

    P = [[[0.2, 0, 0, 0.8, 0, 0], [0, 0.2, 0, 0, 0.8, 0], [0, 0, 0.2, 0, 0, 0.8],
          [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]], # action: up
         [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
          [0.8, 0, 0, 0.2, 0, 0], [0, 0.8, 0, 0, 0, 0.2], [0, 0, 0.8, 0, 0, 0.2]],  # action: down
         [[0.2, 0.8, 0, 0, 0, 0], [0, 0.2, 0.8, 0, 0, 0], [0, 0, 1, 0, 0, 0],
          [0, 0, 0, 0.2, 0.8, 0], [0, 0, 0, 0, 0.2, 0.8], [0, 0, 0, 0, 0, 1]],  # action: right
         [[1, 0, 0, 0, 0, 0], [0.8, 0.2, 0, 0, 0, 0], [0, 0.8, 0.2, 0, 0, 0],
          [0, 0, 0, 1, 0, 0], [0, 0, 0, 0.8, 0.2, 0], [0, 0, 0, 0, 0.8, 0.2]],  # action: left
          ] # stochastic transition

    c_map = [1, 1, 1, 1, -10, 10] # cost map
    gamma = 0.9 # discount factor
    S0 = 0 # The initial state

    # should be 4 actions and 6 states
    print(len(P))

    model = grb.Model("risk_lp")
    y = model.addVars(S_num, action_num, vtype=grb.GRB.CONTINUOUS, name = 'x')

    x = []
    for s in range(S_num): # compute occupation
        x += [grb.quicksum(y[s,a] for a in range(action_num))]

    xi = [] 
    for sn in range(S_num): # compute incoming occupation
        # from s to s' sum_a x(s, a) P(s,a,s)'
        print(sn)
        xi += [gamma * grb.quicksum(y[s,a] * P[a][s][sn] for a in range(action_num) for s in range(S_num))]

    lhs = [x[i]-xi[i] for i in range(len(xi))]
    rhs = [0] * S_num
    rhs[S0] = 1

    obj = grb.quicksum(y[s, a] * c_map[s] for a in range(action_num)
                        for s in range(S_num))


    print('obj',obj)

    for i in range(S_num):
        model.addConstr(lhs[i] == rhs[i])

    model.setObjective(obj, grb.GRB.MAXIMIZE)
    model.optimize()
    sol = model.getAttr('x', y)
    print(sol)


if __name__ == '__main__':
    risk_LP()
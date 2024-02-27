#!/usr/bin/env python
"""Short title.

Long explaaction_numtion
"""

import gurobipy as grb
# from solvers.occupation_lp import *

def lp_prob():
    #     +----+----+----+
    #     | X3 | X4 | X5 |
    #     +----+----+----+
    #     | X0 | X1 | X2 |
    #     +----+----+----+
    # X0: initial position, X4: obstacle, X5: target position

    ST_num = 5  # number of states not in T
    nT = 1  # number of states in T
    action_num = 4  # number of actions
    P = [[[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1],
          [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]], # action: up
         [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
          [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]],  # action: down
         [[0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0],
          [0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1], [0, 0, 1, 0, 0, 1]],  # action: right
         [[1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0],
          [0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0]],  # action: left
          ] # deterministic transition
    c_map = [0, 0, 0, 0, -10, 10] # cost map
    gamma = 0.9 # discount factor

    T = [5]  # the target state is the last state :)
    S0 = 0  # the initial state
    delta = 0.01

    # should be 4 actions and 6 states
    print(len(P))

    model = grb.Model("prob0")
    y = model.addVars(ST_num, action_num,vtype=grb.GRB.CONTINUOUS,name = 'x')
    s = model.addVars(ST_num,vtype=grb.GRB.CONTINUOUS,name = 'slack')

    x = []
    for s in range(ST_num): # compute occupation
        x += [grb.quicksum(y[s,a] for a in range(action_num))]

    xi = [] 
    for sn in range(ST_num): # compute incoming occupation
        # from s to s' sum_a x(s, a) P(s,a,s)'
        print(sn)
        xi += [gamma * grb.quicksum(y[s,a] * P[a][s][sn] for a in range(action_num) for s in range(ST_num))]

    lhs = [x[i]-xi[i] for i in range(len(xi))]
    rhs = [0] * ST_num
    rhs[S0] = 1

    H = grb.quicksum(delta*y[s, a] for a in range(action_num) for s in range(ST_num)) # ???
    obj = grb.quicksum(y[s, a]*P[a][s][sn]  for a in range(action_num)
                        for s in range(ST_num) for sn in T) - H
    # todo: add cost map
    print('obj',obj)

    for i in range(ST_num):
        model.addConstr(lhs[i] <= rhs[i]) # balance function???

    model.setObjective(obj, grb.GRB.MAXIMIZE)
    model.optimize()
    print(y)


if __name__ == '__main__':
    lp_prob()
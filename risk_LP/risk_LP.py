#!/usr/bin/env python
import gurobipy as grb
import numpy as np

class RiskLP:

    def __init__(self):
        self.state_num = None
        self.action_num = None

    def solve(self, P, c_map, initial_state):
    
        self.action_num = P.shape[1]  # number of actions
        self.state_num = P.shape[0] # number of states not in T
        S0 = initial_state # The initial state
        gamma = 0.6 # discount factor
        model = grb.Model("risk_lp")
        y = model.addVars(self.state_num, self.action_num, vtype=grb.GRB.CONTINUOUS, name = 'x')
    
        x = []
        for s in range(self.state_num): # compute occupation
            x += [grb.quicksum(y[s,a] for a in range(self.action_num))]
    
        xi = []
        for sn in range(self.state_num): # compute incoming occupation
            # from s to s' sum_a x(s, a) P(s,a,s)'
            xi += [gamma * grb.quicksum(y[s,a] * P[s][a][sn] for a in range(self.action_num) for s in range(self.state_num))]
    
        lhs = [x[i]-xi[i] for i in range(len(xi))]
        rhs = [0] * self.state_num
        rhs[S0] = 1
    
        obj = grb.quicksum(y[s, a] * c_map[s]
                           for a in range(self.action_num)
                           for s in range(self.state_num))
    
        for i in range(self.state_num):
            model.addConstr(lhs[i] == rhs[i])
    
        model.setObjective(obj, grb.GRB.MAXIMIZE)
        model.optimize()
        sol = model.getAttr('x', y)
        return sol


    
    def extract(self, occup_dict):
        strategy = np.zeros(self.state_num)
        risk_field = np.zeros(self.state_num)
        for occ in occup_dict.items():
            state = occ[0][0]
            action = occ[0][1]
            prob = occ[1]
            if prob > risk_field[state]:
                risk_field[state] = prob
                strategy[state] = int(action)
        return strategy, risk_field


if __name__ == '__main__':
    P = np.array([[[0.2, 0, 0, 0.8, 0, 0], [0, 0.2, 0, 0, 0.8, 0], [0, 0, 0.2, 0, 0, 0.8],
          [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]], # action: up
         [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
          [0.8, 0, 0, 0.2, 0, 0], [0, 0.8, 0, 0, 0, 0.2], [0, 0, 0.8, 0, 0, 0.2]],  # action: down
         [[0.2, 0.8, 0, 0, 0, 0], [0, 0.2, 0.8, 0, 0, 0], [0, 0, 1, 0, 0, 0],
          [0, 0, 0, 0.2, 0.8, 0], [0, 0, 0, 0, 0.2, 0.8], [0, 0, 0, 0, 0, 1]],  # action: right
         [[1, 0, 0, 0, 0, 0], [0.8, 0.2, 0, 0, 0, 0], [0, 0.8, 0.2, 0, 0, 0],
          [0, 0, 0, 1, 0, 0], [0, 0, 0, 0.8, 0.2, 0], [0, 0, 0, 0, 0.8, 0.2]],  # action: left
          ]) # stochastic transition
    c_map = [1, 1, 1, 1, -10, 10]  # cost map
    LP_prob = risk_LP()
    occ_dict = LP_prob.solve(P, c_map, 0)
    strategy, risk_field = LP_prob.extract(occ_dict)
    print(strategy)
    print(risk_field)
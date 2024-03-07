#!/usr/bin/env python
import gurobipy as grb
import numpy as np


class Abstraction:

    def __init__(self, map_range, map_res):
        self.map_res = map_res
        self.map_shape = None
        self.state_set = self.abs_state(map_range, map_res)
        self.action_set = self.abs_action()

    def abs_state(self, map_range, map_res):
        xbl = 0
        xbu = map_range[0]
        ybl = 0
        ybu = map_range[1]
        grid_x = np.arange(xbl, xbu, map_res[0])
        grid_y = np.arange(ybl, ybu, map_res[1])
        X, Y = np.meshgrid(grid_x, grid_y)
        self.map_shape = (len(grid_x), len(grid_y))
        return np.array([X.flatten(), Y.flatten()]).T

    def abs_action(self):
        vx_set = np.array([-2, -1, 0, 1, 2])
        vy_set = np.array([-2, -1, 0, 1, 2])
        A, B = np.meshgrid(vx_set, vy_set)
        return np.array([A.flatten(), B.flatten()]).T

    def linear(self):
        # based on single integrator
        P = None
        for n in range(len(self.action_set)):
            P_a = None
            for i in range(len(self.state_set)):
                position = (i % self.map_shape[0], int(i / self.map_shape[0]))
                action = self.action_set[n]
                P_a_s = self.transition(position, action)
                P_a = np.vstack((P_a, P_a_s)) if P_a is not None else P_a_s
            P_a = np.expand_dims(P_a, axis=0)
            P = np.vstack((P, P_a)) if P is not None else P_a
        return P


    def transition(self, position, action):
        def action_prob(action):
            if action == -2:
                prob = np.array([0.8, 0.2, 0.0, 0.0, 0.0])
            elif action == -1:
                prob = np.array([0.0, 0.9, 0.1, 0.0, 0.0])
            elif action == 0:
                prob = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
            elif action == 1:
                prob = np.array([0.0, 0.0, 0.1, 0.9, 0.0])
            elif action == 2:
                prob = np.array([0.0, 0.0, 0.0, 0.2, 0.8])
            return prob

        # map = self.state_set.reshape([self.map_shape[1], self.map_shape[0], 2]).transpose((1, 0, 2))
        P_sn = np.zeros(len(self.state_set)).reshape(self.map_shape)

        prob_x = action_prob(action[0])
        prob_y = action_prob(action[1])
        prob_map = np.outer(prob_x, prob_y)
        for m in range(len(prob_x)):
            for n in range(len(prob_y)):
                if (0 <= position[0] + m - 2 <= self.map_shape[0] -1) and (0 <= position[1] + n - 2 <= self.map_shape[1]-1):
                    P_sn[position[0] + m - 2, position[1] + n - 2] = prob_map[m, n]

        return P_sn.flatten(order='F')



if __name__ == '__main__':
    pcpt_range = (14, 20)
    pcpt_res = 2
    dt = 1
    abs_model = Abstraction(pcpt_range, pcpt_res)
    P_sn = abs_model.transition((3, 0), (-1, 2))
    abs_model.linear()
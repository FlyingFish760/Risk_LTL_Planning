#!/usr/bin/env python
import numpy as np
from abstraction.abstraction import Abstraction
from abstraction.abstraction import MDP


class Prod_MDP:
    def __init__(self, map_sys, map_env):
        self.map_sys = map_sys
        self.map_env = map_env
        self.prod_state_set = [(x_sys, x_env)
                           for x_sys in self.map_sys.states
                           for x_env in self.map_env.states]
        self.prod_action_set = [(a_sys, a_env)
                                for a_sys in self.map_sys.actions
                                for a_env in self.map_env.actions]
        state_index_set = np.arange(len(self.prod_state_set))
        action_index_set = np.arange(len(self.prod_action_set))
        label_map = self.gen_labels()
        trans_matrix = self.gen_transitions()
        initial_state = self.get_prod_state_index((self.map_sys.initial_state, self.map_env.initial_state))
        self.MDP = MDP(state_index_set, action_index_set, trans_matrix, label_map, initial_state)

    def gen_transitions(self):
        P = []
        for prod_state in self.prod_state_set:
            x_sys, x_env = prod_state
            P_s = []
            for (a_sys, a_env) in self.prod_action_set:
                next_x_sys_prob = self.map_sys.transitions[x_sys, a_sys, :]
                next_x_env_prob = self.map_env.transitions[a_env, x_env, :]
                P_s_a = np.zeros(len(self.prod_state_set))
                for next_x_sys in range(len(next_x_sys_prob)):
                    for next_x_env in range(len(next_x_env_prob)):
                        next_state_index = self.get_prod_state_index((next_x_sys, next_x_env))
                        P_s_a[next_state_index] = next_x_sys_prob[next_x_sys] * next_x_env_prob[next_x_env]
                P_s.append(P_s_a)
            P.append(P_s)
        return np.array(P)

    def gen_labels(self):
        label_map = []
        for prod_state in self.prod_state_set:
            x_sys, x_env = prod_state
            label_map.append(self.map_sys.labelling[x_sys] + '&' + self.map_env.labelling[x_env])
        return np.array(label_map)

    def get_prod_state_index(self, prod_state):
        return self.prod_state_set.index(prod_state)





if __name__ == '__main__':
    pcpt_range = (20, 20)
    pcpt_res = (5, 5)
    dt = 1
    initial_position = (2, 2)
    label_func = {(15, 20, 15, 20): "t",
                  (5, 15, 5, 10): "o",
                  (10, 20, 0, 20): "c"}
    abs_model = Abstraction(pcpt_range, pcpt_res, initial_position, label_func)
    MDP_sys = abs_model.MDP

    traffic_light = ['g', 'y', 'r']
    state_set = range(len(traffic_light))
    action_set = [0, 1]
    transitions = np.array([[[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]],
                           [[0, 1, 0],
                           [0, 0, 1],
                           [1, 0, 0]]])
    initial_state = 1
    MDP_env = MDP(state_set, action_set, transitions, traffic_light, initial_state)
    prod_mdp = Prod_MDP(MDP_sys, MDP_env)
    print(prod_mdp.MDP.transitions)

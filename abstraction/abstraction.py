#!/usr/bin/env python
import gurobipy as grb
import numpy as np
from abstraction.MDP import MDP
from scipy.stats import norm

class Abstraction:

    def __init__(self, route_size, route_res, speed_range, speed_res, initial_state, label_function):
        '''
        route_size: (l, d), where l is the length of the route and d is the width of the route
        '''
        self.route_size = route_size
        self.route_res = route_res
        self.map_shape = None
        self.speed_range = speed_range
        self.speed_res = speed_res
        self.state_set = self.gen_abs_state(route_size, route_res, speed_range, speed_res)
        self.action_set = self.gen_abs_action()
        trans_matrix = self.gen_transitions()
        label_map = self.gen_labels(label_function)
        self.init_abs_state = [int(initial_state[0]//self.route_res[0]), 
                               int(initial_state[1] // self.route_res[1]), 
                               int(initial_state[3]//self.speed_res)]
        initial_state_index = self.get_state_index(self.init_abs_state)
        state_index_set = np.arange(len(self.state_set))
        action_index_set = np.arange(len(self.action_set))
        self.MDP = MDP(state_index_set, action_index_set, trans_matrix, label_map, initial_state_index)

    def gen_abs_state(self, route_size, route_res, speed_range, speed_res):
        # Set state bounds
        r_bl = 0
        r_bu = int(route_size[0] / route_res[0])
        ey_bl = 0
        ey_bu = int(route_size[1] / route_res[1])
        v_bl = 0   
        v_bu = int(speed_range / speed_res) + 1

        self.map_shape = (r_bu - r_bl, ey_bu - ey_bl, v_bu - v_bl)

        states = []
        for v in range(v_bl, v_bu):
            for ey in range(ey_bl, ey_bu):
                for r in range(r_bl, r_bu):
                    states.append([r, ey, v])

        return np.array(states)


    def gen_abs_action(self):
        # vx_set = np.array([-2, -1, 0, 1, 2])
        # vy_set = np.array([-2, -1, 0, 1, 2])
        # vx_set = np.array([-1, 0, 1])
        # vy_set = np.array([-1, 0, 1])

        move_set = np.array(['l', 'f', 'r'])   # 'l' for 'left', 'f' for 'forward', 'r' for 'right'
        speed_set = np.array(['d', 'c', 'a'])  # 'd' for 'decelerate', 'c' for 'cruise', 'a' for 'accelerate'
        A, B = np.meshgrid(move_set, speed_set)
        return np.array([A.flatten(), B.flatten()]).T

    def gen_transitions(self):
        P = None
        for state in self.state_set:
            P_s = None
            for action in self.action_set:
                P_s_a = self.trans_func(state, action)
                P_s = np.vstack((P_s, P_s_a)) if P_s is not None else P_s_a
            P_s = np.expand_dims(P_s, axis=0)
            P = np.vstack((P, P_s)) if P is not None else P_s
        return P

    def gen_labels(self, label_function):
        label_map = np.array(["_"]*len(self.state_set), dtype=object)
        for state_region, label in label_function.items():
            r_bl, r_bu, ey_bl, ey_bu, v_bl, v_bu = state_region
            
            r_bl_idx = int(np.floor(r_bl / self.route_res[0]))
            r_bu_idx = int(np.ceil(r_bu / self.route_res[0]))
            
            ey_bl_idx = int(np.floor(ey_bl / self.route_res[1]))
            ey_bu_idx = int(np.ceil(ey_bu / self.route_res[1]))
            
            v_bl_idx = int(np.floor(v_bl / self.speed_res))
            v_bu_idx = int(np.ceil(v_bu / self.speed_res))
            
            for n in range(len(self.state_set)):
                if (r_bl_idx <= self.state_set[n, 0] < r_bu_idx and 
                    ey_bl_idx <= self.state_set[n, 1] < ey_bu_idx and 
                    v_bl_idx <= self.state_set[n, 2] < v_bu_idx):
                    label_map[n] = label if label_map[n] == '_' else label_map[n] + label
        
        # def sanity_gen_labels(label_map):
        #     for i, label in enumerate(label_map):
        #         if label != "_":
        #             print(f"state {self.state_set[i]} has label {label}")
        # sanity_gen_labels(label_map)
        
        return label_map

    def get_abs_state(self, system_state):
        abs_state = [int(system_state[0]//self.route_res[0]), 
                     int(system_state[1]//self.route_res[1]), 
                     int(system_state[3]//self.speed_res)]
        return abs_state
    
    def get_state_index(self, abs_state):
        state_index = self.state_set.tolist().index(abs_state)
        return state_index

    # def get_abs_ind_state(self, position):
    #     abs_state = [int(position[0]//self.route_res[0]), int(position[1]//self.route_res[1])]
    #     state_index = self.get_state_index(abs_state)
    #     return state_index, abs_state


    def trans_func(self, state, action):
        def get_move_transitions(move_action):
            """
            Returns combined probability distribution for (r, ey) changes based on move action.
            """
            # Create a 5x5 probability matrix for (δr, δey) changes
            # Index 2,2 represents no change (δr=0, δey=0)
            prob_spatial = np.zeros((5, 5))
            
            if move_action == 'l': 
                # Primary transition: r+1, ey+1 (forward and left)
                prob_spatial[3, 3] = 0.8  # r+1, ey+1
                # Some uncertainty around the main transition
                prob_spatial[3, 2] = 0.1  # r+1, ey+0 (forward only)
                prob_spatial[2, 3] = 0.1  # r+0, ey+1 (left only)
                
            elif move_action == 'f': 
                # Primary transition: r+1, ey+0 (forward only)
                prob_spatial[3, 2] = 0.9  # r+1, ey+0
                # Some uncertainty
                prob_spatial[2, 2] = 0.1  # r+0, ey+0 (no movement)
                # prob_spatial[4, 2] = 0.1 # r+2, ey+0 (more forward)
                
            elif move_action == 'r':
                # Primary transition: r+1, ey-1 (forward and right)
                prob_spatial[3, 1] = 0.8  # r+1, ey-1
                # Some uncertainty around the main transition
                prob_spatial[3, 2] = 0.1  # r+1, ey+0 (forward only)
                prob_spatial[2, 1] = 0.1  # r+0, ey-1 (right only)

            return prob_spatial

        def get_speed_transitions(speed_action):
            """
            Returns probability distribution for velocity changes based on speed action.
            """
            prob_v = np.zeros(5)  # Index 2 represents no change (δv=0)
            
            if speed_action == 'a':
                prob_v[3] = 0.9  # v+1
                prob_v[2] = 0.1  # v+0 (failed acceleration)
                # prob_v[4] = 0.1  # v+2 (over-acceleration)
            elif speed_action == 'c':  
                prob_v[2] = 1.0  # v+0 (no velocity change)
            elif speed_action == 'd':  
                prob_v[1] = 0.9  # v-1
                prob_v[2] = 0.1  # v+0 (failed deceleration)
                # prob_v[0] = 0.1  # v-2 (over-deceleration)
            
            return prob_v
        
        # Initialize probability matrix for next states
        P_sn = np.zeros(len(self.state_set)).reshape(self.map_shape)

        # Get transition probabilities for the action
        move_action, speed_action = action[0], action[1]
        prob_spatial = get_move_transitions(move_action)  # 5x5 matrix for (r, ey)
        prob_v = get_speed_transitions(speed_action)      # 1x5 array for v

        # Apply transitions to the current state
        for i in range(5):  # δr changes
            for j in range(5):  # δey changes
                for k in range(5):  # δv changes
                    if prob_spatial[i, j] > 0 and prob_v[k] > 0:
                        # Calculate next state indices
                        next_r = state[0] + i - 2
                        next_ey = state[1] + j - 2
                        next_v = state[2] + k - 2

                        # Check bounds
                        r_bl = np.min(self.state_set[:, 0])
                        r_bu = np.max(self.state_set[:, 0])
                        ey_bl = np.min(self.state_set[:, 1])
                        ey_bu = np.max(self.state_set[:, 1])
                        v_bl = np.min(self.state_set[:, 2])
                        v_bu = np.max(self.state_set[:, 2])


                        if (r_bl <= next_r <= r_bu) and \
                           (ey_bl <= next_ey <= ey_bu) and \
                           (v_bl <= next_v <= v_bu):
                           # Turn the state indices into the matrix indices (due to some negative states)
                            r_idx = next_r - r_bl
                            ey_idx = next_ey - ey_bl
                            v_idx = next_v - v_bl

                            P_sn[r_idx, ey_idx, v_idx] = prob_spatial[i, j] * prob_v[k]

        # if np.array_equal(state, np.array([0, 2, 0])) and np.array_equal(action, np.array(['l', 'c'])): 
        #     print("T(s, a, s'):", P_sn[1, 3, 1])

        return P_sn.flatten(order='F')


    def update_abs_init_state(self, system_state):
        abs_state = self.get_abs_state(system_state)
        state_index = self.get_state_index(abs_state)
        self.init_abs_state = abs_state
        self.MDP.initial_state = state_index



class Abstraction_2:

    def __init__(self, route_size, route_res):
        self.route_res = route_res
        self.map_shape = None
        self.state_set = self.abs_state(route_size, route_res)
        self.action_set = self.abs_action()


    def abs_state(self, route_size, route_res):
        r_bl = 0
        r_bu = route_size[0]
        ey_bl = 0
        ey_bu = route_size[1]
        grid_x = np.arange(r_bl, r_bu, route_res[0])
        grid_y = np.arange(ey_bl, ey_bu, route_res[1])
        X, Y = np.meshgrid(grid_x, grid_y)
        self.map_shape = (len(grid_x), len(grid_y))
        return np.array([X.flatten(), Y.flatten()]).T

    def abs_action(self):
        vx_set = np.array([-1, 0, 1])
        # vy_set = np.array([-1, 0, 1])
        # vx_set = np.array([-2, -1, 0, 1, 2])
        vy_set = np.array([-2, -1, 0, 1, 2])
        A, B = np.meshgrid(vx_set, vy_set)
        return np.array([A.flatten(), B.flatten()]).T

    def get_state_index(self, abs_state):
        state_index = self.state_set.tolist().index(abs_state)
        return state_index

    def linear(self):
        # based on single integrator
        P = None
        for i in range(len(self.state_set)):
            P_s = None
            for n in range(len(self.action_set)):
                position = (i % self.map_shape[0], int(i / self.map_shape[0]))
                action = self.action_set[n]
                P_s_a = self.transition(position, action)
                P_s = np.vstack((P_s, P_s_a)) if P_s is not None else P_s_a
            P_s = np.expand_dims(P_s, axis=0)
            P = np.vstack((P, P_s)) if P is not None else P_s
        return P


    def transition(self, position, action):
        def action_prob(action, std_dev, size):
            n = int(size / 2)
            x = np.linspace(-n, n, size)
            # gaussian_array = norm.pdf(x, 0, (abs(action) + 1) * std_dev)
            gaussian_array = norm.pdf(x, action,  std_dev)
            gaussian_array /= gaussian_array.sum()
            return gaussian_array

        P_sn = np.zeros(len(self.state_set)).reshape(self.map_shape)
        prob_x = action_prob(action[0], 1.0, 5)
        prob_y = action_prob(action[1],  1.0, 5)
        prob_map = np.outer(prob_x, prob_y)
        k = int(len(prob_x) / 2)

        for m in range(len(prob_x)):
            for n in range(len(prob_y)):
                if (0 <= position[0] + m - k <= self.map_shape[0] -1) and (0 <= position[1] + n - k <= self.map_shape[1]-1):
                    P_sn[position[0] + m - k, position[1] + n - k] = prob_map[m, n]

        return P_sn.flatten(order='F')





if __name__ == '__main__':
    pcpt_range = (50, 10)
    pcpt_res = (5, 2)
    dt = 1
    initial_state = (2, 0, 0, 0)
    speed_range = 80
    speed_res = 10
    label_func = {(15, 20, 1, 3, 0, 80): "t",
                  (5, 15, -3, 1, 0, 80): "o",
                  (15, 20, 2, 3, 0, 80): "r"}

    abs_model = Abstraction(pcpt_range, pcpt_res, speed_range, speed_res, initial_state, label_func)
    MDP = abs_model.MDP
    print("action_set:", abs_model.action_set)
    print("transitions:", abs_model.MDP.transitions)
    print("labelling:", abs_model.MDP.labelling)
    print("initial_state:", abs_model.MDP.initial_state)
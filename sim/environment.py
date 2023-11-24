# Author: Shuhao Qi
# E-mail: s.qi@tue.nl
# Date: July 7, 2023

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from strategy.gameTreeLoad import strategyNet


class GridSimulator():

    def __init__(self, grid_size, obs_pos):
        self.grid_size = grid_size
        self.unit_size = (1, 1)
        self.obs_pos = (obs_pos[0]-1 , obs_pos[1]-1)

    def RTtraj_3d(self, ax, state_seq):
        def generate_cubes(axes, obs_pos, state_seq):
            X, Y, T = np.indices(axes)
            cube_obs = (X == obs_pos[0]) & (Y == obs_pos[1]) & (T >= 0)
            cube_zom = np.array([[[((x+1 == state_seq[t][0]) and (y+1 == state_seq[t][1]))  for t in range(axes[2])]
                                  for y in range(axes[1])] for x in range(axes[0])])
            cube_ego = np.array([[[((x+1 == state_seq[t][2]) and (y+1 == state_seq[t][3]))  for t in range(axes[2])]
                                  for y in range(axes[1])] for x in range(axes[0])])
            return cube_obs, cube_zom, cube_ego

        axes = [self.grid_size[0], self.grid_size[1], len(state_seq)]

        cube_obs, cube_zom, cube_ego = generate_cubes(axes, self.obs_pos, state_seq)
        empty = np.zeros(axes, dtype=bool)
        # combine the objects into a single boolean array
        voxels = empty | cube_zom | cube_obs | cube_ego

        # set the colors of each object
        colors = np.empty(voxels.shape, dtype=object)
        colors[cube_zom] = 'blue'
        colors[cube_obs] = 'red'
        colors[cube_ego] = 'green'

        # plot everything, Note ax = fig.add_subplot(projection='3d')
        # plt.cla()
        ax.voxels(voxels, facecolors=colors, alpha=0.7)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
        ax.set_aspect('equal')

        # plt.show()
        # plt.pause(0.01)


    def RTgrid_2d(self, ax, joint_state):
        plt.cla()
        grid_world = np.zeros(self.grid_size)
        grid_world[joint_state[0]-1, joint_state[1]-1] = 1 # zombie_car
        grid_world[joint_state[2] - 1, joint_state[3] - 1] = 2 # ego_car
        grid_world[self.obs_pos[0], self.obs_pos[1]] = 3 # obstacle

        colors = ['white', 'blue', 'green', 'red']
        cmap = ListedColormap(colors)

        # Create a figure and axis
        x_ticks = np.arange(0, self.grid_size[0], self.unit_size[0]) + self.unit_size[0]/2
        y_ticks = np.arange(0, self.grid_size[1], self.unit_size[1]) + self.unit_size[1]/2
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.grid(True, which='both', color='grey', linewidth=1)

        # Use imshow to display the grid world with the custom color map
        ax.imshow(grid_world.transpose(), cmap=cmap, origin="lower")
        # plt.pause(0.3)

    def zombie_agent(self, zom_state, driving_style='random'):
        go_ahead = False
        if driving_style == 'random':
            go_ahead = random.choices([True, False], weights = [0.3, 0.7], k=1)
        elif driving_style == 'aggressive':
            go_ahead = random.choices([True, False], weights = [0.5, 0.5], k=1)
        elif driving_style == 'conservative':
            go_ahead = random.choices([True, False], weights = [0.2, 0.8], k=1)
        else:
            print("Wrong driving style is selected!")
        if go_ahead[0]:
            zom_state[1] = max(1, zom_state[1]-1)
        return zom_state

    def ego_agent(self, ego_state, action):
        if action == 'up':
            ego_state[1] += 1
        elif action == 'down':
            ego_state[1] -= 1
        elif action == 'left':
            ego_state[0] -= 1
        elif action == 'right':
            ego_state[0] += 1
        elif action != 'keep':
            print("Action input is wrong!")
        return ego_state



if __name__ == '__main__':

    strategy_file = "../prob_examples/output/strategy_2.csv"
    state_file = "../prob_examples/output/state_2.csv"
    init_state = (2, 7, 1, 1)
    goal_state = (1, 7)

    grid_size = (2, 7)
    obs_pos = (1, 3)

    grid_sim = GridSimulator(grid_size, obs_pos)

    strategy = strategyNet(strategy_file, state_file, init_state, goal_state)
    strategy.build_network()
    strategy.extract_tree()
    fig = plt.figure()
    ax_1 =fig.add_subplot(1, 2, 1)
    ax_2 = fig.add_subplot(1, 2, 2, projection='3d')

    state_seq = []
    cur_state_index = strategy.init_index

    while True:
        cur_state = strategy.states_list[cur_state_index]
        zombie_state = [cur_state[0], cur_state[1]]
        ego_state = [cur_state[2], cur_state[3]]
        next_action = strategy.next_action(cur_state_index)

        if next_action:
            ego_state = grid_sim.ego_agent(ego_state, next_action)
        else:
            break
        zombie_state = grid_sim.zombie_agent(zombie_state, driving_style='random')
        cur_state = (zombie_state[0], zombie_state[1], ego_state[0], ego_state[1])
        cur_state_index =  int(strategy.states_list.index(cur_state))

        grid_sim.RTgrid_2d(ax_1, cur_state)
        state_seq.append(cur_state)
        grid_sim.RTtraj_3d(ax_2, state_seq)
        plt.pause(0.3)

    plt.show()





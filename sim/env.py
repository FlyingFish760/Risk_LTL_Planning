import numpy as np
import matplotlib.pyplot as plt


class Environment:

    def __init__(self, unit_size, grid_size, grid_obs):
        self.unit_size = unit_size
        self.grid_size = grid_size
        self.map_size = [grid_size[0] * unit_size[0], grid_size[1] * unit_size[1]]
        self.grid_obs = grid_obs

    def plot_all(self, ax):
        self.plot_boundary_lines(ax)
        self.plot_grid_lines(ax)
        self.plot_obstacle(ax)

    def plot_boundary_lines(self, ax):
        line_y = [n for n in range(self.map_size[1]+1)]
        for n in range(self.grid_size[0]+1):
            line_x = np.ones(len(line_y)) * n * self.unit_size[0]
            if (n == 0) or (n == self.grid_size[0]):
                ax.plot(line_x, line_y, '-', color = 'black')
            else:
                ax.plot(line_x, line_y, '-.', color = 'black')

    def plot_grid_lines(self, ax):
        line_x = [n for n in range( self.map_size[0]+1)]
        for n in range(self.grid_size[1]+1):
            line_y = np.ones(len(line_x)) * n * self.unit_size[1]
            ax.plot(line_x, line_y, '--', color = 'grey',  alpha=0.5)

    def plot_obstacle(self, ax):
        obs_pos = [self.unit_size[0] * (self.grid_obs[0]-1), self.unit_size[1] * (self.grid_obs[1]-1)]
        rect = plt.Rectangle((obs_pos[0], obs_pos[1]), self.unit_size[0],
                             self.unit_size[1], color='red')
        ax.add_patch(rect)

if __name__ == '__main__':
    unit_size = [4, 10]
    grid_size = [2, 8]
    grid_obs = [1, 3]
    env = Environment(unit_size, grid_size, grid_obs)

    fig = plt.figure()
    ax_2 = fig.add_subplot(2, 1, 2)
    ax_1 = fig.add_subplot(2, 1, 1)

    plt.gca().set_aspect(1)
    env.plot_all()
    plt.show()
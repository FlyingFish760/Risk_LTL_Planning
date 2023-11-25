# author: Shuhao Qi
# Email: s.qi@tue.nl
# Date: August 6nd, 2023
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from sim.visualizer import Visualizer
import sim.perception as pept
import sim.simulator as sim
import risk_field as rf
from policy_iteration import PolicyIteration





def main():

    # ---------- Environment Setting ---------------------
    road_size = [12, 80]
    square_obs = [-6, 30, 6, 10]
    params = {"dt": 0.05, "WB": 3.5}
    car_state = np.array([0, 0, np.pi/2])

    # ---------- Realtime States ---------------------
    fig = plt.figure()
    grid = plt.GridSpec(2, 2)
    ax_1 = fig.add_subplot(grid[0:, 0])
    plt.axis('off')
    plt.axis('equal')
    ax_2 = fig.add_subplot(grid[0, 1], projection='3d')
    ax_3 = fig.add_subplot(grid[1, 1])
    ax_3.axis('off')

    vis = Visualizer(ax_1, road_size, square_obs)
    pcpt_range = (30, 20)
    pcpt_res = 1

    while True:

        ax_1.cla()
        ax_2.cla()
        ax_1.set_aspect(1)

        car_pos = car_state[:2]

        pcpt_dic = pept.gen_pcpt_dic(road_size, square_obs)
        cost_map = pept.gen_cost_map(pcpt_dic, car_pos, pcpt_range, pcpt_res)

        X, Y, Z = rf.gen_risk_field(pcpt_range, pcpt_res, -car_state[2] + np.pi/2)
        risk_map = np.multiply(Z.transpose(), cost_map)

        plt.gca().set_aspect(1)
        vis.plot_boundary_lines()
        vis.plot_obstacle()
        vis.plot_perception((car_pos[0], car_pos[1]), pcpt_range)
        vis.plot_car(car_pos[0], car_pos[1], car_state[2], 0)
        ax_2.plot_surface(X, Y, risk_map.transpose(), cmap=cm.coolwarm, linewidth=0, antialiased=False)

        policy_iter = PolicyIteration(pcpt_range, risk_map, discount_factor=0.8)
        optimal_policy = policy_iter.run()
        action = optimal_policy[int(pcpt_range[0]/2)+1, 0]
        print("action:", action)
        car_state = sim.update(car_state, action, params)
        print("state:", car_state)

        plt.pause(0.01)


if __name__ == '__main__':
    main()
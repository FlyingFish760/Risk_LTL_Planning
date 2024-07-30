import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sim.visualizer import Visualizer
import sim.perception as pept
import sim.simulator as sim
from risk_LP import RiskLP
import risk_field.risk_field as rf
from abstraction.abstraction import Abstraction_2


def main():

    # ---------- Environment Setting ---------------------
    map_size = [64, 64]
    # square_obs_list = [[-32, 10, 32, 15], [0, 40, 32, 15]]
    square_obs_list = [[-32, 24, 24, 16]]
    # square_obs_list = [[-32, 20, 0, 0]]
    params = {"dt": 0.1, "WB": 3.5}
    car_state = np.array([0, 0, np.pi/2])

    LP_prob = RiskLP()
    pcpt_res = (4, 4)
    X, Y =  rf.torus_meshgrid(map_size, pcpt_res)
    car_pos = car_state[:2]

    pcpt_dic = pept.gen_pcpt_dic(map_size, square_obs_list)
    cost_map = pept.gen_cost_map(pcpt_dic, car_pos, map_size, pcpt_res)
    cost_map =  cost_map.flatten(order='F')

    abs_model = Abstraction_2(map_size, pcpt_res)
    P_matrix = abs_model.linear()
    init_state = int(map_size[0] / (2 * pcpt_res[0]))
    occ_measure = LP_prob.solve(P_matrix, cost_map, init_state, gamma=0.99)
    optimal_policy, Z = LP_prob.extract(occ_measure)
    Z = Z.reshape((int(map_size[1]/pcpt_res[1]), int(map_size[0]/pcpt_res[0])))
    # Z = np.delete(Z, 0, axis=1)

    # ---------- Visualization ---------------------
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # plt.axis('off')
    # plt.axis('equal')
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.5)

    action_index = optimal_policy[init_state]
    traj_pos = [28, 0]
    action_arrow = np.array([0, 0])
    traj_state = abs_model.get_state_index(traj_pos)
    for i in range(10):
        traj_pos = [int(a) for a in traj_pos + action_arrow]
        state_index = abs_model.get_state_index(traj_pos)
        action_index = optimal_policy[state_index]
        action_vec = abs_model.action_set[int(action_index)]
        action_arrow = np.multiply(action_vec, pcpt_res)
        ax.quiver(traj_pos[0], traj_pos[1], 0.1, action_arrow[0], action_arrow[1], 0)
        plt.plot(traj_pos[0], traj_pos[1], 0.1, 'o', color='r', markersize=2)
    ax.view_init(elev=90, azim=90)
    ax.w_zaxis.line.set_lw(0.)
    ax.set_zticks([])

    action = abs_model.action_set[int(action_index)]
    print("action:", action)
    for i in range(5):
        car_state = sim.update_1(car_state, action, params)
    print("state:", car_state)
    plt.show()


if __name__ == '__main__':
    main()
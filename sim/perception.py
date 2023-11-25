import numpy as np

Inf = 9999


def gen_pcpt_dic(road_size, square_obs):
    pcpt_dic = []
    pcpt_dic.append(([[-Inf, -road_size[0] / 2], [0, Inf]], -5))
    pcpt_dic.append(([[road_size[0] / 2, Inf], [0, Inf]], -5))

    pcpt_dic.append(([[square_obs[0], square_obs[0] + square_obs[2]],
                      [square_obs[1], square_obs[1] + square_obs[3]]], -5))

    # pcpt_dic.append(([[-1, 1], [0, road_size[1]]], 0.1))

    return pcpt_dic



def gen_cost_map(perception_dic, car_pos, perception_range, res):
    n = int(perception_range[0] / (2 * res))
    m = int(perception_range[1] / res)
    cost_map = np.zeros((2 * n, m))

    for (area, cost) in perception_dic:
        # cost_map[int((area[0][0]+dla)/res): int((area[0][1]+dla)/res), int((area[1][0]+dla)/res): int((area[1][1]+dla)/res)] += cost
        relative_area = [[area[0][0] - car_pos[0], area[0][1] - car_pos[0]], [area[1][0] - car_pos[1], area[1][1] - car_pos[1]]]
        cost_map[int(max(np.floor(relative_area[0][0] / res) + n, 0)): int(min(np.ceil(relative_area[0][1] / res) + n, 2*n)),
        int(max(np.floor(relative_area[1][0]/ res), 0)): int(min(np.ceil(relative_area[1][1]/res), m))] = cost
    return cost_map



if __name__ == '__main__':
    road_size = [12, 80]
    square_obs = [-6, 30, 6, 10]
    pcpt_dic = gen_pcpt_dic(road_size, square_obs)

    # pcpt_dic = [([[-Inf, -6], [0, 80]], -5), ([[6, Inf], [0, 80]], -5), ([[-6, 0], [30, 40]], -5)]
    cost_map = gen_cost_map(pcpt_dic, (0.1, 20.2), (20, 20), 2)
    print(cost_map)
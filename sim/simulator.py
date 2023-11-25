import numpy as np
import math


# def f_dyn(q, u, params):
#
#     WB = params['WB']
#     x, y, theta, v = q
#     acc, tan_delta = u
#     q_dot = np.zeros(q.shape)
#     q_dot[0] = v * np.cos(theta)
#     q_dot[1] = v * np.sin(theta)
#     q_dot[2] = v * tan_delta / WB
#     q_dot[3] = acc
#
#     return q_dot

def f_dyn(q, u, params):

    WB = params['WB']
    x, y, theta = q
    v, tan_delta = u
    q_dot = np.zeros(q.shape)
    q_dot[0] = v * np.cos(theta)
    q_dot[1] = v * np.sin(theta)
    q_dot[2] = v * tan_delta / WB

    return q_dot

def update(q, action, params):
    dic = {'-2': 0.2, '-1': 0.1, '0': 0, '1': -0.1, '2': -0.2}
    u = (5, dic[action])
    q_ = q + f_dyn(q, u, params) * params["dt"]
    return q_


def oppo_update(q, u, params, max_vel):
    u[1] = max(-1, min(u[1], 1))
    q_ = q + f_dyn(q, u, params) * params["dt"]
    q_[3] = max(min(max_vel, q_[3]), 0)
    return q_

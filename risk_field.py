import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from policy_iteration import PolicyIteration


def torus_delta(delta_a):
    if abs(delta_a) < 1e-8:
        delta =  1e-8
    else:
        delta = delta_a
    return delta

def torus_arclen(x, y, xv, yv, delta, xc, yc, R):
    mag_u = abs(np.sqrt((xv - xc) ** 2 + (yv - yc) ** 2))
    mag_v = abs(np.sqrt((x - xc) ** 2 + (y - yc) ** 2))
    dot_pro = (xv - xc) * (x - xc) + (yv - yc) * (y - yc)
    costheta = np.clip(dot_pro / (mag_u * mag_v), -1, 1)
    theta_abs = abs(np.arccos(costheta))
    sign_theta = np.sign((xv - xc) * (y - yc) - (x - xc) * (yv - yc))
    theta_pos_neg = np.sign(delta) * sign_theta * theta_abs
    theta = np.remainder(2 * np.pi + theta_pos_neg, 2 * np.pi)
    arc_len = R * theta
    return arc_len

def torus_a(arc_len, par1, dla):
    a_par = par1 * (arc_len - dla) ** 2
    a_par_sign1 = (np.sign(dla - arc_len) + 1) / 2
    a_par_sign2 = (np.sign(a_par) + 1) / 2
    a_par_sign3 = (np.sign(arc_len) + 1) / 2
    a = a_par_sign1 * a_par_sign2 * a_par_sign3 * a_par
    return a


def torus_meshgrid(range, res):
    xbl = 0
    xbu = range[0]
    ybl = 0
    ybu = range[1]
    grid_x = np.arange(xbl, xbu, res)
    grid_y = np.arange(ybl, ybu, res)
    X, Y = np.meshgrid(grid_x, grid_y)
    return X, Y, xbl, xbu, ybl, ybu

def torus_phiv(phiv_a):
    pi2temp = np.ceil(abs(phiv_a / 2 * np.pi)) # how many rotations (for e.g. 6 * pi / 2 * pi = 3)
    phiv = abs(np.remainder(2 * np.pi * pi2temp + phiv_a, 2 * np.pi)) # phiv interms of 0->2 * pi radians
    return phiv

def torus_R(L, delta):
    R = abs(L / np.tan(delta))
    return R

def torus_sigma(arc_len, kexp_1, kexp_2, mcexp, delta, c):
    sigma_1 = (mcexp + kexp_1 * abs(delta)) * arc_len + c # inner
    sigma_2 = (mcexp + kexp_2 * abs(delta)) * arc_len + c # outer
    return sigma_1, sigma_2

def torus_xcyc(xv, yv, phiv, delta, R):
    if delta > 0:
        phil = phiv + np.pi/2
    else:
        phil = phiv - np.pi/2
    xc = R * np.cos(phil) + xv
    yc = R * np.sin(phil) + yv
    return xc, yc

def torus_z(x, y, xc, yc, R, a, sigma1, sigma2):
    dist_R = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
    a_inside = (1 - np.sign(dist_R - R)) / 2
    a_outside = (1 + np.sign(dist_R - R)) / 2
    num = -(np.sqrt((x - xc) ** 2 + (y - yc) ** 2) - R) ** 2
    den1 = 2 * sigma1 ** 2
    zpure1 = a * a_inside * np.exp(num / den1)
    den2 = 2 * sigma2 ** 2
    zpure2 = a * a_outside * np.exp(num / den2)
    zpure = zpure1 + zpure2
    return zpure


def gen_cost_map(perception_dic, dla, res):
    n = int(dla / res)
    cost_map = np.zeros((2 * n, n))

    for (area, cost) in perception_dic:
        # cost_map[int((area[0][0]+dla)/res): int((area[0][1]+dla)/res), int((area[1][0]+dla)/res): int((area[1][1]+dla)/res)] += cost
        cost_map[int((area[0][0] + dla) / res): int((area[0][1] + dla) / res),
        int(area[1][0]/ res): int(area[1][1] / res)] += cost
    return cost_map


# Constants
Sr = 54
L = 5 # wheel-base length
res = 5
xv = 0
yv = 0

## Car States
v = 5 # speed

steer_angle = 0 # steering angle

## DRF parameters
tla = 10 # look-ahead time
par1 = 0.1 # peek value [0, 0.5]
mcexp = 1
kexp1 = 1 * 0.5 # inside circle
kexp2 = 5 * 0.5 # outside circle
cexp = 1
dla = 50 # perception size

def gen_risk_field(range, res, phi):
    delta_fut_h = (np.pi / 180) *  steer_angle / Sr
    phiv_a = (np.pi / 180) * phi
    delta = torus_delta(delta_fut_h) # steering angle
    phiv = torus_phiv(phiv_a) #
    R = torus_R(L,delta) # arc radius
    xc, yc = torus_xcyc(xv,yv,phiv,delta,R)
    X, Y, xbl, xbu, ybl, ybu = torus_meshgrid(range, res)
    arc_len = torus_arclen(X,Y,xv,yv,delta,xc,yc,R)
    a = torus_a(arc_len, par1, dla)
    sigma_1, sigma_2 = torus_sigma(arc_len, kexp1, kexp2, mcexp, delta, cexp)
    Z = torus_z(X, Y, xc, yc, R, a, sigma_1, sigma_2)
    return X, Y, Z


if __name__ == '__main__':
    pcpt_range = (20, 20)
    pcpt_res = 1

    X, Y, Z = gen_risk_field(pcpt_range, pcpt_res)
    # perception_list = [([[-dla, -20], [-dla, dla]], -5), ([[20, dla], [-dla, dla]], -5), ([[-10, 10], [-dla, dla]], 10)]
    perception_list = [([[-dla, -20], [0, dla]], -5), ([[20, dla], [0, dla]], -5)]

    cost_map = gen_cost_map(perception_list, dla, res)
    risk_map = np.multiply(Z, cost_map)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, risk_map, cmap=cm.coolwarm, linewidth = 0, antialiased = False)
    plt.show()

    policy_iter = PolicyIteration(pcpt_range, risk_map, discount_factor =1.0)
    optimal_policy = policy_iter.run()







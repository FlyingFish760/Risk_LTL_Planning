import numpy as np
import math
import matplotlib.pyplot as plt

PI = np.pi

class Car_Para:

    # parameters for vehicle
    K_SIZE = 1.0
    RF = 4.5 * K_SIZE  # [m] distance from rear to vehicle front end of vehicle
    RB = 1.0 * K_SIZE  # [m] distance from rear to vehicle back end of vehicle
    W = 3.0 * K_SIZE  # [m] width of vehicle
    WD = 0.7 * W  # [m] distance between left-right wheels
    WB = 3.5 * K_SIZE  # [m] Wheel base
    TR = 0.5 * K_SIZE  # [m] Tyre radius
    TW = 1 * K_SIZE  # [m] Tyre width

    steer_max = np.deg2rad(45.0)  # max steering angle [rad]
    steer_change_max = np.deg2rad(30.0)  # maximum steering speed [rad/s]
    speed_max = 30  # maximum speed [m/s]
    speed_min = -20.0 # minimum speed [m/s]
    acceleration_max = 20.0  # maximum acceleration [m/s2]


class Visualizer:

    def __init__(self, ax, road_size, square_obs):
        self.road_size = road_size
        self.square_obs = square_obs
        self.ax = ax

    def plot_all(self, ax):
        self.plot_boundary_lines()
        self.plot_obstacle()

    def plot_boundary_lines(self):

        line_y = [n for n in range(self.road_size[1]+1)]

        line_x = - np.ones(len(line_y)) * self.road_size[0] / 2
        self.ax.plot(line_x, line_y, '-', color='black')
        line_x = np.ones(len(line_y)) * self.road_size[0] / 2
        self.ax.plot(line_x, line_y, '-', color='black')
        line_x = np.zeros(len(line_y))
        self.ax.plot(line_x, line_y, '-.', color='black')

    def plot_obstacle(self):
        rect = plt.Rectangle((self.square_obs[0], self.square_obs[1]), self.square_obs[2],
                             self.square_obs[3], color='red')
        self.ax.add_patch(rect)

    def plot_perception(self, pos, range):
        rect = plt.Rectangle((pos[0] - range[0]/2, pos[1]), range[0], range[1], color='grey', alpha=0.5)
        self.ax.add_patch(rect)


    def plot_car(self, x, y, yaw, steer, color='black'):
        C = Car_Para()
        car = np.array([[-C.RB, -C.RB, C.RF, C.RF, -C.RB],
                        [C.W / 2, -C.W / 2, -C.W / 2, C.W / 2, C.W / 2]])

        wheel = np.array([[-C.TR, -C.TR, C.TR, C.TR, -C.TR],
                          [C.TW / 4, -C.TW / 4, -C.TW / 4, C.TW / 4, C.TW / 4]])

        rlWheel = wheel.copy()
        rrWheel = wheel.copy()
        frWheel = wheel.copy()
        flWheel = wheel.copy()

        Rot1 = np.array([[math.cos(yaw), -math.sin(yaw)],
                         [math.sin(yaw), math.cos(yaw)]])

        Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                         [-math.sin(steer), math.cos(steer)]])

        frWheel = np.dot(Rot2, frWheel)
        flWheel = np.dot(Rot2, flWheel)

        frWheel += np.array([[C.WB], [-C.WD / 2]])
        flWheel += np.array([[C.WB], [C.WD / 2]])
        rrWheel[1, :] -= C.WD / 2
        rlWheel[1, :] += C.WD / 2

        frWheel = np.dot(Rot1, frWheel)
        flWheel = np.dot(Rot1, flWheel)

        rrWheel = np.dot(Rot1, rrWheel)
        rlWheel = np.dot(Rot1, rlWheel)
        car = np.dot(Rot1, car)

        frWheel += np.array([[x], [y]])
        flWheel += np.array([[x], [y]])
        rrWheel += np.array([[x], [y]])
        rlWheel += np.array([[x], [y]])
        car += np.array([[x], [y]])

        def arrow(x, y, theta, L, c):
            angle = np.deg2rad(30)
            d = 0.3 * L
            w = 2

            x_start = x
            y_start = y
            x_end = x + L * np.cos(theta)
            y_end = y + L * np.sin(theta)

            theta_hat_L = theta + PI - angle
            theta_hat_R = theta + PI + angle

            x_hat_start = x_end
            x_hat_end_L = x_hat_start + d * np.cos(theta_hat_L)
            x_hat_end_R = x_hat_start + d * np.cos(theta_hat_R)

            y_hat_start = y_end
            y_hat_end_L = y_hat_start + d * np.sin(theta_hat_L)
            y_hat_end_R = y_hat_start + d * np.sin(theta_hat_R)

            self.ax.plot([x_start, x_end], [y_start, y_end], color=c, linewidth=w)
            self.ax.plot([x_hat_start, x_hat_end_L],
                    [y_hat_start, y_hat_end_L], color=c, linewidth=w)
            self.ax.plot([x_hat_start, x_hat_end_R],
                    [y_hat_start, y_hat_end_R], color=c, linewidth=w)

        self.ax.plot(car[0, :], car[1, :], color)
        self.ax.plot(frWheel[0, :], frWheel[1, :], color)
        self.ax.plot(rrWheel[0, :], rrWheel[1, :], color)
        self.ax.plot(flWheel[0, :], flWheel[1, :], color)
        self.ax.plot(rlWheel[0, :], rlWheel[1, :], color)
        arrow(x, y, yaw, C.WB * 0.8, color)




if __name__ == '__main__':
    road_size = [12, 80]
    square_obs = [-6, 30, 6, 10]

    fig = plt.figure()
    ax_1 = fig.add_subplot(1, 2, 1)
    plt.axis('off')
    plt.axis('equal')
    ax_2 = fig.add_subplot(1, 2, 2)
    vis = Visualizer(ax_1, road_size, square_obs)

    car_pos = (-3, 10)

    plt.gca().set_aspect(1)
    vis.plot_boundary_lines()
    vis.plot_obstacle()
    vis.plot_perception((car_pos[0] - 15, car_pos[1]), (30,30))
    vis.plot_car(car_pos[0], car_pos[1], PI/2, 0)
    plt.show()
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PI = np.pi


class Car_Para:

    # parameters for vehicle
    K_SIZE = 0.5
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

    def __init__(self, ax):
        self.ax = ax
        # Velocity parameters for labeling
        # self.max_speed = 20
        self.speed_res = 10
        
    def set_velocity_params(self, speed_res):
        """Set velocity discretization parameters for velocity state labeling"""
        # self.max_speed = max_speed
        self.speed_res = speed_res
    
    # def get_velocity_range(self, velocity_state):
    #     """
    #     Get the continuous velocity range for a given discrete velocity state.
    #     Returns tuple (v_min, v_max) in actual velocity units.
    #     """
    #     v_min = velocity_state * self.speed_res
    #     v_max = (velocity_state + 1) * self.speed_res
    #     return (v_min, v_max)
    
    def plot_road(self, road_size):

        line_y = [n for n in range(road_size[1]+1)]
        line_x = - np.ones(len(line_y)) * road_size[0] / 2
        self.ax.plot(line_x, line_y, '-', color='black')
        line_x = np.ones(len(line_y)) * road_size[0] / 2
        self.ax.plot(line_x, line_y, '-', color='black')
        line_x = np.zeros(len(line_y))
        self.ax.plot(line_x, line_y, '-.', color='black')

    # def plot_grid(self, region_size, region_res, label_func, traffic_light=None):
    #     grid_x = np.arange(0, region_size[0]+1, region_res[0])
    #     grid_y = np.arange(0, region_size[1]+1, region_res[1])
    #     for x in grid_x:
    #         self.ax.plot([x, x], [0, region_size[1]], color='black')
    #     for y in grid_y:
    #         self.ax.plot([0, region_size[0]], [y, y], color='black')

    #     for region, label in label_func.items():
    #         xbl, xbu, ybl, ybu, _, _ = region
            
                
    #         xbl = max(0, xbl)
    #         xbu = min(region_size[0], xbu)
    #         ybl = max(0, ybl)
    #         ybu = min(region_size[1], ybu)
    #         if label == 'c':
    #             c = 'green' if traffic_light == 0 else 'red'
    #             a = 0.1
    #         elif label == 'v1':
    #             c = 'yellow'
    #             a = 0.2
    #         elif label == 'v0':
    #             c = 'yellow'
    #             a = 1
    #         elif label == 'o':
    #             c = 'grey'
    #             a = 1
    #         elif label == 't':
    #             c = 'blue'
    #             a = 1
    #         self.ax.set_xlim(0, region_size[0])
    #         self.ax.set_ylim(0, region_size[1])
    #         rect = plt.Rectangle((xbl, ybl), xbu-xbl, ybu-ybl, color=c, alpha=a)
    #         self.ax.add_patch(rect)


    def plot_obstacle(self, square_obs_list):
        for obs in square_obs_list:
            rect = plt.Rectangle((obs[0], obs[1]), obs[2], obs[3], color='red')
            self.ax.add_patch(rect)

    def plot_perception(self, pos, range):
        rect = plt.Rectangle((pos[0] - range[0]/2, pos[1]), range[0], range[1], color='grey', alpha=0.5)
        self.ax.add_patch(rect)


    # def plot_car(self, x, y, yaw, steer, color='black'):
    #     C = Car_Para()
    #     car = np.array([[-C.RB, -C.RB, C.RF, C.RF, -C.RB],
    #                     [C.W / 2, -C.W / 2, -C.W / 2, C.W / 2, C.W / 2]])

    #     wheel = np.array([[-C.TR, -C.TR, C.TR, C.TR, -C.TR],
    #                       [C.TW / 4, -C.TW / 4, -C.TW / 4, C.TW / 4, C.TW / 4]])

    #     rlWheel = wheel.copy()
    #     rrWheel = wheel.copy()
    #     frWheel = wheel.copy()
    #     flWheel = wheel.copy()

    #     Rot1 = np.array([[math.cos(yaw), -math.sin(yaw)],
    #                      [math.sin(yaw), math.cos(yaw)]])

    #     Rot2 = np.array([[math.cos(steer), math.sin(steer)],
    #                      [-math.sin(steer), math.cos(steer)]])

    #     frWheel = np.dot(Rot2, frWheel)
    #     flWheel = np.dot(Rot2, flWheel)

    #     frWheel += np.array([[C.WB], [-C.WD / 2]])
    #     flWheel += np.array([[C.WB], [C.WD / 2]])
    #     rrWheel[1, :] -= C.WD / 2
    #     rlWheel[1, :] += C.WD / 2

    #     frWheel = np.dot(Rot1, frWheel)
    #     flWheel = np.dot(Rot1, flWheel)

    #     rrWheel = np.dot(Rot1, rrWheel)
    #     rlWheel = np.dot(Rot1, rlWheel)
    #     car = np.dot(Rot1, car)

    #     frWheel += np.array([[x], [y]])
    #     flWheel += np.array([[x], [y]])
    #     rrWheel += np.array([[x], [y]])
    #     rlWheel += np.array([[x], [y]])
    #     car += np.array([[x], [y]])

    #     def arrow(x, y, theta, L, c):
    #         angle = np.deg2rad(30)
    #         d = 0.3 * L
    #         w = 2

    #         x_start = x
    #         y_start = y
    #         x_end = x + L * np.cos(theta)
    #         y_end = y + L * np.sin(theta)

    #         theta_hat_L = theta + PI - angle
    #         theta_hat_R = theta + PI + angle

    #         x_hat_start = x_end
    #         x_hat_end_L = x_hat_start + d * np.cos(theta_hat_L)
    #         x_hat_end_R = x_hat_start + d * np.cos(theta_hat_R)

    #         y_hat_start = y_end
    #         y_hat_end_L = y_hat_start + d * np.sin(theta_hat_L)
    #         y_hat_end_R = y_hat_start + d * np.sin(theta_hat_R)

    #         self.ax.plot([x_start, x_end], [y_start, y_end], color=c, linewidth=w)
    #         self.ax.plot([x_hat_start, x_hat_end_L],
    #                 [y_hat_start, y_hat_end_L], color=c, linewidth=w)
    #         self.ax.plot([x_hat_start, x_hat_end_R],
    #                 [y_hat_start, y_hat_end_R], color=c, linewidth=w)

    #     self.ax.plot(car[0, :], car[1, :], color)
    #     self.ax.plot(frWheel[0, :], frWheel[1, :], color)
    #     self.ax.plot(rrWheel[0, :], rrWheel[1, :], color)
    #     self.ax.plot(flWheel[0, :], flWheel[1, :], color)
    #     self.ax.plot(rlWheel[0, :], rlWheel[1, :], color)
    #     arrow(x, y, yaw, C.WB * 0.8, color)














    def plot_grid(self, region_size, region_res, label_func, traffic_light=None):
        """
        Draw grid in Frenet frame with (r, ey) axes.
        region_size: [r_max, ey_max]
        region_res: [Δr, Δey]
        label_func: {(r_bl, r_bu, ey_bl, ey_bu, v_bl, v_bu): label}
        """
        # Calculate grid ranges for Frenet frame
        grid_r = np.arange(0, region_size[0]+region_res[0], region_res[0])
        grid_ey = np.arange(-region_size[1]/2, region_size[1]/2+region_res[1], region_res[1])
        
        # Draw vertical grid lines (constant r)
        for r in grid_r:
            self.ax.plot([r, r], [-region_size[1]/2, region_size[1]/2], color='black')
        
        # Draw horizontal grid lines (constant ey)
        for ey in grid_ey:
            self.ax.plot([0, region_size[0]], [ey, ey], color='black')

        # Plot labeled regions
        for region, label in label_func.items():
            r_bl, r_bu, ey_bl, ey_bu, _, _ = region
            
            # Clamp to region bounds
            r_bl = max(0, r_bl)
            r_bu = min(region_size[0], r_bu)
            ey_bl = max(-region_size[1]/2, ey_bl)
            ey_bu = min(region_size[1]/2, ey_bu)
            
            if label == 'c':
                c = 'green' if traffic_light == 0 else 'red'
                a = 0.1
            elif label == 'v1':
                c = 'yellow'
                a = 0.2
            elif label == 'v0':
                c = 'yellow'
                a = 1
            elif label == 'o':
                c = 'grey'
                a = 1
            elif label == 't':
                c = 'blue'
                a = 1
            elif label == 's':
                c = 'red'
                a = 1
            
            # Set axis limits exactly to region bounds
            self.ax.set_xlim(0, region_size[0])
            self.ax.set_ylim(-region_size[1]/2, region_size[1]/2)
            
            rect = plt.Rectangle((r_bl, ey_bl), r_bu-r_bl, ey_bu-ey_bl, color=c, alpha=a)
            self.ax.add_patch(rect)


    def plot_car(self, state, steer, color='black'):
        """
        Plot vehicle in Frenet frame. Position is (r, ey).
        """
        r, ey, yaw, v = state
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

        frWheel += np.array([[r], [ey]])
        flWheel += np.array([[r], [ey]])
        rrWheel += np.array([[r], [ey]])
        rlWheel += np.array([[r], [ey]])
        car += np.array([[r], [ey]])

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
            self.ax.plot([x_hat_start, x_hat_end_L], [y_hat_start, y_hat_end_L], color=c, linewidth=w)
            self.ax.plot([x_hat_start, x_hat_end_R], [y_hat_start, y_hat_end_R], color=c, linewidth=w)

        self.ax.plot(car[0, :], car[1, :], color)
        self.ax.plot(frWheel[0, :], frWheel[1, :], color)
        self.ax.plot(rrWheel[0, :], rrWheel[1, :], color)
        self.ax.plot(flWheel[0, :], flWheel[1, :], color)
        self.ax.plot(rlWheel[0, :], rlWheel[1, :], color)
        arrow(r, ey, yaw, C.WB * 0.8, color)
        
        # Add velocity state label 
        velocity_state = int(v // self.speed_res)
        label_text = f"v{velocity_state}"
        # Position label above the car
        label_r = r
        label_ey = ey + C.W * 0.8  # Position above the car
        self.ax.text(label_r, label_ey, label_text, 
                    ha='center', va='bottom', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))





if __name__ == '__main__':
    # road_size = [12, 80]
    # square_obs = [-6, 30, 6, 10]

    # fig = plt.figure()
    # ax_1 = fig.add_subplot(1, 2, 1)
    # plt.axis('off')
    # plt.axis('equal')
    # ax_2 = fig.add_subplot(1, 2, 2)
    # vis = Visualizer(ax_1, road_size, square_obs)

    # car_pos = (-3, 10)

    # plt.gca().set_aspect(1)
    # vis.plot_boundary_lines()
    # vis.plot_obstacle()
    # vis.plot_perception((car_pos[0] - 15, car_pos[1]), (30,30))
    # vis.plot_car(car_pos[0], car_pos[1], PI/2, 0)
    # plt.show()

    # --- Setup ---
    region_size = [60, 7.5]         # [r_max, ey_max] in meters (Frenet)
    region_res = [5, 1.5]         # grid resolution along r and ey
    max_speed = 20
    speed_res = 10

    # Label map in (r_low, r_high, ey_low, ey_high, v_low, v_high)
    label_func = {
        (10, 20, 0, 1.5, 0, 30): "o",  # Overspeed region
        (50, 60, -1.5, 1.5, 0, 30): "t"  # Target region
    }

    # # Vehicle state in Frenet frame: r, ey, yaw, steer
    # r = 30
    # ey = 1.5
    # yaw = 0     # Assume aligned with road
    # steer = 0   # Straight wheel
    
    # Test cars with different velocity states
    test_cars = [
        {'state': [15, 0.5, 0, 0], 'steer': 0},        # v0 (0-10 m/s)
        {'state': [30, -1.0, np.pi/6, 5], 'steer': 0.5}, # v1 (10-20 m/s)
        {'state': [45, 1.0, -np.pi/6, 10], 'steer': -0.5}, # v2 (20-30 m/s)
    ]

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.axis('off')
    plt.axis('equal')
    vis = Visualizer(ax)
    
    # Set velocity parameters
    vis.set_velocity_params(speed_res)

    # Plot grid and regions
    vis.plot_grid(region_size, region_res, label_func)
    # vis.plot_car(r, ey, yaw, steer, color='black')
    
    # Plot cars with velocity state labels
    for car in test_cars:
        vis.plot_car(car['state'], car['steer'])

    ax.set_title("Frenet Frame Visualization (r, $e_y$)")
    # ax.set_xlabel("Longitudinal Position r [m]")
    # ax.set_ylabel("Lateral Deviation $e_y$ [m]")
    plt.grid(True)
    plt.axis("equal")
    plt.show()
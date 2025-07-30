import casadi as ca
import numpy as np

class MPC:

    def __init__(self, car_para, horizon_steps):
        self.x_dim = 4
        self.u_dim = 2
        self.v_bound = [0, 100]
        self.a_bound = [-50, 50]
        self.delta_bound = [-1, 1]
        self.dt = car_para['dt']
        self.WB = car_para['WB']
        self.car_para = car_para
        self.horizon = horizon_steps
        self.opti = ca.Opti()
        self.X = None
        self.U = None
        self.X_ref = self.opti.parameter(3)   # (xref, yref, vref)
        self.X_0 = self.opti.parameter(self.x_dim)
        self.prob_init()

    def prob_init(self):
        # Define the state and control variables
        x = ca.SX.sym('x')  # x position
        y = ca.SX.sym('y')  # y position
        theta = ca.SX.sym('theta')  # orientation angle 
        v = ca.SX.sym('v')  # velocity 
        a = ca.SX.sym('a')  # acceleration 
        delta = ca.SX.sym('delta')  # steering angle 

        states = ca.vertcat(x, y, theta, v)  
        controls = ca.vertcat(a, delta)  

        # Bicycle abstraction dynamics
        xdot = v * ca.cos(theta)
        ydot = v * ca.sin(theta)
        thetadot = v / self.WB * ca.tan(delta)
        vdot = a  

        # Control horizon
        state_dot = ca.vertcat(xdot, ydot, thetadot, vdot)

        # Function to integrate dynamics over each interval
        integrate_f = ca.Function('integrate_f', [states, controls], [state_dot])

        # Objective function and constraints
        self.X = self.opti.variable(self.x_dim, self.horizon + 1)  # state trajectory
        self.U = self.opti.variable(self.u_dim, self.horizon)  # control trajectory

        cost = 0
        # Setup cost function and constraints
        for k in range(self.horizon):
            # Cost function (minimize distance to reference point + velocity tracking)
            cost = cost + (self.X[0, k] - self.X_ref[0]) ** 2 + \
                    (self.X[1, k] - self.X_ref[1]) ** 2 + \
                    (self.X[3, k] - self.X_ref[2]) ** 2

            # System dynamics as constraints
            st_next = self.X[:, k] + integrate_f(self.X[:, k], self.U[:, k]) * self.dt
            self.opti.subject_to(self.X[:, k + 1] == st_next)
            
            # Control bounds
            self.opti.subject_to(self.U[0, k] > self.a_bound[0])  
            self.opti.subject_to(self.U[0, k] < self.a_bound[1])   
            self.opti.subject_to(self.U[1, k] > self.delta_bound[0])
            self.opti.subject_to(self.U[1, k] < self.delta_bound[1])
            
            # State bounds (velocity bounds)
            self.opti.subject_to(self.X[3, k] > self.v_bound[0])  # velocity lower bound
            self.opti.subject_to(self.X[3, k] < self.v_bound[1])  # velocity upper bound
            
        # Boundary conditions
        self.opti.subject_to(self.X[:, 0] == self.X_0)  # initial condition
        # Solver configuration
        self.opti.minimize(cost)
        opts = {'ipopt': {'print_level': 0}}
        self.opti.solver('ipopt', opts)


    def solve(self, current_state, target_ref):
        self.opti.set_value(self.X_0, current_state)
        self.opti.set_value(self.X_ref, target_ref)
        sol = self.opti.solve()

        optimal_states = sol.value(self.X)
        optimal_controls = sol.value(self.U)
        return optimal_controls[:, 0]



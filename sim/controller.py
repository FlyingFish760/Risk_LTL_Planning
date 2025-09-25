import casadi as ca
import numpy as np



class MPC:

    def __init__(self, car_para, horizon_steps):
        self.x_dim = 4
        self.u_dim = 2

        self.v_bound = [0, 8]
        self.ephi_bound = [-np.pi / 2, np.pi / 2]
        
        self.a_bound = [-2, 2]
        self.deltaphi_bound = [-1, 1]  

        self.dt = car_para['dt']
        self.WB = car_para['WB']
        self.car_para = car_para
        self.horizon = horizon_steps
        self.opti = ca.Opti()
        self.X = None
        self.U = None
        self.X_ref = self.opti.parameter(3)   # (r_ref, ey_ref, v_ref)
        self.X_0 = self.opti.parameter(self.x_dim)
        self.prob_init()

    def prob_init(self):
        # Define the state and control variables
        r = ca.SX.sym('r')  
        ey = ca.SX.sym('ey')  
        ephi = ca.SX.sym('ephi')  
        v = ca.SX.sym('v') 
        
        a = ca.SX.sym('a')  
        deltaphi = ca.SX.sym('deltaphi')  

        states = ca.vertcat(r, ey, ephi, v)  
        controls = ca.vertcat(a, deltaphi)  

        # Bicycle abstraction dynamics
        # Assume road curvature κ(t) = 0 for straight road (can be parameterized later)
        kappa = 0  # road curvature
        
        rdot = v * ca.cos(ephi) / (1 - ey * kappa)
        eydot = v * ca.sin(ephi)
        ephidot = v * (ca.tan(deltaphi) / self.WB - kappa * ca.cos(ephi) / (1 - kappa * ey))
        vdot = a

        # Control horizon
        state_dot = ca.vertcat(rdot, eydot, ephidot, vdot)

        # Function to integrate dynamics over each interval
        integrate_f = ca.Function('integrate_f', [states, controls], [state_dot])

        # Objective function and constraints
        self.X = self.opti.variable(self.x_dim, self.horizon + 1)  # state trajectory
        self.U = self.opti.variable(self.u_dim, self.horizon)  # control trajectory

        cost = 0
        # Setup cost function and constraints
        for k in range(self.horizon):
            # State tracking cost
            Q = ca.DM([1.0, 1.0, 2.0])
            Q_rate = 5
            Q *= Q_rate
            cost = cost + Q[0] * (self.X[0, k] - self.X_ref[0]) ** 2 + \
                    Q[1] * (self.X[1, k] - self.X_ref[1]) ** 2 + \
                    Q[2] * (self.X[3, k] - self.X_ref[2]) ** 2
            
            # Control input cost
            R = ca.DM([1.0, 5.0])
            R_rate = 1
            R *= R_rate
            cost = cost + R[0] * self.U[0, k] ** 2 + \
                    R[1] * self.U[1, k] ** 2

            # System dynamics as constraints
            st_next = self.X[:, k] + integrate_f(self.X[:, k], self.U[:, k]) * self.dt
            self.opti.subject_to(self.X[:, k + 1] == st_next)
            
            # Control bounds
            self.opti.subject_to(self.U[0, k] > self.a_bound[0])  
            self.opti.subject_to(self.U[0, k] < self.a_bound[1])  
            self.opti.subject_to(self.U[1, k] > self.deltaphi_bound[0])  
            self.opti.subject_to(self.U[1, k] < self.deltaphi_bound[1])  
            
            # State bounds 
            self.opti.subject_to(self.X[2, k + 1] > self.ephi_bound[0])  
            self.opti.subject_to(self.X[2, k + 1] < self.ephi_bound[1])  
            self.opti.subject_to(self.X[3, k + 1] > self.v_bound[0]) 
            self.opti.subject_to(self.X[3, k + 1] < self.v_bound[1]) 
            
        # Boundary conditions
        self.opti.subject_to(self.X[:, 0] == self.X_0)  # initial condition
        # Solver configuration
        self.opti.minimize(cost)
        # opts = {'ipopt': {'print_level': 0}}
        opts = {
            "ipopt": {"print_level": 0},
            "print_time": 0,     
            "verbose": False    
        }
        self.opti.solver('ipopt', opts)


    def solve(self, current_state, target_ref):
        self.opti.set_value(self.X_0, current_state)
        self.opti.set_value(self.X_ref, target_ref)
        sol = self.opti.solve()

        optimal_states = sol.value(self.X)
        optimal_controls = sol.value(self.U)
        return optimal_controls[:, 0]



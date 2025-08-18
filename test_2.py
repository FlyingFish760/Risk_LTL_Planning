# author: Shuhao Qi
# Email: s.qi@tue.nl
# Date: January 6nd, 2025
import matplotlib.pyplot as plt
import numpy as np

from abstraction.MDP import MDP
from sim.visualizer import Visualizer
import sim.simulator as sim
from risk_LP.ltl_risk_LP import Risk_LTL_LP
from specification.prod_auto import Product
from specification.specification import LTL_Spec
from abstraction.abstraction import Abstraction
from sim.controller import MPC
from abstraction.prod_MDP import Prod_MDP


SPEED_LOW_BOUND = 10


def dyn_labelling(sta_labels, speed_sign):
    labels = sta_labels.copy()

    # g_x = int(np.floor(v_pos[0] / region_res[0]))   # Convert postion of opposite vehicle to grid index
    # g_y = int(np.floor(v_pos[1] / region_res[1]))
    # v_1 = ((g_x + min(0, v_action[0])) * region_res[0],   # The next cell the opponent is moving into
    #         (g_x + max(1, 1 + v_action[0])) * region_res[0],
    #         (g_y + min(0, v_action[1])) * region_res[1], 
    #         (g_y + max(1, 1 + v_action[1])) * region_res[1])
    # v_0 = (g_x * region_res[0],    # Current opponent cell (this is the immediate danger zone)
    #         (g_x + 1) * region_res[0], 
    #         g_y * region_res[1], 
    #         (g_y + 1) * region_res[1])
    # labels[v_0] = "v0"
    # labels[v_1] = "v1"

    speed_limit, speed_pos = speed_sign

    return labels

def cal_target_ref(target_abs_state, region_res, speed_res):
    target_ref_0 = target_abs_state[0] * region_res[0] + region_res[0] / 2
    target_ref_1 = target_abs_state[1] * region_res[1] + region_res[1] / 2 
    target_ref_2 = target_abs_state[2] * speed_res + speed_res / 2
    return np.array([target_ref_0, target_ref_1, target_ref_2])

def action_to_state_transition(action):
    '''
    action: (m, s). m is the movement action, s is the speeding action. 
    m in {'l', 'r', 'f'}, s in {'a', 'd', 'c'}.

    return: delta_state = (delta_r, delta_ey, delta_v)
    '''
    delta_state = np.array([0, 0, 0])

    # movement action
    delta_state[0] = 1
    if action[0] == 'l':
        delta_state[1] = 1
    elif action[0] == 'r':
        delta_state[1] = -1
    elif action[0] == 'f':
        delta_state[1] = 0

    # speeding action
    if action[1] == 'a':
        delta_state[2] = 1
    elif action[1] == 'd':
        delta_state[2] = -1
    elif action[1] == 'c':
        delta_state[2] = 0

    return delta_state
        

def plot_occ_measure(occ_measure, prod_auto, abs_model):
    '''
    Plot occmeasure of states and the occ measure of actions in each state
    '''

    ######## Plot positional occupation measure ########
    state_action_occ = {} 

    for (state_ind, action_ind), occ_value in occ_measure.items():
        x, s_cs, s_s = prod_auto.prod_state_set[state_ind]
        (r, ey, v) = abs_model.state_set[x]
        (m, s) = abs_model.action_set[action_ind]

        # if (r, ey, v) == (0, 2, 0) and (m, s) == ('l', 'd'): print("beta[(0, 2, 0), (l, d)]:", occ_value)
        # elif (r, ey, v) == (0, 2, 0) and (m, s) == ('l', 'c'): print("beta[(0, 2, 0), (l, c)]:", occ_value)
        # elif (r, ey, v) == (0, 2, 0) and (m, s) == ('l', 'a'): print("beta[(0, 2, 0), (l, a)]:", occ_value)
        
        if (r, ey) not in state_action_occ:
            state_action_occ[(r, ey)] = {}
        if m not in state_action_occ[(r, ey)]:
            state_action_occ[(r, ey)][m] = 0
        state_action_occ[(r, ey)][m] += occ_value

    state_occ = {}
    for (r, ey), m_occ in state_action_occ.items():
        if (r, ey) not in state_occ:
            state_occ[(r, ey)] = 0
        state_occ[(r, ey)] += sum(m_occ.values())
    
    
    # Get the dimensions of the grid
    r_max = max([r for (r, ey) in state_occ.keys()]) + 1
    ey_max = max([ey for (r, ey) in state_occ.keys()]) + 1
    
    # Create a grid for the heat map
    heat_map = np.zeros((ey_max, r_max))
    
    # Fill the heat map with occupation values
    for (r, ey), occ_value in state_occ.items():
        heat_map[ey, r] = occ_value
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    im = plt.imshow(heat_map, cmap='hot', interpolation='nearest', origin='lower')
    plt.colorbar(im, label='Occupation Measure')
    
    # Add value labels at each state with better visibility
    max_value = np.max(list(state_occ.values()))
    for (r, ey), occ_value in state_occ.items():
        # Use white text with black outline for better visibility
        plt.text(r, ey-0.3, f'{occ_value:.3f}', ha='center', va='center', 
                color='red', fontsize=10, weight='bold',
                )
    
    # Add action arrows and labels for all actions in each state
    arrow_scale = 0.15
    for (r, ey), action_occ in state_action_occ.items():
        # All arrows start from the same location (center of the cell)
        start_x, start_y = r, ey + 0.2
        
        # Find the action with highest occupation measure for coloring
        max_action_type = max(action_occ.items(), key=lambda x: x[1])[0] if action_occ else None
        
        # Label all actions, not just the maximum
        for action_type, action_value in action_occ.items():
            # Define arrow directions based on action type
            dx, dy = 0, 0
            action_label = ""
            if action_type == 'l':  # left and forward
                dx, dy = arrow_scale, arrow_scale
                action_label = "L"
            elif action_type == 'r':  # right and forward
                dx, dy = arrow_scale, -arrow_scale
                action_label = "R"
            elif action_type == 'f':  # forward
                dx, dy = arrow_scale, 0
                action_label = "F"
            
            # Determine color: blue for max action, red for others
            arrow_color = 'green' if action_type == max_action_type else 'red'
            text_color = 'green' if action_type == max_action_type else 'red'
            
            # Draw arrow if there's a clear direction
            if dx != 0 or dy != 0:
                plt.arrow(start_x, start_y, dx, dy, head_width=0.06, head_length=0.03, 
                         fc=arrow_color, ec=arrow_color, alpha=0.7, linewidth=1.2)
                
                # Add action label and occupation value at the end of the arrow
                plt.text(start_x + dx + 0.05, start_y + dy, f'{action_label}: {action_value:.2f}', 
                        ha='left', va='center', color=text_color, fontsize=6, weight='bold')
    
    plt.xlabel('r (longitudinal position)')
    plt.ylabel('ey (lateral position)')
    plt.title('State Occupation Measure Heat Map with Action Directions')
    plt.show()


    ######## Plot velocity occupation measure ########
    # Get all unique velocity values
    velocity_values = set()
    for (state_ind, action_ind), occ_value in occ_measure.items():
        x, s_cs, s_s = prod_auto.prod_state_set[state_ind]
        (r, ey, v) = abs_model.state_set[x]
        velocity_values.add(v)
    
    velocity_values = sorted(list(velocity_values))
    
    # Create subplots for all velocity values
    n_velocities = len(velocity_values)
    if n_velocities > 0:
        # Calculate subplot layout
        n_cols = min(3, n_velocities)  # Maximum 3 columns
        n_rows = (n_velocities + n_cols - 1) // n_cols  # Ceiling division
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
        if n_velocities == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        
        # For each velocity, create a subplot
        for idx, v_val in enumerate(velocity_values):
            row = idx // n_cols
            col = idx % n_cols
            if n_rows > 1:
                ax = axes[row][col] if isinstance(axes[row], (list, np.ndarray)) else axes[row]
            else:
                ax = axes[col] if isinstance(axes, (list, np.ndarray)) else axes
            
            state_action_occ_velocity = {} 

            for (state_ind, action_ind), occ_value in occ_measure.items():
                x, s_cs, s_s = prod_auto.prod_state_set[state_ind]
                (r, ey, v) = abs_model.state_set[x]
                (m, s) = abs_model.action_set[action_ind]
                
                # Only consider states with the current velocity
                if v != v_val:
                    continue
                    
                if (r, ey) not in state_action_occ_velocity:
                    state_action_occ_velocity[(r, ey)] = {}
                if s not in state_action_occ_velocity[(r, ey)]:
                    state_action_occ_velocity[(r, ey)][s] = 0
                state_action_occ_velocity[(r, ey)][s] += occ_value

            state_occ_velocity = {}
            for (r, ey), s_occ in state_action_occ_velocity.items():
                if (r, ey) not in state_occ_velocity:
                    state_occ_velocity[(r, ey)] = 0
                state_occ_velocity[(r, ey)] += sum(s_occ.values())

            # if v_val == 2: print("beta[(1, 1, 2)]:", state_occ_velocity[(1, 1)])
            
            if not state_occ_velocity:  # Skip if no states for this velocity
                ax.set_title(f'Velocity {v_val} (No Data)')
                ax.axis('off')
                continue
            
            # Get the dimensions of the grid
            r_max = max([r for (r, ey) in state_occ_velocity.keys()]) + 1
            ey_max = max([ey for (r, ey) in state_occ_velocity.keys()]) + 1
            
            # Create a grid for the heat map
            heat_map_velocity = np.zeros((ey_max, r_max))
            
            # Fill the heat map with occupation values
            for (r, ey), occ_value in state_occ_velocity.items():
                heat_map_velocity[ey, r] = occ_value
            
            # Create the heat map on the subplot
            im = ax.imshow(heat_map_velocity, cmap='hot', interpolation='nearest', origin='lower')
            
            # Add value labels at each state with better visibility
            for (r, ey), occ_value in state_occ_velocity.items():
                ax.text(r, ey-0.3, f'{occ_value:.3f}', ha='center', va='center', 
                        color='red', fontsize=8, weight='bold')
            
            # Add speed action labels for all actions in each state
            for (r, ey), action_occ in state_action_occ_velocity.items():
                # Find the action with highest occupation measure for coloring
                if action_occ and any(v > 0 for v in action_occ.values()):
                    max_action_type = max(action_occ.items(), key=lambda x: x[1])[0]
                else:
                    max_action_type = None
                
                # Position labels around the center of the cell
                label_positions = {'a': (0.2, 0.2), 'd': (-0.2, 0.2), 'c': (0, 0.3)}
                
                # Label all speed actions
                for action_type, action_value in action_occ.items():
                    action_label = ""
                    if action_type == 'a':  # accelerate
                        action_label = "A"
                    elif action_type == 'd':  # decelerate
                        action_label = "D"
                    elif action_type == 'c':  # cruise
                        action_label = "C"
                    
                    # Determine color: green for max action, red for others
                    text_color = 'green' if action_type == max_action_type else 'red'
                    
                    # Get label position offset
                    dx, dy = label_positions.get(action_type, (0, 0))
                    
                    # Add action label and occupation value
                    ax.text(r + dx, ey + dy, f'{action_label}: {action_value:.2f}', 
                            ha='center', va='center', color=text_color, fontsize=6, weight='bold')
            
            ax.set_xlabel('r (longitudinal position)')
            ax.set_ylabel('ey (lateral position)')
            ax.set_title(f'Speed Actions at Velocity {v_val}')
        
        # Hide unused subplots
        for idx in range(n_velocities, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            if n_rows > 1:
                axes[row][col].axis('off')
            else:
                if isinstance(axes, (list, np.ndarray)):
                    axes[col].axis('off')
                else:
                    axes.axis('off')
        
        plt.tight_layout()
        plt.show()


    
    





def main():
    # # ---------- MDP Environment  ---------------------
    # # Create a trivial MDP with one state and no actions
    # state_set = [0]
    # action_set = [0]
    # transitions = np.array([[[1.0]]])  # Stay in state 0
    # initial_state = 0
    # mdp_env = MDP(state_set, action_set, transitions, ["e"], initial_state)

    # ---------- MPD System (Abstraction) --------------------
    region_size = (50, 20)
    region_res = (10,4)   # Now region_size[1]/region_res[1] must be odd
    max_speed = 8
    speed_res = 4

    # static_label = {(15, 20, 15, 20): "t"}  
    # label_func = dyn_labelling(static_label, init_oppo_car_pos, [-1, 0])
    # speed_limit = 30

    # label_func = {(40, 50, 6, 10, 0, max_speed): "t",
    #                (20, 30, -2, 2, 0, max_speed): "o"}   # "o" for "obstacle"

    #basic
    # label_func = {(40, 50, 6, 10, 0, max_speed): "t",
    #                (20, 30, -2, 2, 0, max_speed): "o"}   # "o" for "obstacle"
    # label_func = {(40, 50, 16, 20, 0, max_speed): "t",
    #                (20, 30, 8, 12, 0, max_speed): "o"}   # "o" for "obstacle"

    # # speed test 1
    # speed_limit = 2
    # label_func = {(40, 50, 16, 20, 0, max_speed): "t",
    #                (20, 30, 8, 12, 0, max_speed): "o",  # "o" for "obstacle"
    #                (10, 20, 8, 20, speed_limit, max_speed): "s"}   # "s" for "overspeed"

    # speed test 2
    speed_limit = 4
    speed_limit_2 = 4
    label_func = {(40, 50, 16, 20, 0, max_speed): "t",
                   (20, 30, 8, 12, 0, max_speed): "o",  # "o" for "obstacle"
                   (10, 30, 8, 20, speed_limit, max_speed): "s", # "s" for "speed"
                   (30, 40, 8, 20, 0, speed_limit): "s",
                   (0, 50, 0, 20, -speed_res, 0): "n",   # "n" for negative speed 
                   (0, 50, 0, 20, max_speed, max_speed + speed_res): "h"}   # "h" for too high speed
    
    ego_state = np.array([0, 10, np.pi / 2, 3])

    abs_model = Abstraction(region_size, region_res, max_speed, speed_res, ego_state, label_func) 

    # ---------- Specification Define --------------------
    # safe_frag = LTL_Spec("G(~o) & G(~s)", AP_set=['o', 's'])
    safe_frag = LTL_Spec("G(~o) & G(~s) & G(~n) & G(~h)", AP_set=['o', 's', 'n', 'h'])
    scltl_frag = LTL_Spec("F(t)", AP_set=['t'])

    # basic
    # cost_func = {'o': 5, 's': 5}  
    cost_func = {'o': 5, 's': 5, 'n': 7, 'h': 7}  

    # # overspeed test 1
    # cost_func = {'o': 5, 's': 20}  

    # ---------- LP problem --------------------
    LP_prob = Risk_LTL_LP()

    # ---------- MPC --------------------
    params = {"dt": 0.05, "WB": 1.5}
    mpc_con = MPC(params, horizon_steps=5)

    # ---------- Visualization ---------------------
    fig, ax_1 = plt.subplots(1, 1, figsize=(10, 10))
    plt.axis('off')
    plt.axis('equal')
    vis = Visualizer(ax_1)
    vis.set_velocity_params(speed_res)

    # ---------- Initialization --------------------
    # abs_state_env = 0
    abs_state_sys = abs_model.get_abs_state(ego_state) 
    # oppo_abs_state = abs_model.get_abs_state(init_oppo_car_pos)  

    # Initial product MDP state 
    # (the first state does not matter;
    #  the second state is the initial state of the cs DFA;
    #  the third state is the initial state of the safety DFA)
    prod_state = (0, 1, 1)   

    target_abs_state_sys = None
    occ_measure = None
    
    # ---------- Compute optimal policy --------------------
    # abs_model = Abstraction(region_size, region_res, ego_pos, label_func)
    mdp_sys = abs_model.MDP
    # mdp_prod = Prod_MDP(mdp_sys, mdp_env)
    # prod_auto = Product(mdp_prod.MDP, scltl_frag.dfa, safe_frag.dfa)
    prod_auto = Product(mdp_sys, scltl_frag.dfa, safe_frag.dfa)
    P_matrix = prod_auto.prod_transitions
    cost_map = prod_auto.gen_cost_map(cost_func)

    abs_state_sys = abs_model.get_abs_state(ego_state)
    abs_state_sys_index = abs_model.get_state_index(abs_state_sys)
    # oppo_abs_state = abs_model.get_abs_state(oppo_car_pos)
    # state_sys_env_index = mdp_prod.get_prod_state_index((abs_state_sys_index, abs_state_env))
    # prod_state_index, prod_state  = prod_auto.update_prod_state(state_sys_env_index, prod_state)
    prod_state_index, prod_state  = prod_auto.update_prod_state(abs_state_sys_index, prod_state)
    occ_measure = LP_prob.solve(P_matrix, cost_map, prod_state_index,
                                prod_auto.accepting_states, None)

    # print("occ_measure_invalid", occ_measure[])
    # plot_occ_measure(occ_measure, prod_auto, abs_model)
    optimal_policy, Z = LP_prob.extract(occ_measure)

    
    # -------------- Simulation Loop --------------------
    iter = 1
    first_decision_made = False

    while True:
        iter += 1
        # if iter == 150:
        #     # Modified: Remove traffic light state change
        #     # abs_state_env = 0 # change the traffic light
        #     abs_state_sys = [-1, -1]   # force re-plan

        ax_1.cla()
        ax_1.set_aspect(1)
        
        
        # if ((abs_state_sys != abs_model.init_abs_state) or   # The ego vehicle state changes
        #        # (oppo_abs_state != abs_model.get_abs_state(oppo_car_pos)) or   # The opponent vehicle moves to a state
        #         (occ_measure is None)):   # There's no occupation measure
        
        # Make decision
        if ((not first_decision_made) or                      # The first decision is not made
            (np.array_equal(abs_model.init_abs_state, target_abs_state_sys))):    # The ego vehicle state changes
            
            abs_state_sys = abs_model.get_abs_state(ego_state)
            abs_state_sys_index = abs_model.get_state_index(abs_state_sys)
            prod_state_index, prod_state  = prod_auto.update_prod_state(abs_state_sys_index, prod_state)

            decision_index = optimal_policy[prod_state_index]
            sys_decision = abs_model.action_set[int(decision_index)]
            print("abs_state_sys:", abs_state_sys)
            print("decision:", sys_decision)
            delta_state = action_to_state_transition(sys_decision)
            target_abs_state_sys = abs_state_sys + delta_state
            # print("target_abs_state_sys:", target_abs_state_sys)
            target_ref = cal_target_ref(target_abs_state_sys, region_res, speed_res) 
            first_decision_made = True

        control_input = mpc_con.solve(ego_state, target_ref)
        
        # print("target_point", target_point)
        # print("control_input:", control_input)
        # print("ego_pos:", ego_pos)

        if iter == 120:
            print('d')


        ego_state = sim.car_dyn(ego_state, control_input, params)
        print("ego_state:", ego_state)
        abs_model.update_abs_init_state(ego_state)

        ego_pos = ego_state[:2]
        plt.gca().set_aspect(1)
        # vis.plot_grid(region_size, region_res, label_func, abs_state_env)
        vis.plot_grid(region_size, region_res, label_func)
        vis.plot_car(ego_state, -control_input[1])
        plt.pause(0.001)

        print("------------new iter-------------")



if __name__ == '__main__':
    main()
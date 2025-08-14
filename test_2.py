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
    target_ref_1 = target_abs_state[1] * region_res[1]   # no puls region_res[0] / 2, becuase of the symmetric ey coordinate
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
        


def get_position_movement_transition_matrix(abs_model):
    """
    Extract simplified transition matrix for position states (r, ey) and movement actions only.
    
    Args:
        abs_model: Abstraction model containing the full MDP
        
    Returns:
        tuple: (pos_states, move_actions, trans_matrix)
        - pos_states: list of (r, ey) position states
        - move_actions: list of movement actions ['l', 'f', 'r']  
        - trans_matrix: 3D numpy array [pos_state_idx, move_action_idx, next_pos_state_idx]
    """
    
    # Get state space dimensions
    state_shape = (abs_model.route_size[0] // abs_model.route_res[0], 
                   abs_model.route_size[1] // abs_model.route_res[1], 
                   abs_model.speed_range // abs_model.speed_res)
    
    # Extract unique position states (r, ey) from full state set
    pos_states = []
    pos_state_map = {}  # Map (r, ey) to index in pos_states
    
    for i, full_state in enumerate(abs_model.state_set):
        r, ey, v = full_state
        pos_key = (r, ey)
        
        if pos_key not in pos_state_map:
            pos_state_map[pos_key] = len(pos_states)
            pos_states.append(pos_key)
    
    # Movement actions
    move_actions = ['l', 'f', 'r']
    
    # Create simplified transition matrix
    num_pos_states = len(pos_states)
    num_move_actions = len(move_actions)
    trans_matrix = np.zeros((num_pos_states, num_move_actions, num_pos_states))
    
    # Get the full transition matrix from the MDP
    full_transitions = abs_model.MDP.transitions  # Shape: [num_states, num_actions, num_states]
    
    # Aggregate transitions
    for from_state_idx, from_full_state in enumerate(abs_model.state_set):
        from_r, from_ey, from_v = from_full_state
        from_pos_key = (from_r, from_ey)
        from_pos_idx = pos_state_map[from_pos_key]
        
        for action_idx, action in enumerate(abs_model.action_set):
            move_action, speed_action = action
            
            # Only process if this is one of our movement actions
            if move_action in move_actions:
                move_idx = move_actions.index(move_action)
                
                # Get transition probabilities for this state-action pair
                transition_probs = full_transitions[from_state_idx, action_idx, :]
                
                # Aggregate probabilities by target position
                for to_state_idx, prob in enumerate(transition_probs):
                    if prob > 0:
                        to_full_state = abs_model.state_set[to_state_idx]
                        to_r, to_ey, to_v = to_full_state
                        to_pos_key = (to_r, to_ey)
                        to_pos_idx = pos_state_map[to_pos_key]
                        
                        # Sum the probability for this position transition
                        trans_matrix[from_pos_idx, move_idx, to_pos_idx] += prob
    
    return pos_states, move_actions, trans_matrix


def verify_position_transitions(abs_model):
    """
    Verify that the position-movement transition matrix matches the expected behavior
    from the trans_func in the Abstraction class.
    """
    print("=== Verifying Position Transitions ===")
    
    # Test a specific case: state (0, 0) with action 'f'
    test_state = [0, 0, 0]  # r=0, ey=0, v=0
    test_actions = [['f', 'c'], ['f', 'a'], ['f', 'd']]  # Forward with different speeds
    
    print(f"Testing from state {test_state} with forward actions:")
    
    # Get state space dimensions and bounds
    r_bl = 0
    r_bu = int(abs_model.route_size[0] / abs_model.route_res[0]) - 1
    ey_bl = -(int(abs_model.route_size[1] / abs_model.route_res[1]) // 2)
    ey_bu = int(abs_model.route_size[1] / abs_model.route_res[1]) // 2
    v_bl = 0
    v_bu = int(abs_model.speed_range / abs_model.speed_res) - 1
    
    state_shape = (abs_model.route_size[0] // abs_model.route_res[0], 
                   abs_model.route_size[1] // abs_model.route_res[1], 
                   abs_model.speed_range // abs_model.speed_res)
    
    print(f"State bounds: r=[{r_bl},{r_bu}], ey=[{ey_bl},{ey_bu}], v=[{v_bl},{v_bu}]")
    print(f"State shape: {state_shape}")
    
    # Get expected transitions from trans_func
    for action in test_actions:
        P_full = abs_model.trans_func(test_state, action)
        P_reshaped = P_full.reshape(state_shape, order='F')
        
        print(f"\nAction {action}:")
        print("Expected main transitions from trans_func:")
        
        # According to trans_func, 'f' action should have:
        # prob_spatial[3, 2] = 0.8  # r+1, ey+0 (main transition)
        # prob_spatial[2, 2] = 0.1  # r+0, ey+0 (stay in place)
        # prob_spatial[4, 2] = 0.05 # r+2, ey+0 (jump forward)
        # prob_spatial[3, 1] = 0.025 # r+1, ey-1 (slight right)
        # prob_spatial[3, 3] = 0.025 # r+1, ey+1 (slight left)
        
        expected_transitions = [
            ((1, 0), "r+1, ey+0 (main)"),
            ((0, 0), "r+0, ey+0 (stay)"), 
            ((2, 0), "r+2, ey+0 (jump)"),
            ((1, -1), "r+1, ey-1 (right drift)"),
            ((1, 1), "r+1, ey+1 (left drift)")
        ]
        
        for (target_r, target_ey), desc in expected_transitions:
            # Check if target position is within bounds
            if (r_bl <= target_r <= r_bu and ey_bl <= target_ey <= ey_bu):
                # Sum over all velocity states for this position transition
                total_prob = 0.0
                for v in range(state_shape[2]):
                    r_idx = target_r - r_bl
                    ey_idx = target_ey - ey_bl  # Convert ey to array index
                    v_idx = v - v_bl
                    total_prob += P_reshaped[r_idx, ey_idx, v_idx]
                
                if total_prob > 0.001:
                    print(f"  -> ({target_r}, {target_ey}): {total_prob:.3f} ({desc})")
    
    # Now check if the position-movement matrix matches
    pos_states, move_actions, trans_matrix = get_position_movement_transition_matrix(abs_model)
    
    # Find indices for our test case
    try:
        pos_idx_from = pos_states.index((0, 0))
        move_idx = move_actions.index('f')
        
        print(f"\nFrom position-movement matrix (aggregated over all speed actions):")
        print(f"From position (0, 0) with action 'f':")
        
        significant_transitions = []
        for target_pos_idx, target_pos_state in enumerate(pos_states):
            prob = trans_matrix[pos_idx_from, move_idx, target_pos_idx]
            if prob > 0.001:
                target_r, target_ey = target_pos_state
                significant_transitions.append((target_r, target_ey, prob))
        
        # Sort by probability
        significant_transitions.sort(key=lambda x: x[2], reverse=True)
        
        for target_r, target_ey, prob in significant_transitions:
            print(f"  -> ({target_r}, {target_ey}): {prob:.3f}")
            
    except ValueError as e:
        print(f"Error: {e}")
        print(f"Available position states: {pos_states[:10]}...")  # Show first 10
    
    print("\n=== Analysis ===")
    print("The position-movement matrix should show:")
    print("1. Higher probability for (1, 0) - main forward transition")
    print("2. Some probability for (0, 0) - staying in place")
    print("3. Lower probabilities for other nearby states")
    print("4. Values should be sums across all speed actions and velocity states")


def print_position_transition_matrix(abs_model, verbose=True):
    """
    Print the simplified position-movement transition matrix in a readable format.
    
    Args:
        abs_model: Abstraction model
        verbose: If True, print detailed transition probabilities
    """
    pos_states, move_actions, trans_matrix = get_position_movement_transition_matrix(abs_model)
    
    print("=== Position-Movement Transition Matrix ===")
    print(f"Position states: {len(pos_states)} states")
    print(f"Movement actions: {move_actions}")
    print(f"Transition matrix shape: {trans_matrix.shape}")
    
    if verbose:
        print("\nDetailed Transitions (showing probabilities > 0.01):")
        for pos_idx, pos_state in enumerate(pos_states):
            r, ey = pos_state
            print(f"\nFrom position ({r}, {ey}):")
            
            for move_idx, move_action in enumerate(move_actions):
                print(f"  Action '{move_action}':")
                
                # Find significant transitions
                significant_transitions = []
                for target_pos_idx, target_pos_state in enumerate(pos_states):
                    prob = trans_matrix[pos_idx, move_idx, target_pos_idx]
                    if prob > 0.01:  # Only show significant probabilities
                        target_r, target_ey = target_pos_state
                        significant_transitions.append((target_r, target_ey, prob))
                
                # Sort by probability (highest first)
                significant_transitions.sort(key=lambda x: x[2], reverse=True)
                
                if significant_transitions:
                    for target_r, target_ey, prob in significant_transitions:
                        print(f"    -> ({target_r}, {target_ey}): {prob:.3f}")
                else:
                    print(f"    -> No significant transitions")
    
    return pos_states, move_actions, trans_matrix

from collections import defaultdict     
import matplotlib.pyplot as plt
import numpy as np
def show_occ_measure(occ_measure, abs_model, prod_auto, prod_state_index=None):
    # Get the flattened occ measure (state, action) in the format whose key is 
    # ((r, ey, v, s_cs, s_s, action_index), (move, speed))
    occ_measure_flattened = {}
    for state_action, occ_prob in occ_measure.items():
        state_index = state_action[0]
        action_index = state_action[1]

        # Parse the state index to the exact states
        x, s_cs, s_s = prod_auto.prod_state_set[state_index]
        r, ey, v = abs_model.state_set[x]

        # Parse the action index to the exact action
        move, speed = abs_model.action_set[action_index]
        occ_measure_flattened[((r, ey, v, s_cs, s_s), (move, speed))] = occ_prob

    # print("occ_measure_flattened_length:", len(occ_measure_flattened))
    # print("occ_measure_flattened:", occ_measure_flattened)


    # Compute occ measure (state)
    # Store state occupation measure
    occ_measure_states = defaultdict(float)

    for (state_tuple, action_tuple), occ_prob in occ_measure_flattened.items():
        r, ey, v, s_cs, s_s = state_tuple  # ignore action_index for state measure
        
        # Aggregate over actions
        occ_measure_states[(r, ey, v, s_cs, s_s)] += occ_prob

    # Convert to dict if needed
    occ_measure_states = dict(occ_measure_states)

    # print("occ_measure_states_length:", len(occ_measure_states))
    # print("occ_measure_states:", occ_measure_states)

    # Aggregate over v, s_cs, s_s to get only (r, ey)
    occ_measure_r_ey = defaultdict(float)
    for (r, ey, v, s_cs, s_s), val in occ_measure_states.items():
        occ_measure_r_ey[(r, ey)] += val

    # Normalize occupation measures globally across all states
    total_global_occ = sum(occ_measure_r_ey.values())
    if total_global_occ > 0:
        for pos in occ_measure_r_ey:
            occ_measure_r_ey[pos] /= total_global_occ

    print("occ_measure_r_ey:", occ_measure_r_ey)
    print()

    # Aggregate movement actions by (r, ey) position
    occ_measure_movement = defaultdict(lambda: defaultdict(float))
    for (state_tuple, action_tuple), occ_prob in occ_measure_flattened.items():
        r, ey, v, s_cs, s_s = state_tuple
        move, speed = action_tuple
        
        # Aggregate over v, s_cs, s_s, and speed actions - keep only movement actions
        occ_measure_movement[(r, ey)][move] += occ_prob

    # Normalize movement action occupation measures globally across all state-action pairs
    # modified
    # all_movement_occ_values = []
    # for pos_data in occ_measure_movement.values():
    #     all_movement_occ_values.extend(pos_data.values())
    
    # total_movement_occ = sum(all_movement_occ_values)
    # if total_movement_occ > 0:
    for pos in occ_measure_movement:
        for move in occ_measure_movement[pos]:
            occ_measure_movement[pos][move] /= total_global_occ

    print("occ_measure_movement:", occ_measure_movement)
    print()

    # Test: Check flow balance equation with proper initial state handling
    # Get raw (unnormalized) values for flow balance check
    occ_measure_r_ey_raw = defaultdict(float)
    for (r, ey, v, s_cs, s_s), val in occ_measure_states.items():
        occ_measure_r_ey_raw[(r, ey)] += val
    
    occ_measure_movement_raw = defaultdict(lambda: defaultdict(float))
    for (state_tuple, action_tuple), occ_prob in occ_measure_flattened.items():
        r, ey, v, s_cs, s_s = state_tuple
        move, speed = action_tuple
        occ_measure_movement_raw[(r, ey)][move] += occ_prob
    
    # Get the initial position from the abstraction model
    initial_pos = (abs_model.init_abs_state[0], abs_model.init_abs_state[1])
    
    # Run the flow balance check (expected to fail due to aggregation)
    flow_balance_ok = check_flow_balance(occ_measure_r_ey_raw, occ_measure_movement_raw, abs_model, initial_pos)
    
    # Run the CORRECT flow balance check on product states
    if prod_state_index is not None:
        product_flow_balance_ok = check_product_flow_balance(occ_measure, abs_model, prod_auto, prod_state_index)
    else:
        print("\n⚠️  Skipping product flow balance test - prod_state_index not provided")

    # Save occupation measures to files
    import json
    import numpy as np
    
    # Convert defaultdict and numpy types to regular dict/lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, defaultdict):
            return dict(obj)
        return obj
    
    # Save state occupation measures
    state_occ_data = {}
    for (r, ey), occ_val in occ_measure_r_ey.items():
        state_key = f"r{int(r)}_ey{int(ey)}"
        state_occ_data[state_key] = float(occ_val)
    
    with open('occupation_measure_states.json', 'w') as f:
        json.dump(state_occ_data, f, indent=2)
    
    # Save state-action occupation measures
    state_action_occ_data = {}
    for (r, ey), actions in occ_measure_movement.items():
        state_key = f"r{int(r)}_ey{int(ey)}"
        state_action_occ_data[state_key] = {}
        for move, occ_val in actions.items():
            state_action_occ_data[state_key][move] = float(occ_val)
    
    with open('occupation_measure_state_actions.json', 'w') as f:
        json.dump(state_action_occ_data, f, indent=2)
    
    print("Saved occupation measures to:")
    print("  - occupation_measure_states.json (state occupation measures)")
    print("  - occupation_measure_state_actions.json (state-action occupation measures)")


    # Create grid for plotting
    r_vals = sorted(set(r for r, ey in occ_measure_r_ey.keys()))
    ey_vals = sorted(set(ey for r, ey in occ_measure_r_ey.keys()))
    R, EY = np.meshgrid(r_vals, ey_vals)
    # print("R_shape:", R.shape)
    # print("R:", R)
    # print("EY_shape:", EY.shape)   
    # print("EY:", EY)

    Z = np.zeros_like(R, dtype=float)
    for i, ey in enumerate(ey_vals):
        for j, r in enumerate(r_vals):
            # modified
            Z[i, j] = occ_measure_r_ey[(r, ey)]
            # Z[i, j] = occ_measure_r_ey.get((r, ey), 0.0)

    print("Z_shape :", Z.shape)
    print("Z:", Z)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(
        Z,
        origin='lower',
        cmap='Greys_r',
        aspect='equal'  # each cell square
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Occupation Measure", fontsize=12)

    ax.set_xlabel("r index", fontsize=12)
    ax.set_ylabel("e_y index", fontsize=12)
    ax.set_title("Occupation Measure with Movement Actions", fontsize=14)

    # Optional: set tick labels to show actual indices
    ax.set_xticks(range(Z.shape[1]))
    ax.set_yticks(range(Z.shape[0]))

    # Label each cell with the corresponding value
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            # text_color = "black" if Z[i, j] > Z.max() / 2 else "black"  # You can adjust color for visibility
            text_color = "red"
            ax.text(j, i, f"{Z[i, j]:.2f}", ha='center', va='center', color=text_color, fontsize=8)

    # Add arrows for movement actions
    arrow_scale = 0.25  # Fixed arrow length
    arrow_offset = 0.15  # Offset from center for multiple arrows
    
    for i, ey in enumerate(ey_vals):
        for j, r in enumerate(r_vals):
            if (r, ey) in occ_measure_movement:
                movement_data = occ_measure_movement[(r, ey)]  # Already globally normalized
                
                # Find the action with highest occupation measure at this state
                if movement_data:
                    max_action = max(movement_data.keys(), key=lambda k: movement_data[k])
                    
                    # Define arrow directions and positions
                    arrow_configs = {
                        'l': {'dx': arrow_scale, 'dy': arrow_scale, 'pos_offset': (arrow_offset, arrow_offset)},     # Left: up-right (r+1, ey+1)
                        'f': {'dx': arrow_scale, 'dy': 0, 'pos_offset': (0, 0)},                                   # Forward: right (r+1, ey+0)
                        'r': {'dx': arrow_scale, 'dy': -arrow_scale, 'pos_offset': (arrow_offset, -arrow_offset)}  # Right: down-right (r+1, ey-1)
                    }
                    
                    for move_action, global_occ_value in movement_data.items():
                        if move_action in arrow_configs:
                            config = arrow_configs[move_action]
                            
                            # Determine arrow color: red for highest, gray for others
                            if move_action == max_action:
                                arrow_color = 'red'
                                text_color = 'red'
                                alpha = 1.0
                            else:
                                arrow_color = 'gray'
                                text_color = 'gray'
                                alpha = 0.6
                            
                            # Position for this arrow (offset from cell center)
                            arrow_x = j + config['pos_offset'][0]
                            arrow_y = i + config['pos_offset'][1]
                            
                            # Fixed arrow direction and size
                            dx = config['dx']
                            dy = config['dy']
                            
                            # Draw arrow with fixed length
                            ax.arrow(arrow_x, arrow_y, dx, dy, 
                                    head_width=0.08, head_length=0.06, 
                                    fc=arrow_color, ec=arrow_color, 
                                    alpha=alpha, linewidth=1.5)
                            
                            # Add text label with globally normalized occupation value
                            text_x = arrow_x + dx + 0.1 * np.sign(dx) if dx != 0 else arrow_x + 0.1
                            text_y = arrow_y + dy + 0.1 * np.sign(dy) if dy != 0 else arrow_y + 0.1
                            ax.text(text_x, text_y, f"{global_occ_value:.4f}",  # modified
                                   ha='center', va='center', fontsize=6, 
                                   color=text_color, fontweight='bold',
                                   bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7, edgecolor='none'))

    # Add legend for movement actions
    from matplotlib.patches import FancyArrowPatch
    legend_elements = [
        plt.Line2D([0], [0], marker='>', color='red', label='Highest occupation', markersize=8, linestyle='-'),
        plt.Line2D([0], [0], marker='>', color='gray', label='Lower occupation', markersize=8, linestyle='-'),
        plt.Line2D([0], [0], marker=' ', color='white', label='Actions: L(↗), F(→), R(↘)', markersize=0, linestyle='')
    ]
    ax.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1.0, 0.0))

    plt.tight_layout()
    plt.show()

    return occ_measure_r_ey, occ_measure_movement

def check_flow_balance(occ_measure_r_ey_raw, occ_measure_movement_raw, abs_model, initial_pos):
    """
    Check flow balance equation with proper initial state handling:
    For initial state: occ_measure(s0) = 1 + γ * Σ[incoming flows]
    For other states: occ_measure(s) = γ * Σ[incoming flows]
    
    NOTE: This test is expected to FAIL because:
    1. LP solver enforces flow balance on PRODUCT states (MDP × DFA₁ × DFA₂)
    2. This test checks flow balance on POSITION states (r, ey) 
    3. Aggregating from product space to position space breaks flow balance
    4. Flow balance holds in the expanded product space, not the aggregated space
    """
    print("=== Flow Balance Check with Initial State Handling ===")
    print("⚠️  WARNING: This test is EXPECTED to fail due to state space aggregation")
    print("Flow balance holds in product space (200 states), not position space (25 states)")
    
    # Get the position-movement transition matrix
    pos_states, move_actions, trans_matrix = get_position_movement_transition_matrix(abs_model)
    
    # Create mapping from position to index
    pos_to_idx = {pos: i for i, pos in enumerate(pos_states)}
    move_to_idx = {move: i for i, move in enumerate(move_actions)}
    
    gamma = 0.9  # Discount factor
    flow_balance_errors = []
    
    print(f"Initial position: {initial_pos}")
    print(f"Checking flow balance for {len(pos_states)} positions...")
    print(f"Using transition matrix shape: {trans_matrix.shape}")
    
    for target_pos in pos_states:
        # Left side: occupation measure of target position
        left_side = occ_measure_r_ey_raw.get(target_pos, 0.0)
        
        # Right side: incoming flows + initial injection
        incoming_flow = 0.0
        target_idx = pos_to_idx[target_pos]
        
        # Calculate incoming flows from all source positions and movement actions
        for source_pos in pos_states:
            source_idx = pos_to_idx[source_pos]
            source_movement = occ_measure_movement_raw.get(source_pos, {})
            
            for move_action in move_actions:
                move_idx = move_to_idx[move_action]
                move_occ = source_movement.get(move_action, 0.0)
                
                if move_occ > 0:
                    # Get transition probability from the precomputed matrix
                    transition_prob = trans_matrix[source_idx, move_idx, target_idx]
                    incoming_flow += gamma * move_occ * transition_prob
        
        # Add initial injection term for initial state
        initial_injection = 1.0 if target_pos == initial_pos else 0.0
        right_side = initial_injection + incoming_flow
        
        # Check balance
        error = abs(left_side - right_side)
        relative_error = error / max(left_side, 1e-10)
        
        if error > 1e-6 or relative_error > 0.01:
            flow_balance_errors.append({
                'position': target_pos,
                'left_side': left_side,
                'right_side': right_side,
                'incoming_flow': incoming_flow,
                'initial_injection': initial_injection,
                'error': error,
                'relative_error': relative_error
            })
        
        # Print details for key positions
        if target_pos in [(0, 0), (0, 1), (0, 2), (4, 0), (4, 1)] or target_pos == initial_pos:
            match = error <= 1e-6 and relative_error <= 0.01
            print(f"Position {target_pos}: left={left_side:.6f}, right={right_side:.6f} (init={initial_injection:.1f}+flow={incoming_flow:.6f}), error={error:.2e}, match={match}")
    
    if flow_balance_errors:
        print(f"\n❌ EXPECTED VIOLATIONS! {len(flow_balance_errors)} positions have errors due to state space aggregation:")
        for error_info in flow_balance_errors[:3]:
            pos = error_info['position']
            left = error_info['left_side'] 
            right = error_info['right_side']
            init = error_info['initial_injection']
            flow = error_info['incoming_flow']
            err = error_info['error']
            print(f"  {pos}: left={left:.6f}, right={right:.6f} (init={init:.1f}+flow={flow:.6f}), error={err:.2e}")
        print("\n✅ This is NORMAL: Flow balance holds in product space, not aggregated position space")
    else:
        print(f"🎯 UNEXPECTED! Flow balance satisfied for all {len(pos_states)} positions.")
        print("This would indicate special properties of the aggregation.")
    
    return len(flow_balance_errors) == 0


def check_product_flow_balance(occ_measure, abs_model, prod_auto, initial_prod_state_idx):
    """
    Check flow balance equation on the SAME state space as the LP solver:
    Product automaton states (x, s_cs, s_s) where flow balance should hold exactly.
    """
    print("\n=== Product State Flow Balance Check ===")
    print("Checking flow balance on the same 200-state product space as LP solver")
    
    gamma = 0.9  # Same discount factor as LP solver
    flow_balance_errors = []
    
    # Get product transition matrix and state/action mappings
    P_matrix = prod_auto.prod_transitions
    prod_states = prod_auto.prod_state_set
    prod_actions = prod_auto.prod_action_set
    
    print(f"Product states: {len(prod_states)}")
    print(f"Product actions: {len(prod_actions)}")
    print(f"Initial product state index: {initial_prod_state_idx}")
    print(f"Transition matrix shape: {P_matrix.shape}")
    
    # Check flow balance for each product state
    for target_state_idx, target_state in enumerate(prod_states):
        # Left side: total occupation measure for this product state
        left_side = 0.0
        for action_idx in range(len(prod_actions)):
            state_action_key = (target_state_idx, action_idx)
            # modified
            left_side += occ_measure[state_action_key]
        
        # Right side: incoming flows + initial injection
        incoming_flow = 0.0
        for source_state_idx in range(len(prod_states)):
            for action_idx in range(len(prod_actions)):
                state_action_key = (source_state_idx, action_idx)
                # modified
                source_occupation = occ_measure[state_action_key]
                transition_prob = P_matrix[source_state_idx, action_idx, target_state_idx]
                incoming_flow += gamma * source_occupation * transition_prob
        
        # Add initial injection term for initial state
        initial_injection = 1.0 if target_state_idx == initial_prod_state_idx else 0.0
        right_side = initial_injection + incoming_flow
        
        # Check balance
        error = abs(left_side - right_side)
        relative_error = error / max(left_side, 1e-10)
        
        if error > 1e-6 or relative_error > 0.01:
            flow_balance_errors.append({
                'state_idx': target_state_idx,
                'state': target_state,
                'left_side': left_side,
                'right_side': right_side,
                'incoming_flow': incoming_flow,
                'initial_injection': initial_injection,
                'error': error,
                'relative_error': relative_error
            })
        
        # Print details for first few states and initial state
        if target_state_idx < 5 or target_state_idx == initial_prod_state_idx:
            match = error <= 1e-6 and relative_error <= 0.01
            x, s_cs, s_s = target_state
            r, ey, v = abs_model.state_set[x]
            print(f"State {target_state_idx} ({r},{ey},{v},{s_cs},{s_s}): left={left_side:.6f}, right={right_side:.6f} (init={initial_injection:.1f}+flow={incoming_flow:.6f}), error={error:.2e}, match={match}")
    
    if flow_balance_errors:
        print(f"\n❌ PRODUCT FLOW BALANCE VIOLATIONS! {len(flow_balance_errors)} states have errors:")
        for error_info in flow_balance_errors[:3]:
            state_idx = error_info['state_idx']
            state = error_info['state']
            left = error_info['left_side']
            right = error_info['right_side']
            init = error_info['initial_injection']
            flow = error_info['incoming_flow']
            err = error_info['error']
            x, s_cs, s_s = state
            r, ey, v = abs_model.state_set[x]
            print(f"  State {state_idx} ({r},{ey},{v},{s_cs},{s_s}): left={left:.6f}, right={right:.6f} (init={init:.1f}+flow={flow:.6f}), error={err:.2e}")
        print("\n🚨 This indicates an issue with LP solver or our understanding!")
    else:
        print(f"✅ SUCCESS! Product flow balance satisfied for all {len(prod_states)} states.")
        print("This confirms the LP solver's flow balance constraints are working correctly.")
    
    return len(flow_balance_errors) == 0





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
    max_speed = 4
    speed_res = 2

    # static_label = {(15, 20, 15, 20): "t"}  
    # label_func = dyn_labelling(static_label, init_oppo_car_pos, [-1, 0])
    # speed_limit = 30

    # label_func = {(40, 50, 6, 10, 0, max_speed): "t",
    #                (20, 30, -2, 2, 0, max_speed): "o"}   # "o" for "obstacle"

    #basic
    # label_func = {(40, 50, 6, 10, 0, max_speed): "t",
    #                (20, 30, -2, 2, 0, max_speed): "o"}   # "o" for "obstacle"
    label_func = {(40, 50, 6, 10, 0, max_speed): "t",
                   (20, 30, -2, 2, 0, max_speed): "o"}   # "o" for "obstacle"

    # # overspeed test 1
    # speed_limit = 6
    # label_func = {(30, 40, 6, 10, 0, max_speed): "t",
    #                (20, 30, -2, 2, 0, max_speed): "o",  # "o" for "obstacle"
    #                (10, 20, -2, 10, speed_limit, max_speed): "s"}   # "s" for "overspeed"
    
    ego_state = np.array([0, 0, np.pi / 2, 0])

    abs_model = Abstraction(region_size, region_res, max_speed, speed_res, ego_state, label_func) 

    # ---------- Specification Define --------------------
    safe_frag = LTL_Spec("G(~o)", AP_set=['o'])
    scltl_frag = LTL_Spec("F(t)", AP_set=['t'])

    # basic
    cost_func = {'o': 5}  

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
    

    
    # -------------- Simulation Loop --------------------
    iter = 1

    while True:
        iter += 1
        if iter == 150:
            # Modified: Remove traffic light state change
            # abs_state_env = 0 # change the traffic light
            abs_state_sys = [-1, -1]   # force re-plan

        ax_1.cla()
        ax_1.set_aspect(1)
        
        
        if ((abs_state_sys != abs_model.init_abs_state) or   # The ego vehicle state changes
               # (oppo_abs_state != abs_model.get_abs_state(oppo_car_pos)) or   # The opponent vehicle moves to a state
                (occ_measure is None)):   # There's no occupation measure
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
            # __import__('pickle').dump(occ_measure, open('occ_measure_new.pkl','wb'))
            # # Also save in easy-to-compare CSV and JSON formats (sorted by keys)
            # import csv, json
            # sorted_items = sorted(occ_measure.items(), key=lambda kv: (kv[0][0], kv[0][1]))
            # with open('occ_measure_new.csv', 'w', newline='') as f:
            #     w = csv.writer(f)
            #     w.writerow(['prod_state_idx', 'action_idx', 'value'])
            #     for (s, a), v in sorted_items:
            #         w.writerow([s, a, f"{float(v):.10f}"])
            # with open('occ_measure_new.json', 'w') as f:
            #     json.dump([
            #         {'s': int(s), 'a': int(a), 'v': float(v)} for (s, a), v in sorted_items
            #     ], f, indent=2)
            
            # print_position_transition_matrix(abs_model, True)
            # show_occ_measure(occ_measure, abs_model, prod_auto, prod_state_index)
            # print("occ_measure_length:", len(occ_measure))
            # print("occ_measure:", occ_measure)
            
            optimal_policy, Z = LP_prob.extract(occ_measure)
            decision_index = optimal_policy[prod_state_index]
            sys_decision = abs_model.action_set[int(decision_index)]
            print("abs_state_sys:", abs_state_sys)
            print("decision:", sys_decision)
            delta_state = action_to_state_transition(sys_decision)
            target_abs_state_sys = abs_state_sys + delta_state
            
        target_ref = cal_target_ref(target_abs_state_sys, region_res, speed_res) 
        print("target_ref:", target_ref)
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








        

# def check_P_matrix_bounds(P_matrix, abs_model, prod_auto):
#     """Check if P_matrix allows transitions to invalid states"""
#     print("=== Checking Product Automaton P_matrix for out-of-bounds transitions ===")
    
#     # Get state space bounds for the original MDP part
#     state_bounds = {
#         'x_max': abs_model.map_range[0] // abs_model.map_res[0] - 1,
#         'y_max': abs_model.map_range[1] // abs_model.map_res[1] - 1, 
#         'v_max': abs_model.speed_range // abs_model.speed_res - 1
#     }
#     print(f"MDP State bounds: x=[0,{state_bounds['x_max']}], y=[0,{state_bounds['y_max']}], v=[0,{state_bounds['v_max']}]")
    
#     num_prod_states = len(prod_auto.prod_state_set)
#     num_actions = len(prod_auto.prod_action_set)
    
#     print(f"Product automaton has {num_prod_states} states and {num_actions} actions")
    
#     invalid_transitions = []
    
#     for prod_state_idx in range(num_prod_states):
#         # Product state is (mdp_state_idx, dfa1_state, dfa2_state)
#         prod_state = prod_auto.prod_state_set[prod_state_idx]
#         mdp_state_idx, dfa1_state, dfa2_state = prod_state
        
#         # Get the actual MDP state (x, y, v coordinates)
#         current_mdp_state = abs_model.state_set[mdp_state_idx]
        
#         for action_idx in range(num_actions):
#             action = abs_model.action_set[action_idx]  # This should be the MDP action
            
#             # Get transition probabilities for this product state-action pair
#             trans_probs = P_matrix[prod_state_idx, action_idx, :]
            
#             # Check each possible next product state
#             for next_prod_state_idx in range(num_prod_states):
#                 if trans_probs[next_prod_state_idx] > 0:  # Non-zero transition probability
#                     next_prod_state = prod_auto.prod_state_set[next_prod_state_idx]
#                     next_mdp_state_idx, next_dfa1_state, next_dfa2_state = next_prod_state
                    
#                     # Get the actual next MDP state coordinates
#                     next_mdp_state = abs_model.state_set[next_mdp_state_idx]
                    
#                     # Check if next MDP state is within bounds
#                     if (next_mdp_state[0] < 0 or next_mdp_state[0] > state_bounds['x_max'] or
#                         next_mdp_state[1] < 0 or next_mdp_state[1] > state_bounds['y_max'] or
#                         next_mdp_state[2] < 0 or next_mdp_state[2] > state_bounds['v_max']):
                        
#                         invalid_transitions.append({
#                             'from_prod_state': prod_state,
#                             'from_mdp_state': current_mdp_state,
#                             'action': action,
#                             'to_prod_state': next_prod_state,
#                             'to_mdp_state': next_mdp_state,
#                             'probability': trans_probs[next_prod_state_idx],
#                             'prod_state_idx': prod_state_idx,
#                             'action_idx': action_idx,
#                             'next_prod_state_idx': next_prod_state_idx
#                         })
    
#     if invalid_transitions:
#         print(f"FOUND {len(invalid_transitions)} INVALID TRANSITIONS IN PRODUCT AUTOMATON!")
#         for i, trans in enumerate(invalid_transitions[:5]):  # Show first 5
#             print(f"  {i+1}: MDP State {trans['from_mdp_state']} --{trans['action']}--> MDP State {trans['to_mdp_state']} (prob={trans['probability']:.4f})")
#             print(f"      Product: {trans['from_prod_state']} --> {trans['to_prod_state']}")
#         if len(invalid_transitions) > 5:
#             print(f"  ... and {len(invalid_transitions) - 5} more")
#     else:
#         print("No invalid MDP transitions found in Product Automaton P_matrix")
    
#     # Also check if there are MDP states that can lead to out-of-bounds via simple addition
#     print("\n=== Checking MDP state+action combinations that would be out-of-bounds ===")
#     problematic_combinations = []
    
#     for mdp_state_idx in range(len(abs_model.state_set)):
#         current_mdp_state = abs_model.state_set[mdp_state_idx]
        
#         for action_idx in range(len(abs_model.action_set)):
#             action = abs_model.action_set[action_idx]
#             target_state = current_mdp_state + action
            
#             if (target_state[0] < 0 or target_state[0] > state_bounds['x_max'] or
#                 target_state[1] < 0 or target_state[1] > state_bounds['y_max'] or
#                 target_state[2] < 0 or target_state[2] > state_bounds['v_max']):
                
#                 problematic_combinations.append({
#                     'mdp_state': current_mdp_state,
#                     'action': action, 
#                     'target': target_state,
#                     'mdp_state_idx': mdp_state_idx,
#                     'action_idx': action_idx
#                 })
    
#     if problematic_combinations:
#         print(f"FOUND {len(problematic_combinations)} MDP STATE+ACTION COMBINATIONS THAT LEAD OUT-OF-BOUNDS!")
#         for i, combo in enumerate(problematic_combinations[:10]):  # Show first 10
#             print(f"  {i+1}: MDP State {combo['mdp_state']} + Action {combo['action']} = {combo['target']} (OUT OF BOUNDS)")
#         if len(problematic_combinations) > 10:
#             print(f"  ... and {len(problematic_combinations) - 10} more")
#     else:
#         print("No problematic MDP state+action combinations found")
    
#     print("=== End Product Automaton P_matrix bounds check ===\n")
#     return invalid_transitions, problematic_combinations













if __name__ == '__main__':
    main()
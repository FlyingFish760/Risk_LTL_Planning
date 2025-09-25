import matplotlib.pyplot as plt
import numpy as np
import sys
import os

show_grid = True


def plot_occ_measure(occ_measure, prod_auto, abs_model):
    '''
    Plot occmeasure of states and the occ measure of actions in each state
    '''
    
    # Set white background for plots
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'

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
    
    # Calculate adaptive font sizes based on grid size
    total_cells = r_max * ey_max
    if total_cells <= 20:  # Small grids
        occupation_font_size = 14  # Bigger for occupation measure
        action_font_size = 11
    elif total_cells <= 50:  # Medium grids
        occupation_font_size = 12  # Bigger for occupation measure
        action_font_size = 9
    elif total_cells <= 100:  # Large grids
        occupation_font_size = 10  # Bigger for occupation measure
        action_font_size = 7
    else:  # Very large grids
        occupation_font_size = 8  # Bigger for occupation measure
        action_font_size = 5
    
    # Create a grid for the heat map
    heat_map = np.zeros((ey_max, r_max))
    
    # Fill the heat map with occupation values
    for (r, ey), occ_value in state_occ.items():
        heat_map[ey, r] = occ_value
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    im = plt.imshow(heat_map, cmap='YlOrRd', interpolation='nearest', origin='lower')
    plt.colorbar(im, label='Occupation Measure')
    
    # Add grid lines at cell boundaries and axis labels at cell centers
    if show_grid:
        # Grid lines at cell boundaries (between cells)
        plt.gca().set_xticks(np.arange(-0.5, r_max, 1), minor=True)
        plt.gca().set_yticks(np.arange(-0.5, ey_max, 1), minor=True)
        plt.gca().grid(which='minor', alpha=0.5, color='black', linewidth=0.5)
        
    # Axis labels at cell centers (0, 1, 2, ...)
    plt.gca().set_xticks(np.arange(0, r_max, 1), minor=False)
    plt.gca().set_yticks(np.arange(0, ey_max, 1), minor=False)
    
    # Add value labels at each state with better visibility
    max_value = np.max(list(state_occ.values()))
    for (r, ey), occ_value in state_occ.items():
        plt.text(r, ey-0.3, f'{occ_value:.3f}', 
                ha='center', va='center', color='black',
                fontsize=occupation_font_size, weight='bold')
    
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
            
            # Determine color: green for max action, lighter blue for others
            arrow_color = 'green' if action_type == max_action_type else 'lightblue'
            text_color = 'green' if action_type == max_action_type else 'lightblue'
            
            # Draw arrow if there's a clear direction (positioned to avoid text overlap)
            if dx != 0 or dy != 0:
                # Position arrows further to the right, away from text
                arrow_start_x = start_x - 0.2
                arrow_start_y = start_y
                # Make arrows shorter to fit better
                arrow_dx = dx * 1
                arrow_dy = dy * 1
                plt.arrow(arrow_start_x, arrow_start_y, arrow_dx, arrow_dy, head_width=0.04, head_length=0.02, 
                         fc=arrow_color, ec=arrow_color, alpha=0.8, linewidth=1.0)
                
                # Add action label and occupation value to the left of the arrow
                plt.text(start_x + dx - 0.15, start_y + dy, f'{action_label}: {action_value:.2f}', 
                        ha='left', va='center', color=text_color, fontsize=action_font_size, weight='bold')
    
    plt.xlabel('r (longitudinal position)')
    plt.ylabel('ey (lateral position)')
    plt.title('Positional Occupation Measure Heat Map with Actions')
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
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
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
            
            # Calculate adaptive font sizes for velocity plots (reverse to smaller action font)
            total_cells_velocity = r_max * ey_max
            if total_cells_velocity <= 20:  # Small grids
                velocity_occupation_font_size = 10
                velocity_action_font_size = 7  # Back to smaller size
            elif total_cells_velocity <= 50:  # Medium grids
                velocity_occupation_font_size = 10
                velocity_action_font_size = 6  # Back to smaller size
            elif total_cells_velocity <= 100:  # Large grids
                velocity_occupation_font_size = 6
                velocity_action_font_size = 5  # Back to smaller size
            else:  # Very large grids
                velocity_occupation_font_size = 4
                velocity_action_font_size = 4  # Back to smaller size
            
            # Create a grid for the heat map
            heat_map_velocity = np.zeros((ey_max, r_max))
            
            # Fill the heat map with occupation values
            for (r, ey), occ_value in state_occ_velocity.items():
                heat_map_velocity[ey, r] = occ_value
            
            # Create the heat map on the subplot
            im = ax.imshow(heat_map_velocity, cmap='YlOrRd', interpolation='nearest', origin='lower')
            
            # Add grid lines at cell boundaries and axis labels at cell centers
            if show_grid:
                # Grid lines at cell boundaries (between cells)
                ax.set_xticks(np.arange(-0.5, r_max, 1), minor=True)
                ax.set_yticks(np.arange(-0.5, ey_max, 1), minor=True)
                ax.grid(which='minor', alpha=0.5, color='black', linewidth=0.5)
                
            # Axis labels at cell centers (0, 1, 2, ...)
            ax.set_xticks(np.arange(0, r_max, 1), minor=False)
            ax.set_yticks(np.arange(0, ey_max, 1), minor=False)
            
            # Add value labels at each state with better visibility
            for (r, ey), occ_value in state_occ_velocity.items():
                ax.text(r, ey-0.3, f'{occ_value:.3f}', 
                        ha='center', va='center', color='black',
                        fontsize=velocity_occupation_font_size, weight='bold')
            
            # Add speed action labels for all actions in each state
            for (r, ey), action_occ in state_action_occ_velocity.items():
                # Find the action with highest occupation measure for coloring
                if action_occ and any(v > 0 for v in action_occ.values()):
                    max_action_type = max(action_occ.items(), key=lambda x: x[1])[0]
                else:
                    max_action_type = None
                
                # Position actions in a vertical column at the center of the cell
                action_order = ['a', 'c', 'd']  # Top to bottom order: accelerate, cruise, decelerate
                
                # Label all speed actions in a column
                action_index = 0
                for action_type in action_order:
                    if action_type in action_occ:
                        action_value = action_occ[action_type]
                        action_label = ""
                        if action_type == 'a':  # accelerate
                            action_label = "A"
                        elif action_type == 'd':  # decelerate
                            action_label = "D"
                        elif action_type == 'c':  # cruise
                            action_label = "C"
                        
                        # Determine color: green for max action, lighter blue for others
                        text_color = 'green' if action_type == max_action_type else 'lightblue'
                        
                        # Position in column: start from top and go down
                        column_y_offset = 0.25 - (action_index * 0.15)
                        
                        # Add action label and occupation value
                        ax.text(r, ey + column_y_offset, f'{action_label}: {action_value:.2f}', 
                                ha='center', va='center', color=text_color, fontsize=velocity_action_font_size, weight='bold')
                        
                        action_index += 1
            
            # ax.set_xlabel('r (longitudinal position)')
            # ax.set_ylabel('ey (lateral position)')
            ax.set_title(f'Velocity Occupation measure at Velocity {v_val}', fontsize=10)
        
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


def load_and_plot(filename="latest_occupation_data.pkl"):
    """Load saved occupation measure data and plot it"""
    # Add parent directory to path to import from test_2
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(parent_dir)
    
    try:
        from test_2 import load_occupation_measure
        
        print(f"Loading occupation measure data from: {filename}")
        occ_measure, prod_auto, abs_model = load_occupation_measure(filename)
        
        print(f"Loaded {len(occ_measure)} occupation measure entries")
        print("Generating plots...")
        
        plot_occ_measure(occ_measure, prod_auto, abs_model)
        
        print("Plotting complete!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you have run the main simulation first to generate the data file.")
    except Exception as e:
        print(f"Error loading or plotting data: {e}")


if __name__ == "__main__":
    # Allow filename to be passed as command line argument
    filename = "latest_occupation_data.pkl"
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    
    load_and_plot(filename)
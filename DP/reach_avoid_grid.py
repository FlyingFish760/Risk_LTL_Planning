import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from value_iteration import *
from policy_iteration import *

# Define GridWorld environment
class GridWorld:
    def __init__(self, rows, cols, target_set, avoid_set):
        self.rows = rows
        self.cols = cols
        self.states = [(r, c) for r in range(rows) for c in range(cols)]
        self.actions = ['up', 'down', 'left', 'right']
        self.target_set = target_set
        self.avoid_set = avoid_set

    def transition_prob(self, state, action):
        """Define transition probabilities for each action"""
        r, c = state
        if state in self.target_set or state in self.avoid_set:
            return {state: 1.0}  # Target or avoid states are terminal

        if action == 'up':
            intended_next_state = (max(r - 1, 0), c)
        elif action == 'down':
            intended_next_state = (min(r + 1, self.rows - 1), c)
        elif action == 'left':
            intended_next_state = (r, max(c - 1, 0))
        elif action == 'right':
            intended_next_state = (r, min(c + 1, self.cols - 1))
        else:
            intended_next_state = state

        # Define possible unintended next states
        unintended_next_states = {
            'up': [(r, max(c - 1, 0)), (r, min(c + 1, self.cols - 1))],
            'down': [(r, max(c - 1, 0)), (r, min(c + 1, self.cols - 1))],
            'left': [(max(r - 1, 0), c), (min(r + 1, self.rows - 1), c)],
            'right': [(max(r - 1, 0), c), (min(r + 1, self.rows - 1), c)]
        }

        # Set transition probabilities
        transition_probs = {}
        transition_probs[intended_next_state] = 0.8  # 80% chance to move as intended

        # 20% chance to move to unintended states (10% each)
        for unintended_state in unintended_next_states[action]:
            transition_probs[unintended_state] = 0.1

        # Make sure the sum of probabilities is 1
        return transition_probs

    def reward(self, state, action, next_state):
        """Define the reward structure"""
        if next_state in self.target_set:
            return 100  # Positive reward for reaching the target
        elif next_state in self.avoid_set:
            return -100  # High penalty for entering the avoid state
        else:
            return -5  # Step cost to encourage reaching the target quickly


# Visualization function
def visualize_grid(rows, cols, value_function, policy, target_set, avoid_set):
    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw grid lines
    for x in range(cols + 1):
        ax.plot([x, x], [0, rows], 'k')
    for y in range(rows + 1):
        ax.plot([0, cols], [y, y], 'k')

    # Draw value function and policy arrows
    for r in range(rows):
        for c in range(cols):
            state = (r, c)
            value = value_function.get(state, 0)
            action_dist = policy.get(state, None)
            opt_action = max(action_dist, key=action_dist.get) if action_dist else None

            # Draw avoid and target states with specific colors
            if state in target_set:
                ax.add_patch(patches.Rectangle((c, rows - r - 1), 1, 1, color='green', alpha=0.3))
                ax.text(c + 0.5, rows - r - 0.5, 'G', fontsize=15, ha='center', va='center', color='black')
            elif state in avoid_set:
                ax.add_patch(patches.Rectangle((c, rows - r - 1), 1, 1, color='red', alpha=0.3))
                ax.text(c + 0.5, rows - r - 0.5, 'X', fontsize=15, ha='center', va='center', color='black')
            else:
                ax.text(c + 0.5, rows - r - 0.5, f'{value:.1f}', fontsize=10, ha='center', va='center', color='blue')

            # Draw policy arrow
            if opt_action:
                if opt_action == 'up':
                    dx, dy = 0, 0.3
                elif opt_action == 'down':
                    dx, dy = 0, -0.3
                elif opt_action == 'left':
                    dx, dy = -0.3, 0
                elif opt_action == 'right':
                    dx, dy = 0.3, 0
                else:
                    dx, dy = 0, 0

                ax.arrow(c + 0.5, rows - r - 0.5, dx, dy,
                         head_width=0.1, head_length=0.1, fc='black', ec='black')

    # Set axis labels and limits
    ax.set_xticks(np.arange(0.5, cols, 1))
    ax.set_yticks(np.arange(0.5, rows, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')

    plt.title("GridWorld Value Function and Policy")
    plt.show()

# GridWorld example setup
if __name__ == "__main__":
    # Define the grid dimensions, target, and avoid sets
    rows, cols = 10, 10
    target_set = {(9, 9)}  # Bottom-right corner is the target
    avoid_set = {(5, 5), (5, 6)}  # (1, 1) is an avoid state

    # Create GridWorld environment
    grid_world = GridWorld(rows, cols, target_set, avoid_set)

    # Run value iteration
    # value_function, policy = value_iteration(
    #     grid_world.states,
    #     grid_world.actions,
    #     grid_world.transition_prob,
    #     grid_world.reward,
    #     target_set,
    #     avoid_set
    # )

    # Run policy iteration
    value_function, policy = policy_iteration(
        grid_world.states,
        grid_world.actions,
        grid_world.transition_prob,
        grid_world.reward,
        target_set,
        avoid_set
    )


    # Visualize the results
    visualize_grid(rows, cols, value_function, policy, target_set, avoid_set)

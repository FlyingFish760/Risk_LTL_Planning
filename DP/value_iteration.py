import numpy as np

def value_iteration(states, actions, transition_prob, reward, target_set, avoid_set, discount_factor=0.99, theta=1e-6):
    # Initialize the value function to zero for all states
    value_function = {state: 0 for state in states}

    # Initialize a stochastic policy with equal probabilities for all actions
    policy = {state: {action: 1 / len(actions) for action in actions} for state in states if state not in target_set and state not in avoid_set}

    # Value iteration loop
    while True:
        delta = 0
        # Update each state's value based on the current stochastic policy
        for state in states:
            if state in target_set or state in avoid_set:
                continue

            # Calculate the expected value of the state under the current stochastic policy
            expected_value = 0
            for action, action_prob in policy[state].items():
                action_value = 0
                for next_state, prob in transition_prob(state, action).items():
                    action_value += prob * (reward(state, action, next_state) + discount_factor * value_function[next_state])
                expected_value += action_prob * action_value

            # Update value function
            delta = max(delta, abs(value_function[state] - expected_value))
            value_function[state] = expected_value

        # Check for convergence
        if delta < theta:
            break

    # Update the policy to be greedy with respect to the value function
    for state in states:
        if state in target_set or state in avoid_set:
            continue

        action_values = {}
        # Calculate the value of each action for the current state
        for action in actions:
            action_value = 0
            for next_state, prob in transition_prob(state, action).items():
                action_value += prob * (reward(state, action, next_state) + discount_factor * value_function[next_state])
            action_values[action] = action_value

        # Update the policy to be ε-greedy with respect to the action values
        epsilon = 0.1  # Small probability of exploring other actions
        num_actions = len(actions)
        best_action = max(action_values, key=action_values.get)
        for action in actions:
            if action == best_action:
                policy[state][action] = 1 - epsilon + epsilon / num_actions
            else:
                policy[state][action] = epsilon / num_actions

    return value_function, policy

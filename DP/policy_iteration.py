import numpy as np
# Policy Iteration for Reach-Avoid Problem
def policy_iteration(states, actions, transition_prob, reward, discount_factor=0.9, theta=1e-6):
    # Initialize a stochastic policy with equal probabilities for all actions
    policy = {state: {action: 1 / len(actions) for action in actions} for state in states}
    value_function = {state: 0 for state in states}
    Q_function = {state: {action: 0 for action in actions} for state in states}


    while True:
        # Policy Evaluation
        while True:
            delta = 0
            for state in states:

                # Calculate the expected value of the state under the current stochastic policy
                expected_value = 0
                for action, action_prob in policy[state].items():
                    action_value = 0
                    for next_state, prob in transition_prob(state, action).items():
                        action_value += prob * (reward(state, action, next_state) + discount_factor * value_function[next_state])
                    Q_function[state][action] = action_value
                    expected_value += action_prob * action_value

                delta = max(delta, abs(value_function[state] - expected_value))
                value_function[state] = expected_value

            if delta < theta:
                break

        # Policy Improvement
        policy_stable = True
        for state in states:

            # Find the best action value
            action_values = {}
            for action in actions:
                action_value = 0
                for next_state, prob in transition_prob(state, action).items():
                    action_value += prob * (reward(state, action, next_state) + discount_factor * value_function[next_state])
                action_values[action] = action_value

            # Update the policy to be ε-greedy with respect to the action values
            old_action_probs = policy[state].copy()
            action_value_sum = sum(action_values.values())
            for action in actions:
                policy[state][action] = action_values[action] / action_value_sum

            if old_action_probs != policy[state]:
                policy_stable = False

        if policy_stable:
            break

    return value_function, policy

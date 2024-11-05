# Policy Iteration for Reach-Avoid Problem
def policy_iteration(states, actions, transition_prob, reward, target_set, avoid_set, discount_factor=0.8, theta=1e-10):
    # Initialize a stochastic policy with equal probabilities for all actions
    policy = {state: {action: 1 / len(actions) for action in actions} for state in states if state not in target_set and state not in avoid_set}
    value_function = {state: 0 for state in states}

    while True:
        # Policy Evaluation
        while True:
            delta = 0
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

                delta = max(delta, abs(value_function[state] - expected_value))
                value_function[state] = expected_value

            if delta < theta:
                break

        # Policy Improvement
        policy_stable = True
        for state in states:
            if state in target_set or state in avoid_set:
                continue

            # Find the best action value
            action_values = {}
            for action in actions:
                action_value = 0
                for next_state, prob in transition_prob(state, action).items():
                    action_value += prob * (reward(state, action, next_state) + discount_factor * value_function[next_state])
                action_values[action] = action_value

            # Update the policy to be ε-greedy with respect to the action values
            best_action = max(action_values, key=action_values.get)
            old_action_probs = policy[state].copy()
            epsilon = 0.1
            num_actions = len(actions)
            for action in actions:
                if action == best_action:
                    policy[state][action] = 1 - epsilon + epsilon / num_actions
                else:
                    policy[state][action] = epsilon / num_actions

            if old_action_probs != policy[state]:
                policy_stable = False

        if policy_stable:
            break

    return value_function, policy

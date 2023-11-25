import numpy as np


class PolicyIteration:
    def __init__(self, shape, weights=None, discount_factor=0.9):
        self.shape = shape
        self.weights = weights if weights is not None else np.zeros(shape)
        self.discount_factor = discount_factor
        self.actions = ['-2', '-1', '0', '1', '2']
        self.value_function = np.zeros(shape)
        self.policy = np.random.choice(self.actions, shape)

    def in_local_environment(self, state):
        return ((state[0] in range(self.shape[0]-1)) and (state[1] in range(self.shape[1]-1)))

    def get_next_state(self, state, action):
        i, j = state
        if action == '0':
            state = (i, j + 1)
        elif action == '1':
            state = (i + 1, j + 1)
        elif action == '2':
            state = (i + 2, j + 1)
        elif action == '-1':
            state = (i - 1, j + 1)
        elif action == '-2':
            state = (i - 2, j + 1)
        if not self.in_local_environment(state):
            return (max(0, min(self.shape[0]-1, state[0])), max(0, min(self.shape[1]-1, state[1])))
        return state

    def policy_evaluation(self):
        while True:
            new_value_function = np.copy(self.value_function)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    action = self.policy[i, j]
                    next_state = self.get_next_state((i, j), action)
                    reward = self.weights[next_state]
                    new_value_function[i, j] = reward + self.discount_factor * self.value_function[next_state]
            if np.max(np.abs(new_value_function - self.value_function)) < 1e-3:
                break
            self.value_function = new_value_function

    def policy_improvement(self):
        policy_stable = True
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                old_action = self.policy[i, j]
                best_action = None
                best_value = float('-inf')

                for action in self.actions:
                    next_state = self.get_next_state((i, j), action)
                    reward = self.weights[next_state]
                    value = reward + self.discount_factor * self.value_function[next_state]
                    if value > best_value:
                        best_value = value
                        best_action = action

                self.policy[i, j] = best_action
                if old_action != best_action:
                    policy_stable = False

        return policy_stable

    def run(self):
        while True:
            self.policy_evaluation()
            if self.policy_improvement():
                break
        return self.policy


if __name__ == "__main__":
    shape = (4, 4)
    weights = np.random.rand(*shape)  # Random weights for each cell in the grid
    policy_iter = PolicyIteration(shape, weights)
    optimal_policy = policy_iter.run()

    print("Optimal Policy:")
    print(optimal_policy)
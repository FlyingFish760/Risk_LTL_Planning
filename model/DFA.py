class DFA:
    def __init__(self, states, alphabet, transitions, initial_state, sink_states):
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions
        self.initial_state = initial_state
        self.sink_states = sink_states

    def transition(self, state, symbol):
        return self.transitions.get((state, symbol), None)

    def is_sink_state(self, state):
        return state in self.sink_states

    def process_input(self, input_string):
        current_state = self.initial_state
        for symbol in input_string:
            current_state = self.transition(current_state, symbol)
            if current_state is None:
                return False
        return current_state in self.accept_states
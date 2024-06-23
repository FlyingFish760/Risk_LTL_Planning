import numpy as np

from abstraction import Abstraction
from specification.specification import LTL_Spec

class Product:
    def __init__(self, MDP, DFA_cs, DFA_safe):
        self.MDP = MDP
        self.DFA_cs = DFA_cs
        self.DFA_safe = DFA_safe
        self.prod_state_set = [(x, s_cs, s_s)
                              for x in self.MDP.states
                              for s_cs in self.DFA_cs.states
                              for s_s in self.DFA_safe.states]
        self.prod_action_set = self.MDP.actions
        self.prod_transitions = self.gen_product_transition()
        # self.accepting_states =
        # self.trap_states =

    def gen_product_transition(self):
        P_matrix = np.zeros((len(self.prod_state_set), len(self.prod_action_set), len(self.prod_state_set)))
        for n in range(len(self.prod_state_set)):
            x, s_cs, s_s = self.prod_state_set[n]
            for a in self.MDP.actions:
                next_x_dist = self.MDP.transitions.get((x, a))
                for next_x, prob in next_x_dist.items():
                    for next_s_cs in self.DFA_cs.transitions.get((s_cs, self.MDP.labelling.get(x))):
                        for next_s_s in self.DFA_safe.transitions.get((s_s, self.MDP.labelling.get(x))):
                            next_state_index = self.prod_state_set.index((next_x, next_s_cs, next_s_s))
                            P_matrix[n, a, next_state_index] = prob

        return P_matrix


if __name__ == '__main__':
    pcpt_range = (20, 20)
    pcpt_res = (5, 5)
    dt = 1
    initial_position = (2, 2)
    label_func = {(15, 20, 15, 20): "t",
                  (5, 15, 5, 10): "o",
                  (10, 20, 0, 20): "c"}

    abs_model = Abstraction(pcpt_range, pcpt_res, initial_position, label_func)
    MDP = abs_model.MDP

    # safe_frag = LTL_Spec("G(r->!c)")
    safe_frag = LTL_Spec("G(!o)")
    scltl_frag = LTL_Spec("F(t)")
    Product(MDP, safe_frag.dfa, scltl_frag.dfa)

    # print(abs_model.MDP.transitions)
    # print(abs_model.MDP.labelling)
    # print(abs_model.MDP.initial_state)
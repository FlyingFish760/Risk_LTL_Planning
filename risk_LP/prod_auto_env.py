import numpy as np
from risk_LP.abstraction import Abstraction
from specification.specification import LTL_Spec

class Product:
    def __init__(self, MDP_sys, MDP_env, DFA_cs, DFA_safe):
        self.MDP_sys = MDP_sys
        self.MDP_env = MDP_env
        self.DFA_cs = DFA_cs
        self.DFA_safe = DFA_safe
        self.prod_state_set = [(x_sys, x_env, s_cs, s_s)
                              for x_sys in self.MDP_sys.states
                              for x_env in self.MDP_env.states
                              for s_cs in self.DFA_cs.states
                              for s_s in self.DFA_safe.states]
        self.prod_action_set = self.MDP_sys.actions
        self.prod_transitions = self.gen_product_transition()
        self.accepting_states, self.trap_states = self.gen_final_states()

    def gen_product_transition(self):
        P_matrix = np.zeros((len(self.prod_state_set), len(self.prod_action_set), len(self.prod_state_set)))
        for n in range(len(self.prod_state_set)):
            x_sys, x_env, s_cs, s_s = self.prod_state_set[n]
            for a in self.MDP_sys.actions:
                next_x_sys_prob = self.MDP_sys.transitions[x_sys, a, :]
                for next_x_sys in range(len(next_x_sys_prob)):
                    next_x_sys_label = self.MDP_sys.labelling[next_x_sys]

                    for next_x_env in self.MDP_env.states:

                        if self.DFA_cs.transitions.get((str(s_cs), next_x_sys_label)) is not None:
                            next_s_cs = self.DFA_cs.transitions.get((str(s_cs), next_x_sys_label))
                        else:
                            next_s_cs = s_cs

                        if self.DFA_safe.transitions.get((str(s_s), next_x_sys_label)) is not None:
                            next_s_s = self.DFA_safe.transitions.get((str(s_s), next_x_sys_label))
                        else:
                            next_s_s = s_s

                    next_state_index = self.prod_state_set.index((next_x_sys, next_x_env, int(next_s_cs), int(next_s_s)))
                    P_matrix[n, a, next_state_index] = next_x_prod[next_x]
        return P_matrix


    def gen_final_states(self):
        accepting_states = []
        trap_states = []
        for n in range(len(self.prod_state_set)):
            x, s_cs, s_s = self.prod_state_set[n]
            if self.DFA_safe.is_sink_state(str(s_s)):
                trap_states.append(n)
            if self.DFA_cs.is_sink_state(str(s_cs)):
                accepting_states.append(n)
        return accepting_states, trap_states

    def gen_cost_map(self, cost_func):
        cost_map = np.zeros(len(self.prod_state_set))
        for n in range(len(self.prod_state_set)):
            if n in self.accepting_states:
                cost_map[n] += cost_func["co_safe"]
            if n in self.trap_states:
                cost_map[n] += cost_func["safe"]
        return cost_map
    
    def update_prod_state(self, cur_abs_index, last_product_state):
        _, last_cs_state, last_safe_state  = last_product_state
        label = self.MDP_sys.labelling[cur_abs_index]
        if self.DFA_cs.transitions.get((str(last_cs_state), label)) is not None:
            next_cs_state = self.DFA_cs.transitions.get((str(last_cs_state), label))
        else:
            next_cs_state = last_cs_state
        if self.DFA_safe.transitions.get((str(last_safe_state), label)) is not None:
            next_safe_state = self.DFA_safe.transitions.get((str(last_safe_state), label))
        else:
            next_safe_state = last_safe_state
        cur_product_state = (cur_abs_index, int(next_cs_state), int(next_safe_state))
        product_state_index = self.prod_state_set.index(cur_product_state)
        return int(product_state_index), cur_product_state


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
    prod_auto = Product(MDP, safe_frag.dfa, scltl_frag.dfa)

    for n in prod_auto.accepting_states:
        print(prod_auto.prod_state_set[n])
    for n in prod_auto.trap_states:
        print(prod_auto.prod_state_set[n])

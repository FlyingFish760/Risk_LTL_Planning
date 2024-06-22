
def product_mdp_dfa(mdp, dfa):
    product_states = {(s, d) for s in mdp.states for d in dfa.states}
    product_actions = {s: mdp.get_actions(s[0]) for s in product_states}
    product_transitions = {}
    product_rewards = {}
    product_initial_state = (mdp.initial_state, dfa.initial_state)

    for (s, d) in product_states:
        actions = mdp.get_actions(s)
        for a in actions:
            transitions = mdp.get_transitions(s, a)
            for (prob, next_state) in transitions:
                next_dfa_state = dfa.transition(d, a)  # Assuming action labels are DFA alphabet
                if next_dfa_state:
                    product_transitions[((s, d), a)] = product_transitions.get(((s, d), a), []) + [(prob, (next_state, next_dfa_state))]
                product_rewards[((s, d), a)] = mdp.get_reward(s, a)

    return MDP(product_states, product_actions, product_transitions, product_rewards, product_initial_state)
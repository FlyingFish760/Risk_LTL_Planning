from abstraction.abstraction import Abstraction
from abstraction.prod_MDP import Prod_MDP
from specification.prod_auto import Product
import numpy as np
from abstraction.MDP import MDP
from specification.specification import LTL_Spec

SPEED_LOW_BOUND = 10

# ---------- MDP Environment  ---------------------
# Create a trivial MDP with one state and no actions
state_set = [0]
action_set = [0]
transitions = np.array([[[1.0]]])  # Stay in state 0
initial_state = 0
mdp_env = MDP(state_set, action_set, transitions, ["none"], initial_state)

# ---------- MPD System (Abstraction) --------------------
region_size = (20, 20)
region_res = (5, 5)
max_speed = 80
speed_res = 10

# static_label = {(15, 20, 15, 20): "t"}  
# label_func = dyn_labelling(static_label, init_oppo_car_pos, [-1, 0])
speed_limit = 30
label_func = {(15, 20, 5, 10, 0, max_speed): "t",
                (5, 20, 5, 10, speed_limit, max_speed): "o",   # "o" for "overspeed"
                (0, 20, 0, 20, 0, SPEED_LOW_BOUND): "s"}    # "s" for "slow"

ego_state = np.array([2.5, 12.5, np.pi / 2, 0])

abs_model = Abstraction(region_size, region_res, max_speed, speed_res, ego_state, label_func) 

# ---------- Specification Define --------------------
safe_frag = LTL_Spec("G(~o)", AP_set=['o'])
scltl_frag = LTL_Spec("F(t)", AP_set=['t'])

cost_func = {'o': 1, 's': 0.5}  




abs_state_env = 0
abs_state_sys = abs_model.get_abs_state(ego_state) 

mdp_sys = abs_model.MDP
mdp_prod = Prod_MDP(mdp_sys, mdp_env)
prod_auto = Product(mdp_prod.MDP, scltl_frag.dfa, safe_frag.dfa)

abs_state_sys_index = abs_model.get_state_index(abs_state_sys)
state_sys_env_index = mdp_prod.get_prod_state_index((abs_state_sys_index, abs_state_env))
prod_state = (state_sys_env_index, int(scltl_frag.dfa.initial_state), int(safe_frag.dfa.initial_state))
# prod_state = (0, 1, 1)
prod_state_index, prod_state  = prod_auto.update_prod_state(state_sys_env_index, prod_state)
# print(f"\n Debug: second prod_state={prod_state}")
# print(f"\n Debug: prod_state_index={prod_state_index}")
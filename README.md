# Graduation project code
This code is for testing the risk-aware autonomous driving method of Jingyuan Han's graduation project. A speed limit experiment is implemented in the code.


## Code Structure
Please refer to the .readme file in the main branch for an overview of the code structure.


## Prerequisites

### Dependencies
Install the required Python packages:

```bash
pip install numpy matplotlib gurobipy
pip install ltlf2dfa pygraphviz networkx
pip install scipy casadi
```

### Gurobi License
This project uses Gurobi for optimization. You need:
1. A valid Gurobi license (academic licenses are free)
2. Set up your license credentials in `risk_LP/ltl_risk_LP.py`

## Usage

### Quick Start
Run the main experiment:
```bash
python experipment.py
```

### Online vs Offline Execution

The system supports two execution modes:

**Offline Mode (Recommended - Fast Startup)**
- Uses pre-computed control policy: `optimal_policy_risk_aware2.npy`
- Experiment starts immediately

**Online Mode (Compute Policy Real-time)**
- Computes control policy during runtime
- Takes approximately 4 minutes to start

**To switch modes:**
- **Offline**: Ensure the policy path is filled in `experipment.py` (default)
- **Online**: Comment out lines 117-118 and uncomment lines 105-115 in `experipment.py`

### Configuration

The main parameters can be configured in `experipment.py`:

#### Environment (MDP) Setup
```python
# ---------- MPD(Abstraction) --------------------
region_size = (50, 20)    # (length, width)
region_res = (10, 4)      # Resolution (length, width)
max_speed = 4             # Maximum velocity
speed_res = 1             # Speed resolution
```

#### LTL Specifications
```python
#---------- Specification Define --------------------
# Safety specification
safe_frag = LTL_Spec("G(~s) & G(~h)", AP_set=['s', 'h'])

# Mission specification
scltl_frag = LTL_Spec("F(t)", AP_set=['t'])
```

#### Labels and Costs
```python
# Speed limit test
speed_limit = 3
label_func = {(40, 50, 16, 20, 0, max_speed): "t",   # "t" for target
            (10, 50, 0, 20, speed_limit, max_speed): "s", # "s" for "speeding"
            (0, 50, 0, 20, max_speed, max_speed + speed_res): "h"}   # "h" for too high speed

cost_func = {'s': 3, 'h': 30}
```

## Related Work

This implementation extends the methodology from:
- [Risk-Aware Autonomous Driving with Linear Temporal Logic Specifications](https://github.com/Miracle-qi/Risk_LTL_Planning)
- Paper: "Risk-Aware Autonomous Driving with Linear Temporal Logic Specifications" (arXiv:2409.09769)

## Contact

- **Jingyuan Han** - Eindhoven University of Technology (j.han@student.tue.nl)



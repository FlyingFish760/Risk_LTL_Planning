# Risk Field Project

### Clear Readme is coming soon!!!

## Overview

Related Paper: [ArXiv:2409.09769](https://arxiv.org/abs/2409.09769)
The **Risk Field** project is a computational tool designed for analyzing and visualizing risk-related data fields. The project appears to involve numerical computations and is configured for use with IntelliJ/PyCharm, as indicated by the presence of `.idea` files.

## Repository Structure
- `.git/` - Contains version control metadata used by Git to track changes and manage the project history.
- `.idea/` - Stores IntelliJ/PyCharm project settings, configurations, and workspace-related data.
- `src/` - The main directory for source code files, including scripts and modules that implement the project's functionality.
- `data/` - Contains input data files, processed datasets, or any other files needed for computations.
- `docs/` - Includes project documentation, reports, and supplementary materials.
- `requirements.txt` - Lists the Python dependencies required to run the project.
- `LICENSE` - The MIT license file detailing the legal terms for usage and distribution.
- `README.md` - This file, providing an overview and instructions for the project.

### Python Files
- `controller.py` - Manages overall execution and control flow of the system.
- `reactive_LTL_main.py` - Main script handling reactive Linear Temporal Logic (LTL) computations.
- `abstraction/abstraction.py` - Implements abstraction mechanisms for system modeling.
- `abstraction/MDP.py` - Defines a Markov Decision Process (MDP) abstraction.
- `abstraction/prod_MDP.py` - Implements a product MDP for combining different abstractions.
- `risk_LP/risk_LP.py` - Core logic for solving risk-aware linear programming (LP) problems.
- `risk_LP/ltl_risk_LP.py` - Implements risk-aware LP methods under LTL constraints.
- `risk_LP/prod_auto.py` - Handles product automaton construction.
- `risk_LP/risk_field_plot.py` - Provides visualization tools for risk field data.
- `sim/simulator.py` - Simulates the environment and agent interactions.
- `sim/perception.py` - Manages sensor models and perception-related functionalities.
- `sim/visualizer.py` - Renders simulation results for analysis and debugging.
- `specification/specification.py` - Defines formal specifications for the system.
- `specification/DFA.py` - Implements deterministic finite automata (DFA) for specification handling.

## Getting Started
### Prerequisites
Ensure you have the following installed:
- Python 3.x (if applicable)
- Install dependencies: Gurobi, numpy, matplotlib, networkx, and scipy.


### Quick Start
Run the main script:
   ```bash
   python reactive_LTL_main.py
   ```
   
## License
This project is licensed under the MIT License.

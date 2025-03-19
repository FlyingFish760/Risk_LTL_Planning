# Risk-aware Planning Framework with LTL Specifications

## Overview

This repository provides a minimal example of the risk-aware planning framework proposed 
in the paper: [ArXiv:2409.09769](https://arxiv.org/abs/2409.09769), without relying on Carla. 
This example is designed to help users understand the proposed framework in a simplified setting.

## Repository Structure

- `reactive_LTL_main.py` - Main script handling reactive Linear Temporal Logic (LTL) computations.

### Abstraction Module (`abstraction/`)
- `abstraction.py` - Implements abstraction mechanisms for system modeling.
- `MDP.py` - Defines a Markov Decision Process (MDP) abstraction.
- `prod_MDP.py` - Implements a product MDP for combining different abstractions.

### Risk-aware Linear Programming Module (`risk_LP/`)
- `risk_LP.py` - Risk-aware linear programming (LP) problems.
- `ltl_risk_LP.py` - Implements risk-aware LP methods under LTL constraints.
- `prod_auto.py` - Construct product automaton.
- `risk_field_plot.py` - Provides visualization tools for risk field data.

### Simulation Module (`sim/`)
- `simulator.py` - Simulates the environment and vehicle dynamic.
- `perception.py` - Analog perception module to provide sensor information.
- `visualizer.py` - PLot simulation results.
- `controller.py` - Low-level MPC controller for the bicycle model.

### Specification Module (`specification/`)
- `specification.py` - Translate LTL specifications.
- `DFA.py` - Define deterministic finite automata (DFA).

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.x
- Required dependencies: `Gurobi`, `numpy`, `matplotlib`, `networkx`, and `scipy`.

### Quick Start

Run the main script:
```bash
python reactive_LTL_main.py
```

## License

This project is licensed under the MIT License.


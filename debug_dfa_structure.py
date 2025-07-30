#!/usr/bin/env python3

import numpy as np
from specification.specification import LTL_Spec
from ltlf2dfa.parser.ltlf import LTLfParser
import pygraphviz as pgv
from networkx.drawing.nx_agraph import from_agraph

def analyze_dfa_structure():
    # Create F(t) specification
    scltl_frag = LTL_Spec("F(t)", AP_set=['t'])
    
    print("=== F(t) DFA Structure Analysis ===")
    
    # Get the raw DFA from ltlf2dfa
    parser = LTLfParser()
    formula = parser("F(t)")
    dfa_dot = formula.to_dfa()
    
    print(f"Raw DFA DOT representation:")
    print(dfa_dot)
    print()
    
    # Parse the DFA graph
    agraph = pgv.AGraph(string=dfa_dot)
    dfa_graph = from_agraph(agraph)
    
    print(f"DFA Graph nodes: {list(dfa_graph.nodes)}")
    print(f"DFA Graph edges: {list(dfa_graph.edges)}")
    print()
    
    # Analyze edges
    edges_info = {edge: dfa_graph.edges[edge] for edge in dfa_graph.edges}
    print("Edge details:")
    for edge, info in edges_info.items():
        print(f"  {edge}: {info}")
    print()
    
    # Check sink state detection logic
    print("=== Sink State Detection Logic ===")
    sink_states = []
    initial_state = 'init'
    trans_condition = {}
    
    for edge, _obs in edges_info.items():
        if _obs.get('label') is not None:
            obs = _obs.get('label')
            trans_condition[edge] = obs
            print(f"Edge {edge} has label: '{obs}'")
        
        if (_obs.get('label') == "true") and (edge[0] == edge[1]):
            sink_states.append(edge[0])
            print(f"  -> SINK STATE DETECTED: {edge[0]} (self-loop with 'true')")
        
        if (edge[0] == 'init') and (_obs.get('label') is None):
            initial_state = edge[1]
            print(f"  -> INITIAL STATE: {edge[1]}")
    
    print(f"\nDetected sink states: {sink_states}")
    print(f"Initial state: {initial_state}")
    print()
    
    # Check the constructed DFA
    print("=== Constructed DFA Properties ===")
    print(f"DFA states: {scltl_frag.dfa.states}")
    print(f"DFA initial state: {scltl_frag.dfa.initial_state}")
    print(f"DFA sink states: {scltl_frag.dfa.sink_states}")
    print(f"DFA alphabet: {scltl_frag.dfa.alphabet}")
    print()
    
    print("DFA transitions:")
    for transition, next_state in scltl_frag.dfa.transitions.items():
        print(f"  {transition} -> {next_state}")
    print()
    
    # The key insight: check if sink state detection is correct
    print("=== CRITICAL ANALYSIS ===")
    print("For F(t) (Eventually t), the DFA should have:")
    print("- State 1: Initial state, not accepting, waiting for 't'")
    print("- State 2: Accepting state, reached after seeing 't', stays accepting forever")
    print()
    
    if '2' in scltl_frag.dfa.sink_states:
        print("✅ State 2 is correctly identified as a sink state")
        print("   This means once we see 't', we stay in the accepting state forever")
    else:
        print("❌ State 2 is NOT identified as a sink state - this is wrong!")
    
    print()
    print("The problem might be that the product automaton is creating states")
    print("with dfa_cs_state=2 without requiring them to have actually transitioned")
    print("there by seeing 't'. This is a construction bug, not a DFA bug.")

if __name__ == '__main__':
    analyze_dfa_structure() 
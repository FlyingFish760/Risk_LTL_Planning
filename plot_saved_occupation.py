#!/usr/bin/env python3
"""
Standalone script to plot saved occupation measure data
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from test_2 import load_occupation_measure
from risk_LP.occ_measure_plot import plot_occ_measure

def main():
    """Load and plot saved occupation measure data"""
    
    # Default filename - you can change this or pass as command line argument
    filename = "latest_occupation_data.pkl"
    
    # Allow filename to be passed as command line argument
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    
    try:
        # Load the saved occupation measure data
        print(f"Loading occupation measure data from: {filename}")
        occ_measure, prod_auto, abs_model = load_occupation_measure(filename)
        
        print(f"Loaded {len(occ_measure)} occupation measure entries")
        print("Generating plots...")
        
        # Plot the occupation measure
        plot_occ_measure(occ_measure, prod_auto, abs_model)
        
        print("Plotting complete!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you have run the main simulation first to generate the data file.")
    except Exception as e:
        print(f"Error loading or plotting data: {e}")

if __name__ == "__main__":
    main()

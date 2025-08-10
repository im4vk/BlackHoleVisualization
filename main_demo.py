#!/usr/bin/env python3
"""
Black Hole Simulation - Main Demo
=================================

This script demonstrates the complete black hole simulation system,
including mathematical foundation, physics engine, and visualization.

Run this script to generate all demonstration images and plots.
"""

import sys
import os
import time
import traceback

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import numpy as np
    import matplotlib.pyplot as plt
    from blackhole_equations import BlackHolePhysics, demonstrate_equations
    from physics_engine import demonstrate_physics_engine
    from visualization import create_demonstration_images
    
    print("All required modules imported successfully!")
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install required packages using:")
    print("pip install -r requirements.txt")
    sys.exit(1)


def main():
    """
    Main demonstration function
    """
    print("=" * 60)
    print("BLACK HOLE SIMULATION - COMPLETE DEMONSTRATION")
    print("=" * 60)
    print()
    
    start_time = time.time()
    
    try:
        # 1. Mathematical Foundation Demo
        print("STEP 1: Mathematical Foundation")
        print("-" * 30)
        demonstrate_equations()
        print("\n" + "="*60 + "\n")
        
        # 2. Physics Engine Demo  
        print("STEP 2: Physics Engine")
        print("-" * 30)
        demonstrate_physics_engine()
        print("\n" + "="*60 + "\n")
        
        # 3. Create Black Hole Images
        print("STEP 3: Black Hole Visualization")
        print("-" * 30)
        create_demonstration_images()
        print("\n" + "="*60 + "\n")
        
        # 4. Summary
        elapsed_time = time.time() - start_time
        print("DEMONSTRATION COMPLETE!")
        print("-" * 30)
        print(f"Total execution time: {elapsed_time:.2f} seconds")
        print()
        print("Generated files:")
        print("Mathematics:")
        print("  - effective_potential.png")
        print("  - deflection_angle.png")
        print()
        print("Physics Engine:")
        print("  - photon_trajectory.png")
        print("  - accretion_disk_profiles.png")
        print()
        print("Black Hole Images:")
        print("  - gravitational_lensing_demo.png")
        print("  - black_hole_shadow.png")
        print("  - eht_style_blackhole.png")
        print("  - distance_comparison.png")
        print("  - inclination_comparison.png")
        print()
        print("All images have been saved to the current directory.")
        print("These demonstrate:")
        print("  ✓ Einstein's General Relativity equations")
        print("  ✓ Schwarzschild metric and geodesics")
        print("  ✓ Gravitational lensing effects")
        print("  ✓ Event horizon and photon sphere")
        print("  ✓ Accretion disk physics")
        print("  ✓ Realistic black hole appearance")
        
    except Exception as e:
        print(f"An error occurred during demonstration: {e}")
        print("Traceback:")
        traceback.print_exc()
        return False
    
    return True


def quick_demo():
    """
    Quick demonstration with minimal computation
    """
    print("Quick Black Hole Demo")
    print("=" * 20)
    
    # Create black hole
    bh = BlackHolePhysics(mass=1.0, units="geometric")
    
    print(f"Solar mass black hole:")
    print(f"  Event horizon radius: {bh.rs:.3f} gravitational radii")
    print(f"  Photon sphere radius: {bh.photon_sphere_radius():.3f} rg")
    print(f"  Critical impact parameter: {bh.critical_impact_parameter():.3f} rg")
    
    # Test single ray
    try:
        b_test = bh.critical_impact_parameter() * 1.2
        initial_conditions = bh.photon_initial_conditions(
            r_start=50.0, theta_start=np.pi/2, phi_start=0.0, 
            impact_parameter=b_test
        )
        print(f"\nTest photon with impact parameter {b_test:.3f} rg:")
        print("Initial conditions set successfully")
        
        # Calculate deflection angle
        deflection = bh.deflection_angle(b_test)
        print(f"Predicted deflection angle: {np.degrees(deflection):.2f} degrees")
        
    except Exception as e:
        print(f"Quick test failed: {e}")
    
    print("\nFor full demonstration with images, run main()")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        quick_demo()
    else:
        success = main()
        if success:
            print("\n🌟 Black hole simulation completed successfully! 🌟")
        else:
            print("\n❌ Simulation encountered errors")
            sys.exit(1)
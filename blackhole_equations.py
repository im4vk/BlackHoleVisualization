"""
Black Hole Mathematical Foundation
=================================

This module contains the fundamental mathematical equations governing black holes,
based on Einstein's General Relativity theory.

Key Equations Implemented:
1. Schwarzschild Metric - describes spacetime geometry around a non-rotating black hole
2. Geodesic Equations - describe the motion of particles and light in curved spacetime
3. Gravitational Lensing - how light paths bend around massive objects
4. Event Horizon - boundary beyond which nothing can escape
5. Photon Sphere - unstable circular orbit for photons
"""

import numpy as np
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class BlackHolePhysics:
    """
    Implements the mathematical foundation for black hole physics
    """
    
    def __init__(self, mass: float = 1.0, spin: float = 0.0, units: str = "geometric"):
        """
        Initialize black hole parameters
        
        Args:
            mass: Black hole mass (in geometric units where G=c=1, or solar masses)
            spin: Dimensionless spin parameter (0 to 1, where 1 is maximally rotating)
            units: "geometric" (G=c=1) or "physical" (SI units)
        """
        self.mass = mass  # M
        self.spin = spin  # a = J/(Mc) in geometric units
        self.units = units
        
        # Physical constants in SI units
        self.G = 6.67430e-11  # m³/kg⋅s²
        self.c = 299792458    # m/s
        
        # Derived quantities
        if units == "geometric":
            self.rs = 2 * mass  # Schwarzschild radius in geometric units
            self.rg = mass      # Gravitational radius
        else:
            self.rs = 2 * self.G * mass / (self.c**2)  # Schwarzschild radius in meters
            self.rg = self.G * mass / (self.c**2)      # Gravitational radius in meters
    
    def schwarzschild_metric_coefficients(self, r: float, theta: float = np.pi/2) -> dict:
        """
        Schwarzschild metric coefficients in spherical coordinates (t, r, θ, φ)
        
        ds² = -(1-rs/r)dt² + (1-rs/r)⁻¹dr² + r²dθ² + r²sin²(θ)dφ²
        
        Args:
            r: Radial coordinate
            theta: Polar angle (default: equatorial plane)
            
        Returns:
            Dictionary of metric coefficients
        """
        if r <= self.rs:
            raise ValueError(f"Inside event horizon! r={r} <= rs={self.rs}")
        
        f = 1 - self.rs / r  # Metric function
        
        return {
            'g_tt': -f,                    # Time component
            'g_rr': 1 / f,                 # Radial component  
            'g_theta_theta': r**2,         # Polar component
            'g_phi_phi': r**2 * np.sin(theta)**2,  # Azimuthal component
            'f': f                         # Metric function for convenience
        }
    
    def geodesic_equations(self, state: np.ndarray, tau: float) -> np.ndarray:
        """
        Geodesic equations for particle motion in Schwarzschild spacetime
        
        State vector: [t, r, theta, phi, dt/dτ, dr/dτ, dθ/dτ, dφ/dτ]
        
        For null geodesics (photons): ds² = 0
        For timelike geodesics (massive particles): ds² = -1 (geometric units)
        
        Args:
            state: 8-component state vector [coordinates, 4-velocities]
            tau: Proper time parameter
            
        Returns:
            Derivative of state vector
        """
        # Ensure state is an array and handle edge cases
        if not hasattr(state, '__len__') or len(state) != 8:
            return np.zeros(8)  # Return zero derivatives if state is malformed
            
        t, r, theta, phi, dt_dtau, dr_dtau, dtheta_dtau, dphi_dtau = state
        
        # Avoid singularities
        if r <= self.rs * 1.01:
            r = self.rs * 1.01
        if theta <= 1e-6:
            theta = 1e-6
        if theta >= np.pi - 1e-6:
            theta = np.pi - 1e-6
            
        # Metric coefficients
        f = 1 - self.rs / r
        
        # Christoffel symbols (connection coefficients)
        # Only non-zero components for Schwarzschild metric
        
        # Time derivatives (coordinates)
        dt_dtau_new = dt_dtau
        dr_dtau_new = dr_dtau
        dtheta_dtau_new = dtheta_dtau
        dphi_dtau_new = dphi_dtau
        
        # 4-velocity derivatives (geodesic equation: d²x^μ/dτ² + Γ^μ_αβ (dx^α/dτ)(dx^β/dτ) = 0)
        
        # d²t/dτ² = -Γ^t_rr (dr/dτ)² - 2Γ^t_rt (dt/dτ)(dr/dτ)
        Gamma_t_rr = self.rs / (2 * r**2 * f)
        Gamma_t_rt = self.rs / (2 * r * (r - self.rs))
        
        d2t_dtau2 = -Gamma_t_rr * dr_dtau**2 - 2 * Gamma_t_rt * dt_dtau * dr_dtau
        
        # d²r/dτ² = -Γ^r_tt (dt/dτ)² - Γ^r_rr (dr/dτ)² - Γ^r_θθ (dθ/dτ)² - Γ^r_φφ (dφ/dτ)²
        Gamma_r_tt = self.rs * f / (2 * r**2)
        Gamma_r_rr = -self.rs / (2 * r * (r - self.rs))
        Gamma_r_theta_theta = -(r - self.rs)
        Gamma_r_phi_phi = -(r - self.rs) * np.sin(theta)**2
        
        d2r_dtau2 = (-Gamma_r_tt * dt_dtau**2 - Gamma_r_rr * dr_dtau**2 
                     - Gamma_r_theta_theta * dtheta_dtau**2 - Gamma_r_phi_phi * dphi_dtau**2)
        
        # d²θ/dτ² = -2Γ^θ_rθ (dr/dτ)(dθ/dτ) - Γ^θ_φφ (dφ/dτ)²
        Gamma_theta_r_theta = 1 / r
        Gamma_theta_phi_phi = -np.sin(theta) * np.cos(theta)
        
        d2theta_dtau2 = (-2 * Gamma_theta_r_theta * dr_dtau * dtheta_dtau 
                         - Gamma_theta_phi_phi * dphi_dtau**2)
        
        # d²φ/dτ² = -2Γ^φ_rφ (dr/dτ)(dφ/dτ) - 2Γ^φ_θφ (dθ/dτ)(dφ/dτ)
        Gamma_phi_r_phi = 1 / r
        Gamma_phi_theta_phi = np.cos(theta) / np.sin(theta)
        
        d2phi_dtau2 = (-2 * Gamma_phi_r_phi * dr_dtau * dphi_dtau 
                       - 2 * Gamma_phi_theta_phi * dtheta_dtau * dphi_dtau)
        
        return np.array([dt_dtau_new, dr_dtau_new, dtheta_dtau_new, dphi_dtau_new,
                        d2t_dtau2, d2r_dtau2, d2theta_dtau2, d2phi_dtau2])
    
    def photon_initial_conditions(self, r_start: float, theta_start: float, 
                                phi_start: float, impact_parameter: float) -> np.ndarray:
        """
        Set up initial conditions for photon geodesics
        
        For photons: ds² = 0 and we have conserved quantities:
        - Energy: E = (1 - rs/r) dt/dτ
        - Angular momentum: L = r²sin²(θ) dφ/dτ
        
        Args:
            r_start: Initial radial position
            theta_start: Initial polar angle  
            phi_start: Initial azimuthal angle
            impact_parameter: Impact parameter b = L/E
            
        Returns:
            Initial state vector for geodesic integration
        """
        if r_start <= self.rs:
            raise ValueError("Cannot start photon inside event horizon")
        
        # For photons coming from infinity with impact parameter b
        # E = 1 (normalized), L = b
        E = 1.0
        L = impact_parameter
        
        # Initial position
        t0, r0, theta0, phi0 = 0.0, r_start, theta_start, phi_start
        
        # From metric and conserved quantities
        f = 1 - self.rs / r0
        
        # dt/dτ = E / g_tt = E / (-f) = -E/f
        dt_dtau = E / f
        
        # dφ/dτ = L / g_φφ = L / (r²sin²θ)
        dphi_dtau = L / (r0**2 * np.sin(theta0)**2)
        
        # For radial motion: assume initially moving inward
        # From null condition: g_tt(dt/dτ)² + g_rr(dr/dτ)² + g_φφ(dφ/dτ)² = 0
        # Solving for dr/dτ
        dr_dtau_squared = (E**2 - f * L**2 / r0**2) / f
        
        if dr_dtau_squared < 0:
            # Photon cannot reach this radius
            raise ValueError(f"Invalid trajectory: cannot reach r={r0} with b={impact_parameter}")
        
        dr_dtau = -np.sqrt(dr_dtau_squared)  # Negative for inward motion
        
        # Assume motion in equatorial plane
        dtheta_dtau = 0.0
        
        return np.array([t0, r0, theta0, phi0, dt_dtau, dr_dtau, dtheta_dtau, dphi_dtau])
    
    def effective_potential_photon(self, r: np.ndarray, L: float) -> np.ndarray:
        """
        Effective potential for photon motion
        
        V_eff(r) = (1 - rs/r) * L²/r²
        
        Args:
            r: Radial coordinate array
            L: Angular momentum
            
        Returns:
            Effective potential array
        """
        f = 1 - self.rs / r
        return f * L**2 / r**2
    
    def photon_sphere_radius(self) -> float:
        """
        Radius of the photon sphere (unstable circular orbit for photons)
        
        For Schwarzschild: r_ph = 3rs/2 = 3M
        """
        return 1.5 * self.rs
    
    def critical_impact_parameter(self) -> float:
        """
        Critical impact parameter for photon capture
        
        Photons with b < b_crit will be captured by the black hole
        For Schwarzschild: b_crit = 3√3 * rs/2 ≈ 2.598 * rs
        """
        return 3 * np.sqrt(3) * self.rs / 2
    
    def deflection_angle(self, impact_parameter: float) -> float:
        """
        Calculate gravitational lensing deflection angle
        
        For weak field (r >> rs): δφ ≈ 4M/b
        For strong field: need numerical integration
        
        Args:
            impact_parameter: Impact parameter b
            
        Returns:
            Deflection angle in radians
        """
        if impact_parameter < self.critical_impact_parameter():
            return np.inf  # Photon is captured
        
        # Weak field approximation
        if impact_parameter > 10 * self.rs:
            return 4 * self.rg / impact_parameter
        
        # For strong field, need numerical integration (simplified here)
        b_crit = self.critical_impact_parameter()
        weak_deflection = 4 * self.rg / impact_parameter
        
        # Empirical correction for strong field
        correction_factor = 1 + (b_crit / impact_parameter)**2
        return weak_deflection * correction_factor


def demonstrate_equations():
    """
    Demonstrate the key mathematical concepts
    """
    print("Black Hole Mathematical Foundation Demo")
    print("=" * 40)
    
    # Create a solar mass black hole
    bh = BlackHolePhysics(mass=1.0, units="geometric")
    
    print(f"Black hole mass: {bh.mass} M☉")
    print(f"Schwarzschild radius: {bh.rs:.3f} (geometric units)")
    print(f"Photon sphere radius: {bh.photon_sphere_radius():.3f}")
    print(f"Critical impact parameter: {bh.critical_impact_parameter():.3f}")
    
    # Test metric coefficients
    r_test = 10.0  # 10 gravitational radii
    metric = bh.schwarzschild_metric_coefficients(r_test)
    print(f"\nMetric coefficients at r = {r_test}:")
    for key, value in metric.items():
        print(f"  {key}: {value:.6f}")
    
    # Plot effective potential
    r_range = np.linspace(1.5 * bh.rs, 20 * bh.rs, 1000)
    L_values = [3.0, 4.0, 5.0, 6.0]
    
    plt.figure(figsize=(10, 6))
    for L in L_values:
        V_eff = bh.effective_potential_photon(r_range, L)
        plt.plot(r_range / bh.rs, V_eff, label=f'L = {L}')
    
    plt.axvline(bh.photon_sphere_radius() / bh.rs, color='red', linestyle='--', 
                label='Photon sphere')
    plt.axvline(1.0, color='black', linestyle='-', label='Event horizon')
    
    plt.xlabel('r / rs')
    plt.ylabel('Effective Potential')
    plt.title('Photon Effective Potential in Schwarzschild Spacetime')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(1, 10)
    plt.ylim(0, 0.5)
    
    plt.tight_layout()
    plt.savefig('effective_potential.png', dpi=300)
    print(f"\nEffective potential plot saved to effective_potential.png")
    
    # Deflection angles
    b_range = np.linspace(2.7 * bh.rs, 20 * bh.rs, 100)
    deflections = [bh.deflection_angle(b) for b in b_range]
    
    plt.figure(figsize=(10, 6))
    plt.plot(b_range / bh.rs, np.degrees(deflections))
    plt.axvline(bh.critical_impact_parameter() / bh.rs, color='red', linestyle='--',
                label='Critical impact parameter')
    
    plt.xlabel('Impact Parameter b / rs')
    plt.ylabel('Deflection Angle (degrees)')
    plt.title('Gravitational Lensing Deflection Angle')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('deflection_angle.png', dpi=300)
    print(f"Deflection angle plot saved to deflection_angle.png")


if __name__ == "__main__":
    demonstrate_equations()
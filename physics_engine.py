"""
Black Hole Physics Engine
=========================

This module implements the physics engine for black hole simulation,
including ray tracing through curved spacetime, geodesic integration,
and gravitational lensing calculations.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from typing import Tuple, List, Optional, Dict
import multiprocessing as mp
from functools import partial
import time

from blackhole_equations import BlackHolePhysics

class RayTracer:
    """
    Ray tracing engine for photons in curved spacetime around black holes
    """
    
    def __init__(self, black_hole: BlackHolePhysics, integration_method: str = 'RK45'):
        """
        Initialize ray tracer
        
        Args:
            black_hole: BlackHolePhysics instance
            integration_method: ODE integration method
        """
        self.bh = black_hole
        self.integration_method = integration_method
        self.max_steps = 10000
        self.atol = 1e-12
        self.rtol = 1e-9
    
    def trace_ray(self, initial_conditions: np.ndarray, 
                  max_tau: float = 100.0, 
                  stop_at_horizon: bool = True) -> Dict:
        """
        Trace a single photon ray through spacetime
        
        Args:
            initial_conditions: Initial state vector [t, r, θ, φ, dt/dτ, dr/dτ, dθ/dτ, dφ/dτ]
            max_tau: Maximum proper time for integration
            stop_at_horizon: Whether to stop integration at event horizon
            
        Returns:
            Dictionary with trajectory data and metadata
        """
        def event_horizon_event(tau, y):
            """Stop integration when approaching event horizon"""
            return y[1] - self.bh.rs * 1.01  # Stop just outside horizon
        
        def escape_event(tau, y):
            """Stop integration when photon escapes to large distance"""
            return y[1] - 1000 * self.bh.rs  # Stop at large distance
        
        events = []
        if stop_at_horizon:
            event_horizon_event.terminal = True
            event_horizon_event.direction = -1
            events.append(event_horizon_event)
        
        escape_event.terminal = True
        escape_event.direction = 1
        events.append(escape_event)
        
        try:
            # Integrate geodesic equations
            solution = solve_ivp(
                self.bh.geodesic_equations,
                [0, max_tau],
                initial_conditions,
                method=self.integration_method,
                events=events,
                max_step=0.1,
                atol=self.atol,
                rtol=self.rtol,
                dense_output=True
            )
            
            if not solution.success:
                return {
                    'success': False,
                    'message': solution.message,
                    'trajectory': None
                }
            
            # Extract trajectory
            trajectory = {
                't': solution.y[0],
                'r': solution.y[1], 
                'theta': solution.y[2],
                'phi': solution.y[3],
                'dt_dtau': solution.y[4],
                'dr_dtau': solution.y[5],
                'dtheta_dtau': solution.y[6],
                'dphi_dtau': solution.y[7],
                'tau': solution.t
            }
            
            # Determine final state
            final_r = solution.y[1, -1]
            if final_r <= self.bh.rs * 1.02:
                outcome = 'captured'
            elif final_r >= 1000 * self.bh.rs:
                outcome = 'escaped'
            else:
                outcome = 'incomplete'
            
            return {
                'success': True,
                'trajectory': trajectory,
                'outcome': outcome,
                'final_position': {
                    't': solution.y[0, -1],
                    'r': solution.y[1, -1],
                    'theta': solution.y[2, -1], 
                    'phi': solution.y[3, -1]
                },
                'integration_info': {
                    'n_steps': len(solution.t),
                    'events': solution.t_events
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"Integration failed: {str(e)}",
                'trajectory': None
            }
    
    def trace_ray_grid(self, observer_distance: float, 
                      screen_size: float, 
                      resolution: int,
                      parallel: bool = True) -> Dict:
        """
        Trace rays from observer through a screen to create an image
        
        Args:
            observer_distance: Distance of observer from black hole
            screen_size: Half-width of observation screen
            resolution: Number of pixels per side
            parallel: Whether to use parallel processing
            
        Returns:
            Dictionary with ray tracing results
        """
        print(f"Tracing {resolution}x{resolution} rays from distance {observer_distance}")
        
        # Create impact parameter grid
        b_values = np.linspace(-screen_size, screen_size, resolution)
        impact_parameters = []
        pixel_coords = []
        
        for i, bx in enumerate(b_values):
            for j, by in enumerate(b_values):
                b = np.sqrt(bx**2 + by**2)
                if b > 0:  # Avoid exactly zero impact parameter
                    impact_parameters.append(b)
                    pixel_coords.append((i, j, bx, by))
        
        print(f"Total rays to trace: {len(impact_parameters)}")
        
        # Trace rays
        start_time = time.time()
        
        if parallel and len(impact_parameters) > 100:
            # Parallel processing
            with mp.Pool() as pool:
                trace_func = partial(self._trace_single_ray, observer_distance)
                results = pool.map(trace_func, impact_parameters)
        else:
            # Serial processing
            results = [self._trace_single_ray(observer_distance, b) 
                      for b in impact_parameters]
        
        elapsed_time = time.time() - start_time
        print(f"Ray tracing completed in {elapsed_time:.2f} seconds")
        
        # Organize results
        image_data = np.zeros((resolution, resolution, 4))  # RGBA
        deflection_angles = np.zeros((resolution, resolution))
        
        for idx, (result, (i, j, bx, by)) in enumerate(zip(results, pixel_coords)):
            if result.get('success', False) and result.get('outcome') == 'escaped':
                # Calculate deflection
                final_pos = result.get('final_position', {})
                final_phi = final_pos.get('phi', 0)
                deflection = abs(final_phi)
                deflection_angles[i, j] = deflection
                
                # Color based on deflection (red for high deflection)
                intensity = min(deflection / np.pi, 1.0)
                image_data[i, j] = [intensity, 1-intensity, 0, 1]  # Red-green gradient
            elif result.get('outcome') == 'captured':
                # Black for captured rays
                image_data[i, j] = [0, 0, 0, 1]
            else:
                # Blue for other cases
                image_data[i, j] = [0, 0, 1, 1]
        
        return {
            'image_data': image_data,
            'deflection_angles': deflection_angles,
            'impact_parameters': np.array(impact_parameters),
            'pixel_coords': pixel_coords,
            'results': results,
            'parameters': {
                'observer_distance': observer_distance,
                'screen_size': screen_size,
                'resolution': resolution
            },
            'timing': elapsed_time
        }
    
    def _trace_single_ray(self, observer_distance: float, impact_parameter: float) -> Dict:
        """
        Helper function to trace a single ray
        """
        try:
            # Set up initial conditions for photon at observer
            initial_conditions = self.bh.photon_initial_conditions(
                r_start=observer_distance,
                theta_start=np.pi/2,  # Equatorial plane
                phi_start=0.0,
                impact_parameter=impact_parameter
            )
            
            # Trace the ray
            result = self.trace_ray(initial_conditions, max_tau=200.0)
            
            # Ensure result has all required keys
            if not isinstance(result, dict):
                result = {'success': False, 'outcome': 'error'}
            
            if 'outcome' not in result:
                if result.get('success', False):
                    result['outcome'] = 'escaped'  # Default for successful rays
                else:
                    result['outcome'] = 'error'
                    
            return result
            
        except Exception as e:
            return {
                'success': False,
                'message': f"Ray tracing failed: {str(e)}",
                'trajectory': None,
                'outcome': 'error',
                'final_position': {'phi': 0}  # Default values
            }


class AccretionDisk:
    """
    Models the accretion disk around a black hole
    """
    
    def __init__(self, black_hole: BlackHolePhysics, 
                 inner_radius: Optional[float] = None,
                 outer_radius: float = 20.0,
                 disk_luminosity: float = 1.0):
        """
        Initialize accretion disk
        
        Args:
            black_hole: BlackHolePhysics instance
            inner_radius: Inner edge (default: ISCO radius)
            outer_radius: Outer edge in gravitational radii
            disk_luminosity: Total luminosity scaling factor
        """
        self.bh = black_hole
        self.inner_radius = inner_radius or self._isco_radius()
        self.outer_radius = outer_radius * self.bh.rg
        self.luminosity = disk_luminosity
    
    def _isco_radius(self) -> float:
        """
        Innermost Stable Circular Orbit (ISCO) radius
        For Schwarzschild: r_isco = 6M
        """
        return 6 * self.bh.rg
    
    def temperature_profile(self, r: np.ndarray) -> np.ndarray:
        """
        Temperature profile of the accretion disk
        
        Simplified model: T(r) ∝ r^(-3/4) for standard thin disk
        
        Args:
            r: Radial coordinates
            
        Returns:
            Temperature array
        """
        # Mask for disk region
        mask = (r >= self.inner_radius) & (r <= self.outer_radius)
        T = np.zeros_like(r)
        
        # Temperature profile (normalized)
        T0 = 1.0  # Normalization constant
        T[mask] = T0 * (r[mask] / self.inner_radius)**(-0.75)
        
        return T
    
    def surface_brightness(self, r: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Surface brightness of the disk
        
        Args:
            r, theta: Coordinates
            
        Returns:
            Surface brightness array
        """
        # Temperature
        T = self.temperature_profile(r)
        
        # Stefan-Boltzmann law: brightness ∝ T^4
        brightness = T**4
        
        # Geometric factor for disk thickness
        disk_height = 0.1 * self.bh.rg  # Thin disk approximation
        height_factor = np.exp(-0.5 * (r * np.sin(theta - np.pi/2) / disk_height)**2)
        
        return brightness * height_factor
    
    def doppler_factor(self, r: np.ndarray, phi: np.ndarray, 
                       observer_angle: float = 0.0) -> np.ndarray:
        """
        Doppler boosting/de-boosting due to disk rotation
        
        Args:
            r, phi: Coordinates
            observer_angle: Viewing angle relative to disk normal
            
        Returns:
            Doppler factor array
        """
        # Keplerian velocity
        v_phi = np.sqrt(self.bh.rg / r)  # Orbital velocity
        
        # Line-of-sight velocity component
        v_los = v_phi * np.sin(observer_angle) * np.sin(phi)
        
        # Relativistic Doppler factor
        gamma = 1 / np.sqrt(1 - v_phi**2)  # Lorentz factor
        doppler = gamma * (1 + v_los)  # Simplified
        
        return doppler


def demonstrate_physics_engine():
    """
    Demonstrate the physics engine capabilities
    """
    print("Black Hole Physics Engine Demo")
    print("=" * 35)
    
    # Create black hole
    bh = BlackHolePhysics(mass=1.0, units="geometric")
    ray_tracer = RayTracer(bh)
    
    print(f"Black hole mass: {bh.mass} M☉")
    print(f"Event horizon: {bh.rs:.3f}")
    print(f"Photon sphere: {bh.photon_sphere_radius():.3f}")
    
    # Test single ray tracing
    print("\nTracing single photon ray...")
    
    try:
        # Ray with impact parameter slightly larger than critical
        b_test = bh.critical_impact_parameter() * 1.1
        initial_conditions = bh.photon_initial_conditions(
            r_start=100.0,
            theta_start=np.pi/2,
            phi_start=0.0,
            impact_parameter=b_test
        )
        
        result = ray_tracer.trace_ray(initial_conditions, max_tau=50.0)
        
        if result['success']:
            print(f"Ray outcome: {result['outcome']}")
            print(f"Final position: r={result['final_position']['r']:.3f}")
            print(f"Integration steps: {result['integration_info']['n_steps']}")
            
            # Plot trajectory
            traj = result['trajectory']
            
            plt.figure(figsize=(12, 4))
            
            # Radial trajectory
            plt.subplot(1, 3, 1)
            plt.plot(traj['tau'], traj['r'])
            plt.axhline(bh.rs, color='red', linestyle='--', label='Event horizon')
            plt.axhline(bh.photon_sphere_radius(), color='orange', linestyle='--', 
                       label='Photon sphere')
            plt.xlabel('Proper time τ')
            plt.ylabel('Radial coordinate r')
            plt.title('Radial Motion')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Angular trajectory  
            plt.subplot(1, 3, 2)
            plt.plot(traj['tau'], traj['phi'])
            plt.xlabel('Proper time τ')
            plt.ylabel('Azimuthal angle φ')
            plt.title('Angular Motion')
            plt.grid(True, alpha=0.3)
            
            # Orbit in r-φ plane
            plt.subplot(1, 3, 3)
            x = traj['r'] * np.cos(traj['phi'])
            y = traj['r'] * np.sin(traj['phi'])
            plt.plot(x, y, 'b-', alpha=0.7)
            
            # Draw black hole and key radii
            circle_bh = plt.Circle((0, 0), bh.rs, color='black', alpha=0.8)
            circle_ph = plt.Circle((0, 0), bh.photon_sphere_radius(), 
                                 fill=False, color='orange', linestyle='--')
            plt.gca().add_patch(circle_bh)
            plt.gca().add_patch(circle_ph)
            
            plt.axis('equal')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Photon Trajectory')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('photon_trajectory.png', dpi=300)
            print("Photon trajectory plot saved to photon_trajectory.png")
            
        else:
            print(f"Ray tracing failed: {result['message']}")
            
    except Exception as e:
        print(f"Single ray test failed: {e}")
    
    # Test accretion disk
    print("\nTesting accretion disk model...")
    
    disk = AccretionDisk(bh, outer_radius=15.0)
    
    r_disk = np.linspace(disk.inner_radius, disk.outer_radius, 100)
    theta_disk = np.ones_like(r_disk) * np.pi/2  # Equatorial plane
    
    temp_profile = disk.temperature_profile(r_disk)
    brightness_profile = disk.surface_brightness(r_disk, theta_disk)
    
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(r_disk / bh.rg, temp_profile)
    plt.xlabel('r / rg')
    plt.ylabel('Temperature (normalized)')
    plt.title('Accretion Disk Temperature Profile')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    plt.plot(r_disk / bh.rg, brightness_profile)
    plt.xlabel('r / rg')
    plt.ylabel('Surface Brightness')
    plt.title('Accretion Disk Brightness Profile')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('accretion_disk_profiles.png', dpi=300)
    print("Accretion disk profiles saved to accretion_disk_profiles.png")
    
    print(f"\nISCO radius: {disk.inner_radius / bh.rg:.1f} rg")
    print(f"Disk outer radius: {disk.outer_radius / bh.rg:.1f} rg")


if __name__ == "__main__":
    demonstrate_physics_engine()
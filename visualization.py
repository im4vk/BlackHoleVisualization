"""
Black Hole Visualization System
==============================

This module creates realistic black hole images showing gravitational lensing,
event horizon, photon sphere, and accretion disk effects.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Circle
from scipy import ndimage
from typing import Tuple, List, Optional, Dict
import seaborn as sns

from blackhole_equations import BlackHolePhysics
from physics_engine import RayTracer, AccretionDisk

class BlackHoleRenderer:
    """
    High-level renderer for creating black hole images
    """
    
    def __init__(self, black_hole: BlackHolePhysics):
        """
        Initialize renderer
        
        Args:
            black_hole: BlackHolePhysics instance
        """
        self.bh = black_hole
        self.ray_tracer = RayTracer(black_hole)
        self.accretion_disk = AccretionDisk(black_hole)
        
        # Default rendering parameters
        self.default_resolution = 256
        self.default_observer_distance = 50.0
        self.default_screen_size = 20.0
        
    def create_lensing_image(self, 
                           resolution: int = 256,
                           observer_distance: float = 50.0,
                           screen_size: float = 20.0,
                           background_type: str = 'grid') -> Dict:
        """
        Create an image showing gravitational lensing effects
        
        Args:
            resolution: Image resolution (pixels per side)
            observer_distance: Distance from black hole to observer
            screen_size: Half-width of observation screen
            background_type: Type of background ('grid', 'stars', 'uniform')
            
        Returns:
            Dictionary with image data and metadata
        """
        print(f"Rendering gravitational lensing image ({resolution}x{resolution})...")
        
        # Create background pattern
        background = self._create_background(resolution, background_type)
        
        # Trace rays through spacetime
        ray_results = self.ray_tracer.trace_ray_grid(
            observer_distance=observer_distance,
            screen_size=screen_size,
            resolution=resolution,
            parallel=False  # Disable parallel for demo stability
        )
        
        # Create lensed image
        lensed_image = self._apply_lensing(background, ray_results)
        
        return {
            'image': lensed_image,
            'background': background,
            'ray_results': ray_results,
            'parameters': {
                'resolution': resolution,
                'observer_distance': observer_distance,
                'screen_size': screen_size,
                'background_type': background_type
            }
        }
    
    def create_accretion_disk_image(self,
                                  resolution: int = 512,
                                  observer_distance: float = 30.0,
                                  screen_size: float = 15.0,
                                  inclination_angle: float = np.pi/4,
                                  disk_temperature_scale: float = 1.0) -> Dict:
        """
        Create an image of black hole with accretion disk
        
        Args:
            resolution: Image resolution
            observer_distance: Distance to observer
            screen_size: Field of view half-width
            inclination_angle: Disk inclination relative to line of sight
            disk_temperature_scale: Scale factor for disk temperature
            
        Returns:
            Dictionary with image and metadata
        """
        print(f"Rendering accretion disk image ({resolution}x{resolution})...")
        
        # Create coordinate grids
        x = np.linspace(-screen_size, screen_size, resolution)
        y = np.linspace(-screen_size, screen_size, resolution)
        X, Y = np.meshgrid(x, y)
        
        # Convert to polar coordinates in disk plane
        r_projected = np.sqrt(X**2 + Y**2)
        phi_projected = np.arctan2(Y, X)
        
        # Account for inclination
        # This is a simplified projection - full ray tracing would be more accurate
        r_disk = r_projected / np.sin(inclination_angle)
        
        # Calculate disk emission
        disk_image = self._render_accretion_disk(
            r_disk, phi_projected, inclination_angle, disk_temperature_scale
        )
        
        # Add black hole shadow
        shadow_mask = r_projected < self.bh.rs * 2.6  # Approximate shadow size
        disk_image[shadow_mask] = 0.0
        
        # Add photon sphere ring (enhanced brightness)
        photon_ring_mask = (
            (r_projected > self.bh.photon_sphere_radius() * 0.95) & 
            (r_projected < self.bh.photon_sphere_radius() * 1.05)
        )
        disk_image[photon_ring_mask] *= 2.0
        
        return {
            'image': disk_image,
            'coordinates': {'X': X, 'Y': Y, 'r': r_projected, 'phi': phi_projected},
            'parameters': {
                'resolution': resolution,
                'observer_distance': observer_distance,
                'screen_size': screen_size,
                'inclination_angle': inclination_angle,
                'disk_temperature_scale': disk_temperature_scale
            }
        }
    
    def create_event_horizon_telescope_image(self,
                                           resolution: int = 512,
                                           observer_distance: float = 30.0,
                                           screen_size: float = 10.0) -> Dict:
        """
        Create an EHT-style black hole image with shadow and photon ring
        
        Args:
            resolution: Image resolution
            observer_distance: Distance to observer  
            screen_size: Field of view
            
        Returns:
            Image dictionary
        """
        print(f"Rendering EHT-style image ({resolution}x{resolution})...")
        
        # Create base accretion disk image
        disk_result = self.create_accretion_disk_image(
            resolution=resolution,
            observer_distance=observer_distance,
            screen_size=screen_size,
            inclination_angle=np.pi/3,  # 60 degree inclination
            disk_temperature_scale=1.5
        )
        
        image = disk_result['image']
        X, Y = disk_result['coordinates']['X'], disk_result['coordinates']['Y']
        r = np.sqrt(X**2 + Y**2)
        
        # Enhanced black hole shadow
        shadow_radius = 2.6 * self.bh.rs  # Theoretical shadow size
        shadow_mask = r < shadow_radius
        image[shadow_mask] = 0.0
        
        # Bright photon ring
        ring_inner = 2.9 * self.bh.rs
        ring_outer = 3.2 * self.bh.rs
        ring_mask = (r > ring_inner) & (r < ring_outer)
        
        # Create asymmetric brightness (Doppler boosting)
        phi = np.arctan2(Y, X)
        doppler_boost = 1 + 0.5 * np.sin(phi)  # Simplified Doppler effect
        
        image[ring_mask] = 3.0 * doppler_boost[ring_mask]
        
        # Add some turbulence/hotspots
        noise = np.random.random((resolution, resolution)) * 0.3
        turbulence_mask = (r > shadow_radius) & (r < 8 * self.bh.rs)
        image[turbulence_mask] += noise[turbulence_mask] * image[turbulence_mask]
        
        # Smooth the image slightly
        image = ndimage.gaussian_filter(image, sigma=0.8)
        
        return {
            'image': image,
            'shadow_radius': shadow_radius,
            'ring_radius': (ring_inner + ring_outer) / 2,
            'coordinates': disk_result['coordinates'],
            'parameters': disk_result['parameters']
        }
    
    def _create_background(self, resolution: int, background_type: str) -> np.ndarray:
        """
        Create background pattern for lensing demonstration
        """
        if background_type == 'grid':
            # Coordinate grid pattern
            background = np.zeros((resolution, resolution, 3))
            
            # Grid lines
            grid_spacing = resolution // 16
            for i in range(0, resolution, grid_spacing):
                background[i, :] = [1, 1, 1]  # Horizontal lines
                background[:, i] = [1, 1, 1]  # Vertical lines
                
        elif background_type == 'stars':
            # Random star field
            background = np.zeros((resolution, resolution, 3))
            n_stars = resolution * 2
            
            for _ in range(n_stars):
                x = np.random.randint(0, resolution)
                y = np.random.randint(0, resolution)
                brightness = np.random.random() * 0.8 + 0.2
                background[y, x] = [brightness, brightness, brightness]
                
        elif background_type == 'uniform':
            # Uniform background with gradient
            background = np.ones((resolution, resolution, 3)) * 0.2
            
        else:
            # Default: simple gradient
            x = np.linspace(0, 1, resolution)
            y = np.linspace(0, 1, resolution)
            X, Y = np.meshgrid(x, y)
            
            background = np.zeros((resolution, resolution, 3))
            background[:, :, 0] = X  # Red gradient
            background[:, :, 1] = Y  # Green gradient
            background[:, :, 2] = 0.5  # Constant blue
            
        return background
    
    def _apply_lensing(self, background: np.ndarray, ray_results: Dict) -> np.ndarray:
        """
        Apply gravitational lensing to background image
        """
        resolution = background.shape[0]
        lensed_image = np.zeros_like(background)
        
        # Map each pixel based on ray tracing results
        for idx, (result, (i, j, bx, by)) in enumerate(
            zip(ray_results['results'], ray_results['pixel_coords'])):
            
            if result['success'] and result['outcome'] == 'escaped':
                # Ray escaped - map to deflected position
                final_phi = result['final_position']['phi']
                
                # Calculate source position (simplified)
                # This is where the ray would have come from without lensing
                source_x = int(i + final_phi * 10) % resolution
                source_y = int(j + final_phi * 5) % resolution
                
                # Copy pixel from source to observer position
                lensed_image[i, j] = background[source_x, source_y]
                
            elif result['outcome'] == 'captured':
                # Ray captured by black hole - black pixel
                lensed_image[i, j] = [0, 0, 0]
            else:
                # Default background
                lensed_image[i, j] = background[i, j] * 0.5
                
        return lensed_image
    
    def _render_accretion_disk(self, r: np.ndarray, phi: np.ndarray, 
                              inclination: float, temperature_scale: float) -> np.ndarray:
        """
        Render accretion disk emission
        """
        # Disk temperature and brightness
        temp = self.accretion_disk.temperature_profile(r) * temperature_scale
        
        # Only render within disk boundaries
        disk_mask = (r >= self.accretion_disk.inner_radius) & (r <= self.accretion_disk.outer_radius)
        
        # Create emission image
        emission = np.zeros_like(r)
        emission[disk_mask] = temp[disk_mask]**4  # Stefan-Boltzmann law
        
        # Apply Doppler boosting due to rotation
        doppler_factor = self.accretion_disk.doppler_factor(r, phi, inclination)
        emission *= doppler_factor**3  # Relativistic beaming
        
        # Add some turbulence
        turbulence = 1 + 0.2 * np.sin(5 * phi) * np.cos(3 * r / self.bh.rg)
        emission *= turbulence
        
        return emission
    
    def save_image(self, image_data: np.ndarray, filename: str, 
                   title: str = "Black Hole Simulation",
                   colormap: str = 'hot',
                   add_scale: bool = True) -> None:
        """
        Save image with proper formatting and scale
        """
        plt.figure(figsize=(10, 8))
        
        if len(image_data.shape) == 3:
            # RGB image
            plt.imshow(image_data, origin='lower')
        else:
            # Grayscale/intensity image
            plt.imshow(image_data, origin='lower', cmap=colormap)
            plt.colorbar(label='Intensity')
        
        plt.title(title, fontsize=16, fontweight='bold')
        
        if add_scale:
            # Add scale bar
            scale_length = image_data.shape[0] // 10
            scale_start = image_data.shape[0] // 20
            
            plt.plot([scale_start, scale_start + scale_length], 
                    [scale_start, scale_start], 'white', linewidth=3)
            plt.text(scale_start + scale_length//2, scale_start + 5, 
                    f'{scale_length/10:.0f} rs', 
                    color='white', ha='center', fontweight='bold')
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='black')

        plt.close()
        
        print(f"Image saved to {filename}")


def create_demonstration_images():
    """
    Create a series of demonstration images showing different aspects
    """
    print("Creating Black Hole Demonstration Images")
    print("=" * 40)
    
    # Initialize black hole and renderer
    bh = BlackHolePhysics(mass=1.0, units="geometric")
    renderer = BlackHoleRenderer(bh)
    
    # 1. Gravitational Lensing Demo
    print("\n1. Creating gravitational lensing demonstration...")
    lensing_result = renderer.create_lensing_image(
        resolution=64,  # Much smaller for faster demo
        observer_distance=50.0,
        screen_size=20.0,
        background_type='grid'
    )
    
    renderer.save_image(
        lensing_result['image'],
        'gravitational_lensing_demo.png',
        'Gravitational Lensing by Black Hole',
        colormap='viridis'
    )
    
    # 2. Black Hole Shadow
    print("\n2. Creating black hole shadow image...")
    shadow_result = renderer.create_accretion_disk_image(
        resolution=128,  # Reduced for faster demo
        observer_distance=30.0,
        screen_size=10.0,
        inclination_angle=np.pi/4
    )
    
    renderer.save_image(
        shadow_result['image'],
        'black_hole_shadow.png',
        'Black Hole with Accretion Disk',
        colormap='hot'
    )
    
    # 3. Event Horizon Telescope Style
    print("\n3. Creating EHT-style image...")
    eht_result = renderer.create_event_horizon_telescope_image(
        resolution=128,  # Reduced for faster demo
        observer_distance=30.0,
        screen_size=8.0
    )
    
    renderer.save_image(
        eht_result['image'],
        'eht_style_blackhole.png',
        'Event Horizon Telescope Style Image',
        colormap='afmhot'
    )
    
    # 4. Comparison at different distances
    print("\n4. Creating distance comparison...")
    distances = [20.0, 50.0, 100.0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, dist in enumerate(distances):
        result = renderer.create_accretion_disk_image(
            resolution=64,  # Reduced for faster demo
            observer_distance=dist,
            screen_size=15.0,
            inclination_angle=np.pi/3
        )
        
        axes[i].imshow(result['image'], origin='lower', cmap='hot')
        axes[i].set_title(f'Distance: {dist} rg')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('distance_comparison.png', dpi=300, bbox_inches='tight', facecolor='black')

    plt.close()
    print("Distance comparison saved to distance_comparison.png")
    
    # 5. Different inclination angles
    print("\n5. Creating inclination comparison...")
    inclinations = [np.pi/6, np.pi/3, np.pi/2]
    inclination_names = ['30°', '60°', '90°']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (inc, name) in enumerate(zip(inclinations, inclination_names)):
        result = renderer.create_accretion_disk_image(
            resolution=64,  # Reduced for faster demo
            observer_distance=30.0,
            screen_size=10.0,
            inclination_angle=inc
        )
        
        axes[i].imshow(result['image'], origin='lower', cmap='hot')
        axes[i].set_title(f'Inclination: {name}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('inclination_comparison.png', dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()
    print("Inclination comparison saved to inclination_comparison.png")
    
    print("\nAll demonstration images created successfully!")
    print("\nGenerated files:")
    print("- gravitational_lensing_demo.png")
    print("- black_hole_shadow.png") 
    print("- eht_style_blackhole.png")
    print("- distance_comparison.png")
    print("- inclination_comparison.png")


if __name__ == "__main__":
    create_demonstration_images()
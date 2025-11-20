import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union
import math
from scipy.interpolate import griddata

def make_circular_actuator_grid(telescope_diameter: float, grid_size: int) -> np.ndarray:
    """Return actuator coordinates for a square grid clipped by the circular pupil."""
    pupil_radius = 0.5 * telescope_diameter
    axis = np.linspace(-pupil_radius, pupil_radius, grid_size)
    xx, yy = np.meshgrid(axis, axis)
    coords = np.column_stack((xx.ravel(), yy.ravel()))
    radius_sq = pupil_radius**2
    # Small epsilon to include points exactly on the edge if needed
    mask = np.sum(coords**2, axis=1) <= radius_sq * 1.0000001
    positions = coords[mask]
    return positions

def plot_basis_modes(
    modes: np.ndarray,
    positions: np.ndarray,
    count: int = 6,
    outfile: Union[Path, str, None] = None,
    cmap: str = "coolwarm",
    title_prefix: str = "Mode",
    interpolate: bool = False,
    resolution: int = 200
) -> None:
    """
    Plot the first `count` modes on the actuator grid.
    
    Args:
        modes: (n_actuators, n_modes) matrix.
        positions: (n_actuators, 2) matrix of coordinates.
        count: Number of modes to plot.
        outfile: If provided, save to this file.
        cmap: Colormap to use.
        title_prefix: Prefix for the title of each subplot.
        interpolate: If True, interpolate the modes onto a dense grid for visualization.
        resolution: Resolution of the interpolation grid (resolution x resolution).
    """
    n_act = positions.shape[0]
    if modes.shape[0] != n_act:
        raise ValueError(f"Mode dimension 0 ({modes.shape[0]}) does not match actuator count ({n_act})")

    cols = min(count, 4)
    rows = math.ceil(count / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes_arr = np.atleast_1d(axes).flatten()
    
    # Determine common scale
    absmax = np.max(np.abs(modes[:, :count])) if count > 0 else 1.0
    
    # Pre-calculate grid if interpolation is requested
    if interpolate:
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
        # Add a small buffer
        margin = (x_max - x_min) * 0.05
        grid_x, grid_y = np.mgrid[
            x_min-margin:x_max+margin:complex(0, resolution), 
            y_min-margin:y_max+margin:complex(0, resolution)
        ]
        # Calculate max radius for masking
        max_radius = np.max(np.sqrt(positions[:, 0]**2 + positions[:, 1]**2))

    for idx in range(len(axes_arr)):
        ax = axes_arr[idx]
        if idx < count and idx < modes.shape[1]:
            mode_data = modes[:, idx]
            
            if interpolate:
                grid_z = griddata(positions, mode_data, (grid_x, grid_y), method='cubic')
                # Mask outside the pupil (approximate based on max actuator radius)
                radius = np.sqrt(grid_x**2 + grid_y**2)
                grid_z[radius > max_radius * 1.05] = np.nan # 5% tolerance
                
                ax.imshow(
                    grid_z.T, 
                    extent=(x_min-margin, x_max+margin, y_min-margin, y_max+margin), 
                    origin='lower', 
                    cmap=cmap, 
                    vmin=-absmax, 
                    vmax=absmax
                )
            else:
                ax.scatter(
                    positions[:, 0], 
                    positions[:, 1], 
                    c=mode_data, 
                    cmap=cmap, 
                    vmin=-absmax, 
                    vmax=absmax,
                    s=20
                )
            
            ax.set_aspect('equal')
            ax.set_title(f"{title_prefix} {idx + 1}")
            ax.axis('off')
        else:
            ax.axis('off')
            
    plt.tight_layout()
    
    if outfile:
        plt.savefig(outfile, dpi=150)
        plt.close(fig)
    else:
        plt.show()

def make_concentric_actuator_grid(telescope_diameter: float, n_rings: int, n_points_innermost: int = 6) -> np.ndarray:
    """
    Generate actuator positions in concentric rings.
    
    Args:
        telescope_diameter: Diameter of the outermost ring.
        n_rings: Number of rings (excluding center).
        n_points_innermost: Number of points in the first ring. Subsequent rings have i * n_points_innermost points.
    """
    positions = [[0.0, 0.0]] # Central actuator
    
    radius_step = (telescope_diameter / 2) / n_rings
    
    for ring in range(1, n_rings + 1):
        radius = ring * radius_step
        n_points = n_points_innermost * ring
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        
        positions.extend(np.column_stack((x, y)))
        
    return np.array(positions)

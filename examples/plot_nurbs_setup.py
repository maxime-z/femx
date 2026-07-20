import numpy as np
import matplotlib.pyplot as plt
from femx.core.mesh import NurbsPatch
from femx.visualization.matplotlib_vis import plot_nurbs_geometry

def create_annular_patch() -> NurbsPatch:
    """Create a curved annular sector NURBS patch of degree p=2."""
    knots_u = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    knots_v = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    
    # 3x3 Control points coordinate grid representing a quarter circular arc block
    r_inner = 1.0
    r_outer = 2.0
    
    # Coordinates of control points:
    # Outer radius arc points, inner radius arc points, and intermediate points
    cp_x = np.array([
        [r_inner, 0.707 * r_inner, 0.0],
        [r_inner + 0.5, 0.707 * (r_inner + 0.5), 0.0],
        [r_outer, 0.707 * r_outer, 0.0]
    ])
    cp_y = np.array([
        [0.0, 0.707 * r_inner, r_inner],
        [0.0, 0.707 * (r_inner + 0.5), r_inner + 0.5],
        [0.0, 0.707 * r_outer, r_outer]
    ])
    control_points = np.stack([cp_x, cp_y], axis=-1)
    
    w = np.cos(np.pi / 4.0)
    weights = np.array([
        [1.0, w, 1.0],
        [1.0, w, 1.0],
        [1.0, w, 1.0]
    ])
    
    return NurbsPatch(
        p_u=2, p_v=2,
        knots_u=knots_u, knots_v=knots_v,
        control_points=control_points,
        weights=weights
    )

def main():
    print("--- Visualizing Annular NURBS Geometry & Control Grid ---")
    
    # 1. Create a curved NURBS patch
    patch = create_annular_patch()
    
    # 2. Plot geometry
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_nurbs_geometry(patch, show_control_grid=True, ax=ax)
    
    # Save image
    image_path = "examples/nurbs_setup.png"
    plt.savefig(image_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"NURBS geometry setup plot successfully saved to: {image_path}")

if __name__ == "__main__":
    main()

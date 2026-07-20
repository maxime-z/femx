# femx

`femx` is a modern Python finite element (FEM) and Isogeometric Analysis (IGA) framework. It provides modular, array-first abstractions for coupled multi-physics simulations, higher-order interpolation, and interactive post-processing.

## Features

- **Core FEM Infrastructure**:
  - `Mesh`, `FieldSpec`, `DofMap`, and state management.
  - Quadrature rules for 2D Bilinear Quads (Q1) and Linear Triangles (T1).
  - Vectorized COO matrix assembly for stiffness, mass, and load vectors.
- **Isogeometric Analysis (IGA)**:
  - B-spline Cox-de Boor evaluation (`femx.basis.nurbs`).
  - 2D NURBS physical mapping (`NurbsPatch`).
- **Physical Formulations & Materials**:
  - `HeatConductionFormulation` (linear heat equation).
  - `LinearElasticityFormulation` (Plane Strain, Plane Stress, 3D Hooke's Law).
  - Neumann boundary traction and flux integrators ($\int_{\Gamma_e} \mathbf{N}^T \mathbf{t} \, \mathrm{d}\Gamma$).
- **Post-Processing & Visualization**:
  - `femx.core.postprocessing`: Gauss-point strain, stress, and von Mises equivalent stress evaluation.
  - 2D filled contour plots and boundary condition quiver overlays using Matplotlib (`femx.visualization.matplotlib_vis`).
  - 3D interactive rendering and ParaView VTK (`.vtu`) export via PyVista (`femx.visualization.pyvista_vis`).

---

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/YOUR_USERNAME/femx.git
cd femx
pip install -e .
```

To include optional PyVista 3D visualization and VTK export tools:

```bash
pip install -e ".[vis]"
```

---

## Quick Start & Examples

### Running Unit Tests

Execute the test suite directly:

```bash
python run_tests.py
```

Or using `pytest`:

```bash
pytest
```

---

### Running Examples

1. **Cook's Membrane Benchmark (Plane Strain Elasticity)**:
   ```bash
   python -m examples.cooks_membrane
   ```

2. **2D Heat Conduction on Q1 FEM Mesh**:
   ```bash
   python -m examples.heat_q1
   ```

3. **2D Heat Conduction on Quadratic NURBS Patch (IGA)**:
   ```bash
   python -m examples.heat_nurbs
   ```

4. **Cantilever Beam Setup & BC Visualizer**:
   ```bash
   python -m examples.plot_mesh_setup
   ```

---

## License

MIT License

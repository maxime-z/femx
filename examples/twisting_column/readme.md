## How to run
```
python examples/twisting_column/run_elasticity.py --angle 15 --ny 4
python examples/twisting_column/run_neohookean.py --angle 180 --n-steps 4 --ny 2
python examples/twisting_column/run_incompressible.py --angle 45 --discretization nurbs --p 1
python examples/twisting_column/run_incompressible_torch.py --angle 30 --nx 1 --ny 4 --nz 1
python examples/twisting_column/bench_numpy_torch.py --nx 4 --ny 20 --nz 4
```

`run_incompressible_torch.py` times NumPy mixed / Neo-Hookean against a hybrid Newton that
evaluates Q1-hex Neo-Hookean residuals with PyTorch (CPU float64 by default; `--device mps`
for Apple GPU float32 residual timing). Mixed u-p still uses NumPy.
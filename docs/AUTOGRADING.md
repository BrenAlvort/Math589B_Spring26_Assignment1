# Autograding contract (Gradescope)

This assignment is graded automatically on Gradescope.

## What you implement

You must implement **both**:
1. Segment–segment WCA self-avoidance **energy + gradient** in `csrc/rod_energy.cpp`.
2. BFGS + line search in `src/elastic_rod/bfgs.py`.

You may refactor internally as you like, but the following **Python-level contract must hold**:

```python
from elastic_rod.model import RodEnergy
E, g = RodEnergy(...).value_and_grad(x)   # x shape (3N,)
```

## How Gradescope runs your submission

Gradescope will:
1. build your C++ shared library by running `bash csrc/build.sh`
2. run Python tests that import your package and call `RodEnergy.value_and_grad`
3. run a speed test (larger N) and a BFGS convergence test

## What we grade

### Correctness
- finite-difference gradient checks on random configurations (small N)
- sanity checks (finite values, no NaNs/Infs)

### Efficiency
- energy+grad runtime for larger N (you should not do cubic work)
- optimizer evaluation count (`n_feval`) and wall-clock time

### Optimization performance
- BFGS must produce a significant energy decrease on fixed benchmark instances.

## Reproducing autograder locally
After building the library, run:
```bash
pytest -q
python scripts/autograde_local.py --mode accuracy
python scripts/autograde_local.py --mode speed
```

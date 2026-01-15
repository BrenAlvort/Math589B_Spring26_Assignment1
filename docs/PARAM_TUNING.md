# Parameter tuning notes (for instructors)

With purely repulsive self-avoidance, the global minimum tends to be a round loop.
To obtain visually interesting packed/coiled states, include confinement `kc > 0`.

Recommended defaults (striking + stable once WCA is correct):
- N = 120
- kb = 1.0
- ks = 80 to 100
- l0 = 0.5
- kc = 0.015 to 0.03
- eps = 1.0
- sigma = 0.33 to 0.38

Qualitative regimes:
1) Loose loop (mild packing):
   kc=0.01, kb=1.2, ks=80, sigma=0.33

2) Coiled / "plectoneme-like" packing (more dramatic):
   kc=0.02, kb=0.8, ks=90, sigma=0.35

3) Very tight ball (harder optimization, more backtracking):
   kc=0.03, kb=0.6, ks=100, sigma=0.36

If students see numerical trouble:
- increase kb slightly
- decrease kc slightly
- ensure they clamp distances away from 0 (e.g. d=max(d,1e-12))

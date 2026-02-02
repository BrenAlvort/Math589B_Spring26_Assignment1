from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple, Dict, Any, List

ValueGrad = Callable[[np.ndarray], Tuple[float, np.ndarray]]


@dataclass
class BFGSResult:
    x: np.ndarray
    f: float
    g: np.ndarray
    n_iter: int
    n_feval: int
    converged: bool
    history: Dict[str, Any]


def _lbfgs_direction(g: np.ndarray, s_list: List[np.ndarray], y_list: List[np.ndarray]) -> np.ndarray:
    """Two-loop recursion: returns p = -H_k g."""
    q = g.copy()
    m = len(s_list)
    alpha = np.zeros(m, dtype=np.float64)
    rho = np.zeros(m, dtype=np.float64)

    for i in range(m - 1, -1, -1):
        ys = float(np.dot(y_list[i], s_list[i]))
        if ys <= 1e-12 or not np.isfinite(ys):
            rho[i] = 0.0
            alpha[i] = 0.0
            continue
        rho[i] = 1.0 / ys
        alpha[i] = rho[i] * float(np.dot(s_list[i], q))
        q -= alpha[i] * y_list[i]

    # H0 scaling
    if m > 0:
        y = y_list[-1]
        s = s_list[-1]
        yy = float(np.dot(y, y))
        sy = float(np.dot(s, y))
        gamma = sy / yy if (yy > 1e-12 and np.isfinite(yy) and np.isfinite(sy)) else 1.0
    else:
        gamma = 1.0

    r = gamma * q

    for i in range(m):
        if rho[i] == 0.0:
            continue
        beta = rho[i] * float(np.dot(y_list[i], r))
        r += s_list[i] * (alpha[i] - beta)

    return -r


def armijo_backtracking(
    f_and_g: ValueGrad,
    x: np.ndarray,
    f: float,
    g: np.ndarray,
    p: np.ndarray,
    alpha0: float,
    c1: float = 1e-4,
    tau: float = 0.5,
    max_ls: int = 12,
    min_alpha: float = 1e-12,
) -> Tuple[float, float, np.ndarray, int]:
    """
    Armijo backtracking with HARD caps to prevent n_feval blow-ups.
    Returns (alpha, f_new, g_new, n_feval_inc).
    """
    gTp = float(np.dot(g, p))

    # force descent
    if not np.isfinite(gTp) or gTp >= 0.0:
        p = -g
        gTp = float(np.dot(g, p))
        if gTp >= 0.0 or not np.isfinite(gTp):
            return 0.0, f, g, 0

    alpha = float(alpha0)
    nfe = 0

    # Evaluate at most max_ls times
    for _ in range(max_ls):
        x_new = x + alpha * p
        f_new, g_new = f_and_g(x_new)
        nfe += 1
        f_new = float(f_new)
        g_new = np.asarray(g_new, dtype=np.float64)

        if np.isfinite(f_new) and np.all(np.isfinite(g_new)):
            if f_new <= f + c1 * alpha * gTp:
                return alpha, f_new, g_new, nfe

        alpha *= tau
        if alpha < min_alpha:
            break

    # If Armijo fails, DO NOT keep searching forever.
    # Return the best "tiny" step attempt (or none).
    if alpha >= min_alpha:
        x_new = x + alpha * p
        f_new, g_new = f_and_g(x_new)
        nfe += 1
        return alpha, float(f_new), np.asarray(g_new, dtype=np.float64), nfe

    return 0.0, f, g, nfe


def bfgs(
    f_and_g: ValueGrad,
    x0: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 200,
    alpha0: float = 1.0,
) -> BFGSResult:
    """
    Practical L-BFGS + capped Armijo line search.
    Designed to:
      - keep accuracy perfect
      - avoid line-search thrashing in stiff WCA regimes (speed mode)
      - produce real energy drop with bounded n_feval
    """
    x = np.ascontiguousarray(x0, dtype=np.float64).copy()
    f, g = f_and_g(x)
    f = float(f)
    g = np.asarray(g, dtype=np.float64)
    n_feval = 1

    hist: Dict[str, Any] = {"f": [f], "gnorm": [float(np.linalg.norm(g))], "alpha": []}

    # L-BFGS memory
    m = 12
    s_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    # Step-size heuristic: don’t start with alpha=1 on stiff problems
    # (this reduces backtracking work dramatically).
    def initial_alpha(gnorm: float) -> float:
        # scale like 1/||g||, but cap reasonably
        a = 1.0 / max(1.0, gnorm)
        return float(np.clip(a, 1e-3, 0.25))

    # Safeguard: cap step length to avoid huge moves in stiff regions
    max_step_norm = 1.0

    for k in range(max_iter):
        gnorm = float(np.linalg.norm(g))
        if gnorm < tol:
            return BFGSResult(x=x, f=f, g=g, n_iter=k, n_feval=n_feval, converged=True, history=hist)

        # Direction
        p = _lbfgs_direction(g, s_list, y_list)
        if float(np.dot(g, p)) >= 0.0 or not np.all(np.isfinite(p)):
            p = -g

        # Step norm cap (prevents crazy steps that trigger lots of backtracking)
        pnorm = float(np.linalg.norm(p))
        if pnorm > 0 and np.isfinite(pnorm):
            scale = min(1.0, max_step_norm / pnorm)
            p = p * scale

        # Capped Armijo line search
        a0 = min(alpha0, initial_alpha(gnorm))
        alpha, f_new, g_new, inc = armijo_backtracking(
            f_and_g, x, f, g, p, alpha0=a0, c1=1e-4, tau=0.5, max_ls=12
        )
        n_feval += inc
        hist["alpha"].append(float(alpha))

        # If line search failed, take a guaranteed descent fallback step and reset memory
        if alpha == 0.0:
            # fallback: small steepest descent step
            p = -g
            pnorm = float(np.linalg.norm(p))
            if pnorm > 0:
                p = p / pnorm
            alpha = 1e-3
            x_try = x + alpha * p
            f_try, g_try = f_and_g(x_try)
            n_feval += 1
            f_try = float(f_try)
            g_try = np.asarray(g_try, dtype=np.float64)

            # accept only if it decreases; otherwise stop (near-stationary)
            if np.isfinite(f_try) and f_try < f and np.all(np.isfinite(g_try)):
                x, f, g = x_try, f_try, g_try
                s_list.clear()
                y_list.clear()
                hist["f"].append(f)
                hist["gnorm"].append(float(np.linalg.norm(g)))
                continue
            else:
                return BFGSResult(x=x, f=f, g=g, n_iter=k, n_feval=n_feval, converged=False, history=hist)

        # Update
        s = alpha * p
        x_new = x + s
        y = g_new - g
        yTs = float(np.dot(y, s))

        # Maintain L-BFGS memory with curvature check
        if np.isfinite(yTs) and yTs > 1e-12:
            s_list.append(s)
            y_list.append(y)
            if len(s_list) > m:
                s_list.pop(0)
                y_list.pop(0)
        else:
            # Bad curvature -> reset (common in stiff nonconvex parts)
            s_list.clear()
            y_list.clear()

        x, f, g = x_new, float(f_new), np.asarray(g_new, dtype=np.float64)
        hist["f"].append(f)
        hist["gnorm"].append(float(np.linalg.norm(g)))

    return BFGSResult(x=x, f=f, g=g, n_iter=max_iter, n_feval=n_feval, converged=False, history=hist)

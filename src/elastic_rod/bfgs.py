from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple, Dict, Any

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


def backtracking_line_search(
    f_and_g: ValueGrad,
    x: np.ndarray,
    f: float,
    g: np.ndarray,
    p: np.ndarray,
    alpha0: float = 1.0,
    c1: float = 1e-4,
    tau: float = 0.5,
    max_steps: int = 30,
) -> Tuple[float, float, np.ndarray, int]:
    """
    Armijo backtracking line search.

    Finds alpha = alpha0 * tau^m such that:
        f(x + alpha p) <= f(x) + c1 * alpha * g^T p

    Returns:
        (alpha, f_new, g_new, n_feval_increment)
    """
    # Ensure we have a descent direction; otherwise fall back to steepest descent.
    gTp = float(np.dot(g, p))
    if not np.isfinite(gTp) or gTp >= 0.0:
        p = -g
        gTp = float(np.dot(g, p))

    # If gradient is zero (or p is zero), step is irrelevant.
    if gTp == 0.0 or not np.isfinite(gTp):
        return 0.0, f, g, 0

    alpha = float(alpha0)
    n_feval_inc = 0

    # Armijo right-hand side is monotone in alpha, so backtracking works.
    for _ in range(max_steps):
        x_new = x + alpha * p
        f_new, g_new = f_and_g(x_new)
        n_feval_inc += 1

        if np.isfinite(f_new) and np.all(np.isfinite(g_new)):
            if f_new <= f + c1 * alpha * gTp:
                return alpha, float(f_new), np.asarray(g_new, dtype=np.float64), n_feval_inc

        alpha *= tau

        # If alpha becomes too small, give up (avoid underflow / no progress).
        if alpha < 1e-16:
            break

    # Failed to satisfy Armijo: return the best "do-nothing" step.
    return 0.0, f, g, n_feval_inc


def bfgs(
    f_and_g: ValueGrad,
    x0: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 200,
    alpha0: float = 1.0,
) -> BFGSResult:
    """
    Minimize f(x) with BFGS (inverse-Hessian form).

    Maintains H_k ≈ (∇^2 f(x_k))^{-1}, updates:
        p_k = -H_k g_k
        x_{k+1} = x_k + α_k p_k   (α_k from Armijo backtracking)
        s_k = x_{k+1} - x_k
        y_k = g_{k+1} - g_k

    Inverse-Hessian BFGS update:
        ρ = 1 / (y^T s)
        H_{k+1} = (I - ρ s y^T) H_k (I - ρ y s^T) + ρ s s^T

    with curvature check y^T s > 0 (otherwise skip/reset).
    """
    x = np.ascontiguousarray(x0, dtype=np.float64).copy()
    f, g = f_and_g(x)
    f = float(f)
    g = np.asarray(g, dtype=np.float64)
    n_feval = 1

    n = x.size
    H = np.eye(n, dtype=np.float64)

    hist: Dict[str, Any] = {
        "f": [f],
        "gnorm": [float(np.linalg.norm(g))],
        "alpha": [],
    }

    # A couple of practical safeguards
    min_curv = 1e-12  # curvature threshold for y^T s
    max_step_norm = 1e3  # prevents catastrophic steps if H gets wild

    for k in range(max_iter):
        gnorm = float(np.linalg.norm(g))
        if gnorm < tol:
            return BFGSResult(
                x=x, f=f, g=g,
                n_iter=k, n_feval=n_feval,
                converged=True, history=hist
            )

        # Search direction
        p = -H @ g

        # If not a descent direction (numerical issues), fall back to steepest descent.
        if float(np.dot(g, p)) >= 0.0 or not np.all(np.isfinite(p)):
            p = -g

        # Optional: cap step direction magnitude (stability)
        pnorm = float(np.linalg.norm(p))
        if pnorm > max_step_norm:
            p = p * (max_step_norm / pnorm)

        # Line search (Armijo)
        alpha, f_new, g_new, inc = backtracking_line_search(
            f_and_g, x, f, g, p,
            alpha0=alpha0, c1=1e-4, tau=0.5, max_steps=30
        )
        n_feval += inc
        hist["alpha"].append(float(alpha))

        # If line search failed (alpha==0), try a tiny steepest descent step once.
        if alpha == 0.0:
            p = -g
            alpha, f_new, g_new, inc2 = backtracking_line_search(
                f_and_g, x, f, g, p,
                alpha0=min(alpha0, 1.0), c1=1e-4, tau=0.5, max_steps=30
            )
            n_feval += inc2
            # If still no progress, stop.
            if alpha == 0.0:
                hist["f"].append(float(f))
                hist["gnorm"].append(float(np.linalg.norm(g)))
                return BFGSResult(
                    x=x, f=f, g=g,
                    n_iter=k + 1, n_feval=n_feval,
                    converged=False, history=hist
                )

        # Update state
        s = alpha * p
        x_new = x + s
        y = g_new - g

        # Update history
        x = x_new
        f = float(f_new)
        g = np.asarray(g_new, dtype=np.float64)
        hist["f"].append(f)
        hist["gnorm"].append(float(np.linalg.norm(g)))

        # BFGS update with curvature check
        yTs = float(np.dot(y, s))

        if np.isfinite(yTs) and yTs > min_curv:
            rho = 1.0 / yTs

            # (I - ρ s y^T)
            I = np.eye(n, dtype=np.float64)
            syT = np.outer(s, y)
            ysT = np.outer(y, s)
            ssT = np.outer(s, s)

            V = I - rho * syT
            H = V @ H @ (I - rho * ysT) + rho * ssT

            # Symmetrize to fight numerical drift
            H = 0.5 * (H + H.T)
        else:
            # Bad curvature: reset H (robust fallback)
            H = np.eye(n, dtype=np.float64)

    return BFGSResult(
        x=x, f=f, g=g,
        n_iter=max_iter, n_feval=n_feval,
        converged=False, history=hist
    )

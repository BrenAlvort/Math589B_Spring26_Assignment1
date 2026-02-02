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


def strong_wolfe_line_search(
    f_and_g: ValueGrad,
    x: np.ndarray,
    f: float,
    g: np.ndarray,
    p: np.ndarray,
    alpha0: float = 1.0,
    c1: float = 1e-4,
    c2: float = 0.9,
    max_iter: int = 25,
) -> Tuple[float, float, np.ndarray, int]:
    """
    Strong Wolfe line search:
      f(x+αp) <= f(x) + c1 α g^T p
      |∇f(x+αp)^T p| <= c2 |g^T p|
    Returns (alpha, f_new, g_new, n_feval_inc).
    """
    gTp0 = float(np.dot(g, p))
    if not np.isfinite(gTp0) or gTp0 >= 0.0:
        p = -g
        gTp0 = float(np.dot(g, p))
    if gTp0 == 0.0 or not np.isfinite(gTp0):
        return 0.0, f, g, 0

    nfe = 0
    alpha_prev = 0.0
    f_prev = f

    def phi(a: float):
        nonlocal nfe
        xa = x + a * p
        fa, ga = f_and_g(xa)
        nfe += 1
        return float(fa), np.asarray(ga, dtype=np.float64)

    def zoom(alo: float, ahi: float, flo: float):
        for _ in range(max_iter):
            aj = 0.5 * (alo + ahi)
            fj, gj = phi(aj)
            gTpj = float(np.dot(gj, p))

            if (fj > f + c1 * aj * gTp0) or (fj >= flo):
                ahi = aj
            else:
                if abs(gTpj) <= c2 * abs(gTp0):
                    return aj, fj, gj
                if gTpj * (ahi - alo) >= 0.0:
                    ahi = alo
                alo, flo = aj, fj

            if abs(ahi - alo) < 1e-16:
                break

        fj, gj = phi(alo)
        return alo, fj, gj

    alpha = float(alpha0)

    for it in range(max_iter):
        f_new, g_new = phi(alpha)

        if (f_new > f + c1 * alpha * gTp0) or (it > 0 and f_new >= f_prev):
            a, fn, gn = zoom(alpha_prev, alpha, f_prev)
            return a, fn, gn, nfe

        gTp = float(np.dot(g_new, p))
        if abs(gTp) <= c2 * abs(gTp0):
            return alpha, f_new, g_new, nfe

        if gTp >= 0.0:
            a, fn, gn = zoom(alpha, alpha_prev, f_new)
            return a, fn, gn, nfe

        alpha_prev = alpha
        f_prev = f_new
        alpha *= 2.0
        if alpha > 1e6:
            break

    # If line search fails, return a conservative tiny step rather than alpha=0.
    # This prevents "drop=0" early exits.
    a = 1e-6
    fa, ga = f_and_g(x + a * p)
    return a, float(fa), np.asarray(ga, dtype=np.float64), nfe + 1


def _lbfgs_direction(g: np.ndarray, s_list: List[np.ndarray], y_list: List[np.ndarray]) -> np.ndarray:
    """
    Two-loop recursion to compute p = -H_k g without forming H explicitly.
    """
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

    # Scaling for initial Hessian H0: gamma = (s_{k-1}^T y_{k-1})/(y_{k-1}^T y_{k-1})
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


def bfgs(
    f_and_g: ValueGrad,
    x0: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 200,
    alpha0: float = 1.0,
) -> BFGSResult:
    """
    L-BFGS (memory m=10) + Strong Wolfe line search.
    Uses same signature/return type as the assignment.
    """
    x = np.ascontiguousarray(x0, dtype=np.float64).copy()
    f, g = f_and_g(x)
    f = float(f)
    g = np.asarray(g, dtype=np.float64)
    n_feval = 1

    hist: Dict[str, Any] = {"f": [f], "gnorm": [float(np.linalg.norm(g))], "alpha": []}

    # L-BFGS memory
    m = 10
    s_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    for k in range(max_iter):
        gnorm = float(np.linalg.norm(g))
        if gnorm < tol:
            return BFGSResult(x=x, f=f, g=g, n_iter=k, n_feval=n_feval, converged=True, history=hist)

        # Search direction via two-loop recursion
        p = _lbfgs_direction(g, s_list, y_list)

        # Ensure descent
        if float(np.dot(g, p)) >= 0.0 or not np.all(np.isfinite(p)):
            p = -g

        alpha, f_new, g_new, inc = strong_wolfe_line_search(
            f_and_g, x, f, g, p, alpha0=alpha0, c1=1e-4, c2=0.9, max_iter=25
        )
        n_feval += inc
        hist["alpha"].append(float(alpha))

        s = alpha * p
        x_new = x + s
        y = np.asarray(g_new, dtype=np.float64) - g

        # Update memory if curvature is good
        ys = float(np.dot(y, s))
        if np.isfinite(ys) and ys > 1e-12:
            s_list.append(s)
            y_list.append(y)
            if len(s_list) > m:
                s_list.pop(0)
                y_list.pop(0)

        x = x_new
        f = float(f_new)
        g = np.asarray(g_new, dtype=np.float64)

        hist["f"].append(f)
        hist["gnorm"].append(float(np.linalg.norm(g)))

    return BFGSResult(x=x, f=f, g=g, n_iter=max_iter, n_feval=n_feval, converged=False, history=hist)

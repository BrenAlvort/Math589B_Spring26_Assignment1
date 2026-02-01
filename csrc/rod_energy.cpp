#include <cmath>
#include <algorithm>

extern "C" {

int rod_api_version() { return 2; }

static inline int imod(int i, int N) {
    int r = i % N;
    return (r < 0) ? (r + N) : r;
}

// Robust closest points between segments P0P1 and Q0Q1 in R^3.
// Returns (s,t) in [0,1]x[0,1] such that
//   C1 = P0 + s*(P1-P0),  C2 = Q0 + t*(Q1-Q0)
// are closest points.
// Based on standard routine (Ericson, Real-Time Collision Detection).
static inline void closest_params_segment_segment(
    const double P0[3], const double P1[3],
    const double Q0[3], const double Q1[3],
    double &s, double &t
) {
    const double EPS = 1e-12;

    double d1[3] = { P1[0]-P0[0], P1[1]-P0[1], P1[2]-P0[2] };
    double d2[3] = { Q1[0]-Q0[0], Q1[1]-Q0[1], Q1[2]-Q0[2] };
    double r[3]  = { P0[0]-Q0[0], P0[1]-Q0[1], P0[2]-Q0[2] };

    auto dot3 = [](const double a[3], const double b[3]) {
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
    };

    double a = dot3(d1,d1);  // |d1|^2
    double e = dot3(d2,d2);  // |d2|^2
    double f = dot3(d2,r);

    // Handle degenerate segments
    if (a <= EPS && e <= EPS) { s = 0.0; t = 0.0; return; }
    if (a <= EPS) { // P is a point
        s = 0.0;
        t = (e > EPS) ? (f / e) : 0.0;
        t = std::clamp(t, 0.0, 1.0);
        return;
    }
    double c = dot3(d1,r);
    if (e <= EPS) { // Q is a point
        t = 0.0;
        s = -c / a;
        s = std::clamp(s, 0.0, 1.0);
        return;
    }

    double b = dot3(d1,d2);
    double denom = a*e - b*b;

    double sN, sD = denom;
    double tN, tD = denom;

    if (denom < EPS) {
        // Almost parallel: choose s = 0 and clamp t from there
        sN = 0.0; sD = 1.0;
        tN = f;   tD = e;
    } else {
        sN = (b*f - c*e);
        tN = (a*f - b*c);
    }

    // Clamp s to [0,1], update t accordingly
    if (sN < 0.0) {
        sN = 0.0;
        tN = f;
        tD = e;
    } else if (sN > sD) {
        sN = sD;
        tN = f + b;
        tD = e;
    }

    // Clamp t to [0,1], update s accordingly
    if (tN < 0.0) {
        tN = 0.0;
        sN = -c;
        sD = a;
        sN = std::clamp(sN, 0.0, sD);
    } else if (tN > tD) {
        tN = tD;
        sN = b - c;
        sD = a;
        sN = std::clamp(sN, 0.0, sD);
    }

    s = (std::abs(sD) > EPS) ? (sN / sD) : 0.0;
    t = (std::abs(tD) > EPS) ? (tN / tD) : 0.0;

    s = std::clamp(s, 0.0, 1.0);
    t = std::clamp(t, 0.0, 1.0);
}

void rod_energy_grad(
    int N,
    const double* x,
    double kb,
    double ks,
    double l0,
    double kc,
    double eps,
    double sigma,
    double* energy_out,
    double* grad_out
) {
    const int M = 3*N;
    for (int i = 0; i < M; ++i) grad_out[i] = 0.0;
    double E = 0.0;

    auto idx = [N](int i) {
        int r = i % N;
        return (r < 0) ? (r + N) : r;
    };
    auto get = [&](int i, int d) -> double {
        return x[3*idx(i) + d];
    };
    auto addg = [&](int i, int d, double v) {
        grad_out[3*idx(i) + d] += v;
    };

    // ---- Bending: kb * sum ||x_{i+1} - 2 x_i + x_{i-1}||^2
    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < 3; ++d) {
            const double b = get(i+1,d) - 2.0*get(i,d) + get(i-1,d);
            E += kb * b * b;
            const double c = 2.0 * kb * b;
            addg(i-1, d, c);
            addg(i,   d, -2.0*c);
            addg(i+1, d, c);
        }
    }

    // ---- Stretching: ks * sum (||x_{i+1}-x_i|| - l0)^2
    for (int i = 0; i < N; ++i) {
        double dx0 = get(i+1,0) - get(i,0);
        double dx1 = get(i+1,1) - get(i,1);
        double dx2 = get(i+1,2) - get(i,2);
        double r = std::sqrt(dx0*dx0 + dx1*dx1 + dx2*dx2);
        r = std::max(r, 1e-12);
        double diff = r - l0;
        E += ks * diff * diff;

        double coeff = 2.0 * ks * diff / r;
        addg(i+1,0,  coeff * dx0);
        addg(i+1,1,  coeff * dx1);
        addg(i+1,2,  coeff * dx2);
        addg(i,0,   -coeff * dx0);
        addg(i,1,   -coeff * dx1);
        addg(i,2,   -coeff * dx2);
    }

    // ---- Confinement: kc * sum ||x_i||^2
    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < 3; ++d) {
            double xi = get(i,d);
            E += kc * xi * xi;
            addg(i,d, 2.0 * kc * xi);
        }
    }

    // ---- Segment–segment WCA self-avoidance ----
    // Segment i is (i,i+1). Exclude adjacent segments (share an endpoint),
    // including wrap neighbors.
    if (eps != 0.0 && sigma > 0.0) {
        const double rc = std::pow(2.0, 1.0/6.0) * sigma;
        const double rc2 = rc * rc;
        const double sig = sigma;
        const double EPSD = 1e-12;

        for (int i = 0; i < N; ++i) {
            int i0 = i;
            int i1 = idx(i+1);

            double P0[3] = { get(i0,0), get(i0,1), get(i0,2) };
            double P1[3] = { get(i0+1,0), get(i0+1,1), get(i0+1,2) }; // periodic via get

            for (int j = i+1; j < N; ++j) {
                // Exclusions: skip if segments share a node:
                // (i,i+1) adjacent to (i-1,i), (i,i+1), (i+1,i+2)
                // In terms of segment index j: j == i, i-1, i+1 (mod N)
                int jm = idx(j);
                if (jm == idx(i) || jm == idx(i-1) || jm == idx(i+1)) continue;

                // Also exclude wrap-adjacent pair: segment 0 and segment N-1
                // This is already handled by the modular adjacency above, but keep safe.
                // (If i=0, then j=N-1 is adjacent.)
                if (idx(i)==0 && idx(j)==N-1) continue;
                if (idx(j)==0 && idx(i)==N-1) continue;

                double Q0[3] = { get(j,0), get(j,1), get(j,2) };
                double Q1[3] = { get(j+1,0), get(j+1,1), get(j+1,2) };

                double u, v;
                closest_params_segment_segment(P0, P1, Q0, Q1, u, v);

                // Closest points
                double Ci[3] = {
                    P0[0] + u*(P1[0]-P0[0]),
                    P0[1] + u*(P1[1]-P0[1]),
                    P0[2] + u*(P1[2]-P0[2]),
                };
                double Cj[3] = {
                    Q0[0] + v*(Q1[0]-Q0[0]),
                    Q0[1] + v*(Q1[1]-Q0[1]),
                    Q0[2] + v*(Q1[2]-Q0[2]),
                };

                double rvec[3] = { Ci[0]-Cj[0], Ci[1]-Cj[1], Ci[2]-Cj[2] };
                double d2 = rvec[0]*rvec[0] + rvec[1]*rvec[1] + rvec[2]*rvec[2];

                if (d2 >= rc2) continue;

                double d = std::sqrt(std::max(d2, EPSD));

                // WCA energy
                double s = sig / d;
                double s2 = s*s;
                double s6 = s2*s2*s2;
                double s12 = s6*s6;
                double U = 4.0*eps*(s12 - s6) + eps;
                E += U;

                // dU/dd = (24 eps / d) * (-2 s12 + s6)
                double dU_dd = (24.0 * eps / d) * (-2.0*s12 + s6);

                // ∂U/∂Ci = dU/dd * ∂d/∂Ci = dU/dd * (rvec/d)
                // force vector (gradient contribution on Ci)
                double scale = dU_dd / d;
                double gCi[3] = { scale*rvec[0], scale*rvec[1], scale*rvec[2] };
                // ∂U/∂Cj = -gCi
                double gCj[3] = { -gCi[0], -gCi[1], -gCi[2] };

                // Distribute to endpoints:
                // Ci = (1-u) P0 + u P1
                // Cj = (1-v) Q0 + v Q1
                const double wP0 = (1.0 - u);
                const double wP1 = u;
                const double wQ0 = (1.0 - v);
                const double wQ1 = v;

                for (int dcomp = 0; dcomp < 3; ++dcomp) {
                    addg(i0,     dcomp, wP0 * gCi[dcomp]);
                    addg(i0+1,   dcomp, wP1 * gCi[dcomp]);
                    addg(j,      dcomp, wQ0 * gCj[dcomp]);
                    addg(j+1,    dcomp, wQ1 * gCj[dcomp]);
                }
            }
        }
    }

    *energy_out = E;
}

} // extern "C"

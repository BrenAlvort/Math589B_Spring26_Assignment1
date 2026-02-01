#include <cmath>
#include <algorithm>

extern "C" {

// Bump when you change the exported function signatures.
int rod_api_version() { return 2; }

static inline int imod(int i, int N) {
    int r = i % N;
    return (r < 0) ? (r + N) : r;
}

static inline double dot3(const double a[3], const double b[3]) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

static inline void sub3(double out[3], const double a[3], const double b[3]) {
    out[0] = a[0]-b[0];
    out[1] = a[1]-b[1];
    out[2] = a[2]-b[2];
}

static inline double norm3(const double a[3]) {
    return std::sqrt(dot3(a,a));
}

// Robust closest points parameters between segments P0P1 and Q0Q1.
// Returns s,t in [0,1]x[0,1].
// Standard routine (Ericson-style).
static inline void closest_params_segment_segment(
    const double P0[3], const double P1[3],
    const double Q0[3], const double Q1[3],
    double &s, double &t
) {
    const double EPS = 1e-12;

    double d1[3] = { P1[0]-P0[0], P1[1]-P0[1], P1[2]-P0[2] };
    double d2[3] = { Q1[0]-Q0[0], Q1[1]-Q0[1], Q1[2]-Q0[2] };
    double r[3]  = { P0[0]-Q0[0], P0[1]-Q0[1], P0[2]-Q0[2] };

    double a = dot3(d1,d1);
    double e = dot3(d2,d2);
    double f = dot3(d2,r);

    if (a <= EPS && e <= EPS) { s = 0.0; t = 0.0; return; } // both points
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
        // nearly parallel
        sN = 0.0; sD = 1.0;
        tN = f;   tD = e;
    } else {
        sN = (b*f - c*e);
        tN = (a*f - b*c);
    }

    // clamp s
    if (sN < 0.0) {
        sN = 0.0;
        tN = f;
        tD = e;
    } else if (sN > sD) {
        sN = sD;
        tN = f + b;
        tD = e;
    }

    // clamp t
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

static inline void lerp3(double out[3], const double A[3], const

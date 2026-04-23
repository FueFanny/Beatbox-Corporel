"""
3D Trilateration Solver
=======================
Find the coordinates of an unknown point P in 3D space given:
  - A world/reference origin W0
  - N anchor points (coordinates relative to W0), where N >= 4
  - N measured distances from P to each anchor

Method
------
Subtracting the first distance equation from the remaining N-1 equations
cancels the quadratic |P|^2 term and yields a (N-1)x3 linear system.

  - N == 4  : exact system  (3x3) → solved with np.linalg.solve
  - N  > 4  : overdetermined (Nx3) → solved with np.linalg.lstsq
                               (least-squares best fit, handles noise)

Requirements
------------
    pip install numpy

Usage
-----
    python trilateration.py
    or import and call trilaterate() directly in your own script.
"""

import numpy as np


# =============================================================================
# SOLVER FUNCTION
# =============================================================================

def trilaterate(W0, anchors, distances):
    """
    Compute the 3D coordinates of unknown point P.

    Parameters
    ----------
    W0        : array-like (3,)       World/reference origin (X0, Y0, Z0)
    anchors   : list of array-like    Anchor coordinates relative to W0,
                                      each entry is (Xi, Yi, Zi).
                                      Minimum 4 anchors required.
    distances : list of float         Measured distances from P to each
                                      anchor. Must match len(anchors).

    Returns
    -------
    P_world   : ndarray (3,)          Coordinates of P in the world frame
    residuals : ndarray (N,)          |recomputed_dist - given_dist| per
                                      anchor (close to 0 = good solution)

    Raises
    ------
    ValueError
        If fewer than 4 anchors are provided.
        If anchors and distances have different lengths.
        If the anchor configuration is degenerate (coplanar / collinear).
    """

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    if len(anchors) != len(distances):
        raise ValueError(
            f"Number of anchors ({len(anchors)}) must match "
            f"number of distances ({len(distances)})."
        )

    if len(anchors) < 4:
        raise ValueError(
            f"At least 4 anchors are required. Got {len(anchors)}."
        )

    # Convert all inputs to numpy float arrays
    W0      = np.asarray(W0, dtype=float)
    anchors = [np.asarray(O, dtype=float) for O in anchors]
    dists   = [float(D) for D in distances]

    O1_a = anchors[0]
    D1_a = dists[0]

    N = len(anchors)

    # ------------------------------------------------------------------
    # Build the (N-1) x 3 linear system
    # ------------------------------------------------------------------
    # Each distance equation:
    #   |P - Oi|^2 = Di^2
    #   |P|^2 - 2*P.Oi + |Oi|^2 = Di^2
    #
    # Subtract equation 1 from equations 2..N to cancel |P|^2:
    #   2*(Oi - O1).P = D1^2 - Di^2 + |Oi|^2 - |O1|^2
    #
    # This gives A*P = b  (N-1 equations, 3 unknowns x, y, z)
    # ------------------------------------------------------------------

    A = np.zeros((N - 1, 3))
    b = np.zeros(N - 1)

    for i, (Oi, Di) in enumerate(zip(anchors[1:], dists[1:])):
        A[i] = 2.0 * (Oi - O1_a)
        b[i] = (D1_a**2 - Di**2
                + np.dot(Oi, Oi) - np.dot(O1_a, O1_a))

    # ------------------------------------------------------------------
    # Solve the linear system
    #   N == 4  →  exact 3x3 system   (np.linalg.solve)
    #   N  > 4  →  overdetermined     (np.linalg.lstsq, least-squares)
    # ------------------------------------------------------------------

    if N == 4:
        # Exact solve
        try:
            P_local = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            raise ValueError(
                "Anchor configuration is degenerate (singular matrix).\n"
                "Make sure the 4 anchors are not all coplanar or collinear."
            )
    else:
        # Least-squares solve (best fit when N > 4)
        result  = np.linalg.lstsq(A, b, rcond=None)
        P_local = result[0]
        rank    = result[2]
        if rank < 3:
            raise ValueError(
                "Anchor configuration is degenerate (rank deficient).\n"
                "Make sure the anchors are not all coplanar or collinear."
            )

    # Convert result from W0-relative frame to world frame
    P_world = P_local + W0

    # ------------------------------------------------------------------
    # Verification: recompute distances and compare to given values
    # ------------------------------------------------------------------

    residuals = np.array([
        abs(np.linalg.norm(P_local - Oi) - Di)
        for Oi, Di in zip(anchors, dists)
    ])

    return P_world, residuals


# =============================================================================
# EXAMPLE
# =============================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # INPUT — set your values here
    # Add or remove anchors and distances as needed (minimum 4)
    # ------------------------------------------------------------------

    W0 = [0.0, 0.0, 0.0]           # World/reference origin (X0, Y0, Z0)

    anchors = [
        [0.0, 0.0, 0.0],            # O1 coordinate relative to W0
        [0.0, -150.0, 65.0],            # O2 coordinate relative to W0
        [150.0, 100.0, 0.0],            # O3 coordinate relative to W0
        [250.0, -100.0, 0.0],            # O4 coordinate relative to W0
        [380.0, -100.0, 100.0],            # O5 coordinate relative to W0
        [180.0, -200.0, 200.0],            # O6 coordinate relative to W0
    ]
# Real D=[ 216.1, 216.16, 179.44, 119.16, 196.72,186.28]
    distances = [
        210,                   # D1 — measured distance from P to O1
        216,                   # D2 — measured distance from P to O2
        182,                   # D3 — measured distance from P to O3
        115,                   # D4 — measured distance from P to O4
        196,                   # D5 — measured distance from P to O5
        184,                   # D6 — measured distance from P to O6
    ]

    # ------------------------------------------------------------------
    # SOLVE
    # ------------------------------------------------------------------

    P, residuals = trilaterate(W0, anchors, distances)

    # ------------------------------------------------------------------
    # OUTPUT
    # ------------------------------------------------------------------

    print("=" * 50)
    print("  3D Trilateration Solver")
    print("=" * 50)

    print(f"\n  Inputs ({len(anchors)} anchors):")
    print(f"    World origin W0 = {W0}")
    for i, (O, D) in enumerate(zip(anchors, distances), 1):
        print(f"    Anchor O{i} = {O},  D{i} = {D}")

    print(f"\n  Result:")
    print(f"    P = ({P[0]:.6f},  {P[1]:.6f},  {P[2]:.6f})")

    print(f"\n  Verification (residuals should be ~0):")
    for i, r in enumerate(residuals, 1):
        status = "ok" if r < 1e-4 else "FAIL"
        print(f"    [{status}]  |recomputed D{i} - given D{i}| = {r:.2e}")

    print("=" * 50)

"""
3D Trilateration Solver
=======================
Given 4 anchor points O1..O4 (relative to world origin W0) and distances
D1..D4 from unknown point P, compute P's coordinates relative to W0.

Method
------
Subtracting the first distance equation from the remaining three cancels
the quadratic |P|^2 term and yields three linear equations in (x, y, z).
The 3x3 system is solved with NumPy.

All anchor coordinates and the returned P are expressed relative to W0.
"""

import numpy as np


def trilaterate(W0, O1, O2, O3, O4, D1, D2, D3, D4):
    """
    Find point P given 4 anchors and their distances to P.

    Parameters
    ----------
    W0          : array-like (3,)  world/reference origin  (X0, Y0, Z0)
    O1..O4      : array-like (3,)  anchor coords relative to W0
    D1..D4      : float            distances |P - Oi|

    Returns
    -------
    P_world     : ndarray (3,)     coords of P in world frame (relative to W0)
    residuals   : ndarray (4,)     |recomputed_dist - given_dist| per anchor
    """
    W0      = np.asarray(W0, dtype=float)
    anchors = [np.asarray(O, dtype=float) for O in (O1, O2, O3, O4)]
    dists   = [float(D) for D in (D1, D2, D3, D4)]

    O1_a, D1_a = anchors[0], dists[0]

    # Build 3x3 linear system by subtracting eq_1 from eq_i (i = 2, 3, 4)
    # Row i:  2(Oi - O1) · p  =  Di² - D1² - |Oi|² + |O1|²
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for i, (Oi, Di) in enumerate(zip(anchors[1:], dists[1:])):
        A[i] = 2.0 * (Oi - O1_a)
        b[i] = (D1_a**2 - Di**2
                + np.dot(Oi, Oi) - np.dot(O1_a, O1_a))

    # Solve for P in the W0-relative frame
    try:
        P_local = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        raise ValueError(
            "Anchor configuration is degenerate (singular matrix).\n"
            "Ensure no three anchors are collinear and not all four are coplanar."
        )

    # P in the world frame
    P_world = P_local + W0

    # Verification residuals (computed in the local/W0-relative frame)
    residuals = np.array([
        abs(np.linalg.norm(P_local - Oi) - Di)
        for Oi, Di in zip(anchors, dists)
    ])

    return P_world, residuals


# ── Example ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # World origin
    W0 = [1.0, 2.0, 3.0]

    # Anchor coords relative to W0
    O1 = [0.0, 0.0, 0.0]
    O2 = [8.0, 0.0, 0.0]
    O3 = [4.0, 6.0, 0.0]
    O4 = [4.0, 2.0, 6.0]

    # True P in the W0-relative frame = [3, 4, 2]
    # → world frame = W0 + [3, 4, 2] = [4, 6, 5]
    P_local_true = np.array([3.0, 4.0, 2.0])
    D1 = np.linalg.norm(P_local_true - np.asarray(O1))
    D2 = np.linalg.norm(P_local_true - np.asarray(O2))
    D3 = np.linalg.norm(P_local_true - np.asarray(O3))
    D4 = np.linalg.norm(P_local_true - np.asarray(O4))

    print("=" * 55)
    print("  3D Trilateration Solver")
    print("=" * 55)
    print(f"  World origin  W0 = {W0}")
    print()
    print(f"  Anchor O1 = {O1}   D1 = {D1:.6f}")
    print(f"  Anchor O2 = {O2}   D2 = {D2:.6f}")
    print(f"  Anchor O3 = {O3}   D3 = {D3:.6f}")
    print(f"  Anchor O4 = {O4}   D4 = {D4:.6f}")
    print()

    P, residuals = trilaterate(W0, O1, O2, O3, O4, D1, D2, D3, D4)

    P_world_expected = np.array(W0) + P_local_true
    print(f"  Solved  P (world frame) = {np.round(P, 8)}")
    print(f"  Expected P (world frame)= {P_world_expected}")
    print()
    print("  Verification — distance residuals (should be ~0):")
    for i, r in enumerate(residuals, 1):
        status = "ok" if r < 1e-6 else "FAIL"
        print(f"    [{status}]  |recomputed D{i} - given D{i}| = {r:.2e}")
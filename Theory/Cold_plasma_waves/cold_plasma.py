"""Cold magnetized plasma waves: Stix parameters, dispersion roots, CMA diagram.

Plasma: electrons + one singly charged ion species (charge +e), cold, uniform,
B0 = B0 z_hat.  Everything is written in the dimensionless CMA variables

    X  = omega_pe^2 / omega^2          (density axis)
    Y  = Omega_e / omega = |omega_ce| / omega   (magnetic-field axis)
    mu = m_i / m_e                     (ion-to-electron mass ratio)

so that omega_pi^2/omega^2 = X/mu and Omega_i/omega = Y/mu.

Stix (1962) parameters, signed cyclotron frequencies (Bellan 2006, Eq. 6.12):

    R = 1 - sum_s omega_ps^2 / (omega (omega + omega_cs))
    L = 1 - sum_s omega_ps^2 / (omega (omega - omega_cs))
    S = (R + L)/2,  D = (R - L)/2,  P = 1 - sum_s omega_ps^2 / omega^2

With omega_ce = -Omega_e and omega_ci = +Omega_i this becomes

    R = 1 - X/(1 - Y) - X/(mu + Y)
    L = 1 - X/(1 + Y) - X/(mu - Y)
    P = 1 - X - X/mu

Dispersion relation (Bellan Eq. 6.47-6.49), n = k c / omega, theta = angle(k, B0):

    A n^4 - B n^2 + C = 0
    A = S sin^2(theta) + P cos^2(theta)
    B = R L sin^2(theta) + P S (1 + cos^2(theta))
    C = P R L
"""

from __future__ import annotations

import numpy as np

MU_PROTON = 1836.15267


# ----------------------------------------------------------------------------
# Stix parameters
# ----------------------------------------------------------------------------
def stix(X, Y, mu=MU_PROTON):
    """Return dict with S, D, P, R, L for CMA coordinates (X, Y) and mass ratio mu."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = 1.0 - X / (1.0 - Y) - X / (mu + Y)
        L = 1.0 - X / (1.0 + Y) - X / (mu - Y)
    P = 1.0 - X - X / mu
    S = 0.5 * (R + L)
    D = 0.5 * (R - L)
    return dict(S=S, D=D, P=P, R=R, L=L)


def stix_from_frequencies(omega, wpe, Oce, mu=MU_PROTON):
    """Same as :func:`stix` but from dimensional (angular) frequencies."""
    omega = np.asarray(omega, dtype=float)
    return stix(wpe**2 / omega**2, Oce / omega, mu)


# ----------------------------------------------------------------------------
# Dispersion roots
# ----------------------------------------------------------------------------
def abc(theta, S, P, R, L):
    """Coefficients A, B, C of A n^4 - B n^2 + C = 0."""
    s2, c2 = np.sin(theta) ** 2, np.cos(theta) ** 2
    A = S * s2 + P * c2
    B = R * L * s2 + P * S * (1.0 + c2)
    C = P * R * L
    return A, B, C


def n2_roots(theta, S, P, R, L):
    """Both roots n^2 (plus, minus) of the cold-plasma dispersion relation.

    n2_plus = (B + sqrt(F)) / (2A),  n2_minus = (B - sqrt(F)) / (2A),  F = B^2 - 4AC >= 0.
    Each root is evaluated with whichever algebraically equivalent form avoids
    cancellation (2C / (B -/+ sqrt F)), so the A -> 0 resonance is handled:
    the plus root -> +-inf there while the minus root stays finite.
    """
    A, B, C = abc(theta, S, P, R, L)
    F = np.clip(B * B - 4 * A * C, 0.0, None)  # >= 0 analytically (Bellan Eq. 6.50)
    sq = np.sqrt(F)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = -0.5 * (-B - np.where(B >= 0, 1.0, -1.0) * sq)  # Numerical Recipes form
        # q = 0.5 (B + sign(B) sq); roots in n^2: q/A and C/q
        r1 = q / A
        r2 = C / q
    # sort into the "+" root (B + sq)/(2A) and "-" root (B - sq)/(2A)
    pos_B = B >= 0
    n2p = np.where(pos_B, r1, r2)
    n2m = np.where(pos_B, r2, r1)
    return n2p, n2m


def phase_velocity(theta, S, P, R, L):
    """v_ph/c = 1/n for both roots; NaN where the root is evanescent (n^2 <= 0)."""
    out = []
    for n2 in n2_roots(theta, S, P, R, L):
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.where(n2 > 0, 1.0 / np.sqrt(np.where(n2 > 0, n2, 1.0)), np.nan)
        v = np.where(np.isinf(n2), 0.0, v)  # resonance: v_ph -> 0
        out.append(v)
    return out


def label_modes(S, P, R, L, eps=1e-6):
    """Name the two roots by their limits at theta = 0 (R/L) and theta = 90 deg (O/X).

    At theta = 0:   n^2 in {R, L}         (plus a P = 0 electrostatic oscillation)
    At theta = 90:  n^2 in {P, RL/S}      (O mode: E || B0;  X mode: E perp B0)
    Returns two strings for (plus, minus) like 'R / X' or 'L / O'; '' if that
    root is evanescent at that angle.
    """
    labels = []
    with np.errstate(divide="ignore", invalid="ignore"):
        X_mode = R * L / S
    for which in (0, 1):
        n2_0 = n2_roots(eps, S, P, R, L)[which]
        n2_90 = n2_roots(np.pi / 2 - eps, S, P, R, L)[which]
        lab0 = ""
        if np.isfinite(n2_0) and n2_0 > 0:
            lab0 = "R" if abs(n2_0 - R) <= abs(n2_0 - L) else "L"
        lab90 = ""
        if np.isfinite(n2_90) and n2_90 > 0:
            lab90 = "O" if abs(n2_90 - P) <= abs(n2_90 - X_mode) else "X"
        labels.append((lab0, lab90))
    return labels


def wave_normal_type(theta, vph):
    """Classify a v_ph(theta) curve as 'ellipsoid', 'dumbbell', 'wheel' or 'none'."""
    ok = np.isfinite(vph) & (vph > 0)
    if not ok.any():
        return "none"
    # use the samples closest to 0 and 90 deg (not a +-1 deg window), so a resonance
    # cone within a degree of 90 deg (shear Alfven / ICW in the Alfven region) still counts
    th = np.mod(theta, np.pi)
    d0 = np.minimum(th, np.pi - th)
    d90 = np.abs(th - np.pi / 2)
    near0 = ok[d0 <= d0.min() + 1e-12].any()
    near90 = ok[d90 <= d90.min() + 1e-12].any()
    if near0 and near90 and ok.mean() > 0.98:
        return "ellipsoid"
    if near0 and not near90:
        return "dumbbell"
    if near90 and not near0:
        return "wheel"
    return "partial"


# ----------------------------------------------------------------------------
# Characteristic frequencies (all in units of omega_pe)
# ----------------------------------------------------------------------------
def characteristic_frequencies(Oce_over_wpe, mu=MU_PROTON):
    """Cyclotron, cutoff and hybrid frequencies normalised to omega_pe.

    Cutoffs: R = 0 -> omega_R, L = 0 -> omega_L, P = 0 -> omega_p (~omega_pe).
    Hybrids: S = 0 -> omega_UH (upper), omega_LH (lower).
    Alfven speed v_A/c = Omega_i/omega_pi (non-relativistic, cold).
    """
    Oe = Oce_over_wpe
    Oi = Oe / mu
    wpi2 = 1.0 / mu
    wp = np.sqrt(1.0 + wpi2)
    # S(omega) = 0 is a quadratic in omega^2:
    # 1 - wpe^2/(w^2 - Oe^2) - wpi^2/(w^2 - Oi^2) = 0
    a = 1.0
    b = -(Oe**2 + Oi**2 + 1.0 + wpi2)
    c = Oe**2 * Oi**2 + Oi**2 + wpi2 * Oe**2
    disc = np.sqrt(b * b - 4 * a * c)
    wUH = np.sqrt((-b + disc) / 2)
    wLH = np.sqrt((-b - disc) / 2)

    # R = 0 and L = 0 are quadratics in omega:
    # R = 0: w^2 - (Oe - Oi) w - (wp^2 + Oe Oi) = 0   (exact for one ion species)
    def pos_root(bb, cc):
        return 0.5 * (-bb + np.sqrt(bb * bb - 4 * cc))

    wR = pos_root(-(Oe - Oi), -(wp**2 + Oe * Oi))
    wL = pos_root(+(Oe - Oi), -(wp**2 + Oe * Oi))
    vA = Oi / np.sqrt(wpi2)
    return dict(Oe=Oe, Oi=Oi, wp=wp, wR=wR, wL=wL, wUH=wUH, wLH=wLH, vA=vA)


# ----------------------------------------------------------------------------
# omega-k branches for theta = 0 and 90 deg
# ----------------------------------------------------------------------------
def omega_k_branches(Oce_over_wpe, mu, theta, w=None):
    """k c / omega_pe versus omega / omega_pe for each propagating branch.

    theta = 0:   n^2 = R and n^2 = L     (returns keys 'R', 'L')
    theta = 90:  n^2 = P and n^2 = RL/S  (returns keys 'O', 'X')
    Each value is a list of contiguous (k, w) segments where n^2 > 0.
    """
    if w is None:
        w = np.geomspace(1e-4, 5, 40000)
    p = stix(1.0 / w**2, Oce_over_wpe / w, mu)
    if np.isclose(theta, 0):
        n2s = dict(R=p["R"], L=p["L"])
    elif np.isclose(theta, np.pi / 2):
        with np.errstate(divide="ignore", invalid="ignore"):
            n2s = dict(O=p["P"], X=p["R"] * p["L"] / p["S"])
    else:
        raise ValueError("only theta = 0 or pi/2 are implemented here")
    out = {}
    for name, n2 in n2s.items():
        ok = np.isfinite(n2) & (n2 > 0)
        k = np.where(ok, w * np.sqrt(np.where(ok, n2, 0)), np.nan)
        segs, start = [], None
        for i, flag in enumerate(np.append(ok, False)):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                if i - start > 2:
                    segs.append((k[start:i], w[start:i]))
                start = None
        out[name] = segs
    return out


# ----------------------------------------------------------------------------
# CMA diagram boundaries: X as a function of Y on each bounding surface
# ----------------------------------------------------------------------------
def cma_boundaries(Y, mu):
    """Return {name: X(Y)} for the cutoffs R=0, L=0, P=0 and resonance S=0.

    The resonances R = inf (Y = 1) and L = inf (Y = mu) are horizontal lines.
    Non-physical (X <= 0) parts are NaN.
    """
    Y = np.asarray(Y, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        XR = (1 - Y) * (mu + Y) / (mu + 1)
        XL = (1 + Y) * (mu - Y) / (mu + 1)
        XS = 1.0 / (1.0 / (1 - Y**2) + mu / (mu**2 - Y**2))
    XP = np.full_like(Y, mu / (mu + 1))
    clean = lambda a: np.where(np.isfinite(a) & (a > 0), a, np.nan)
    return {"R=0": clean(XR), "L=0": clean(XL), "S=0": clean(XS), "P=0": XP}


def region_signature(X, Y, mu):
    """Integer code from the signs of P, R, L, S and the positions of the two resonances.

    Points with the same code lie in the same CMA 'bounded volume'.
    """
    p = stix(X, Y, mu)
    bits = [p["P"] > 0, p["R"] > 0, p["L"] > 0, p["S"] > 0, Y > 1, Y > mu]
    code = np.zeros(np.shape(X), dtype=int)
    for i, b in enumerate(bits):
        code |= (np.asarray(b, dtype=int) << i)
    return code

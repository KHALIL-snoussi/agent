"""
Low-level color science utilities shared across the QBRIX pipeline.

Provides:
    - sRGB ↔ Lab conversion helpers (D65 white point, IEC 61966-2-1 profile)
    - Vectorised CIEDE2000 implementation for robust palette distance checks

All functions accept NumPy arrays so callers can operate on entire grids, yet they
also work with plain Python iterables for single color conversions.
"""

from __future__ import annotations

import numpy as np

SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)

XYZ_TO_SRGB = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ],
    dtype=np.float64,
)

D65_WHITE = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
DELTA = 6 / 29


def _to_ndarray(color) -> np.ndarray:
    arr = np.asarray(color, dtype=np.float64)
    if arr.shape[-1] != 3:
        raise ValueError("Input color must have three channels")
    return arr


def srgb_to_linear(rgb) -> np.ndarray:
    """Convert sRGB (0-255 or 0-1 range) to linear RGB."""
    rgb = _to_ndarray(rgb)
    if rgb.max() > 1.0 + 1e-6:
        rgb = rgb / 255.0
    mask = rgb <= 0.04045
    return np.where(mask, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(linear_rgb) -> np.ndarray:
    """Convert linear RGB (0-1) back to sRGB in 0-255 range."""
    linear_rgb = _to_ndarray(linear_rgb)
    mask = linear_rgb <= 0.0031308
    srgb = np.where(
        mask,
        12.92 * linear_rgb,
        1.055 * np.power(np.clip(linear_rgb, 0.0, 1.0), 1 / 2.4) - 0.055,
    )
    return np.clip(srgb * 255.0, 0.0, 255.0)


def rgb_to_xyz(rgb) -> np.ndarray:
    """Convert sRGB to CIE XYZ (D65)."""
    linear = srgb_to_linear(rgb)
    flat = linear.reshape(-1, 3)
    xyz = flat @ SRGB_TO_XYZ.T
    return xyz.reshape(linear.shape)


def xyz_to_lab(xyz) -> np.ndarray:
    """Convert XYZ to Lab (D65)."""
    xyz = _to_ndarray(xyz) / D65_WHITE

    def f(t):
        return np.where(t > DELTA**3, np.cbrt(t), t / (3 * DELTA**2) + 4 / 29)

    fx = f(xyz[..., 0])
    fy = f(xyz[..., 1])
    fz = f(xyz[..., 2])

    L = (116 * fy) - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def lab_to_xyz(lab) -> np.ndarray:
    """Convert Lab (D65) back to XYZ."""
    lab = _to_ndarray(lab)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    fy = (L + 16) / 116
    fx = fy + (a / 500)
    fz = fy - (b / 200)

    def finv(t):
        return np.where(
            t > DELTA, t**3, (t - 4 / 29) * (3 * DELTA**2)
        )

    x = finv(fx)
    y = finv(fy)
    z = finv(fz)
    xyz = np.stack([x, y, z], axis=-1)
    return xyz * D65_WHITE


def rgb_to_lab(rgb) -> np.ndarray:
    """Convenience helper for sRGB → Lab."""
    return xyz_to_lab(rgb_to_xyz(rgb))


def lab_to_rgb(lab) -> np.ndarray:
    """Lab → sRGB convenience helper."""
    xyz = lab_to_xyz(lab)
    linear = xyz.reshape(-1, 3) @ XYZ_TO_SRGB.T
    linear = linear.reshape(xyz.shape)
    return np.clip(linear_to_srgb(linear), 0, 255).astype(np.uint8)


def delta_e2000(lab1, lab2) -> np.ndarray:
    """
    CIEDE2000 color difference with numpy broadcasting.

    lab1 and lab2 may be:
        - matching shapes (...,3)
        - lab1 shape (...,3) and lab2 shape (3,) (broadcast)
    Returns an array with the broadcasted leading dimensions.
    """

    lab1 = _to_ndarray(lab1)
    lab2 = _to_ndarray(lab2)

    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    C_mean = (C1 + C2) / 2

    G = 0.5 * (1 - np.sqrt((C_mean**7) / (C_mean**7 + 25**7 + 1e-12)))
    a1_prime = (1 + G) * a1
    a2_prime = (1 + G) * a2
    C1_prime = np.sqrt(a1_prime**2 + b1**2)
    C2_prime = np.sqrt(a2_prime**2 + b2**2)
    C_mean_prime = (C1_prime + C2_prime) / 2

    def h_func(a_component, b_component):
        angle = np.degrees(np.arctan2(b_component, a_component))
        angle = np.where(angle < 0, angle + 360, angle)
        return angle

    h1_prime = h_func(a1_prime, b1)
    h2_prime = h_func(a2_prime, b2)

    delta_h_prime = h2_prime - h1_prime
    delta_h_prime = np.where(
        np.abs(delta_h_prime) <= 180,
        delta_h_prime,
        delta_h_prime - 360 * np.sign(delta_h_prime),
    )

    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime
    delta_H_prime = 2 * np.sqrt(C1_prime * C2_prime) * np.sin(
        np.radians(delta_h_prime / 2)
    )

    L_mean = (L1 + L2) / 2
    H_sum = h1_prime + h2_prime
    H_mean_prime = np.where(
        np.abs(h1_prime - h2_prime) > 180,
        (H_sum + 360) / 2,
        H_sum / 2,
    )
    H_mean_prime = np.where(
        (C1_prime * C2_prime) == 0, H_sum, H_mean_prime
    )

    T = (
        1
        - 0.17 * np.cos(np.radians(H_mean_prime - 30))
        + 0.24 * np.cos(np.radians(2 * H_mean_prime))
        + 0.32 * np.cos(np.radians(3 * H_mean_prime + 6))
        - 0.20 * np.cos(np.radians(4 * H_mean_prime - 63))
    )

    delta_theta = 30 * np.exp(-(((H_mean_prime - 275) / 25) ** 2))
    R_C = 2 * np.sqrt((C_mean_prime**7) / (C_mean_prime**7 + 25**7 + 1e-12))
    R_T = -np.sin(np.radians(2 * delta_theta)) * R_C

    S_L = 1 + ((0.015 * (L_mean - 50) ** 2) / np.sqrt(20 + (L_mean - 50) ** 2))
    S_C = 1 + 0.045 * C_mean_prime
    S_H = 1 + 0.015 * C_mean_prime * T

    delta_E = np.sqrt(
        (delta_L_prime / S_L) ** 2
        + (delta_C_prime / S_C) ** 2
        + (delta_H_prime / S_H) ** 2
        + R_T * (delta_C_prime / S_C) * (delta_H_prime / S_H)
    )

    return delta_E

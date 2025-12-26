#!/usr/bin/env python3
"""
Geometry utilities for homographies, projection, and Dobot coordinate conversion.
"""

import numpy as np
from .config import DOBOT_CORNERS

def homography_from_4pt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Compute a projective transform H mapping src points to dst points via DLT."""
    A = []
    for (x, y), (X, Y) in zip(src, dst):
        A.append([-x, -y, -1, 0, 0, 0, x * X, y * X, X])
        A.append([0, 0, 0, -x, -y, -1, x * Y, y * Y, Y])
    A = np.asarray(A, dtype=float)
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1]
    H = h.reshape(3, 3)
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H

def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a homography to a set of 2D points."""
    n = pts.shape[0]
    homo = np.hstack([pts, np.ones((n, 1))])
    mapped = (H @ homo.T).T
    w = mapped[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return mapped[:, :2] / w

def project_points(H: np.ndarray, pts):
    """Project points using H with homogeneous normalization."""
    pts = np.asarray(pts, dtype=np.float32)
    ones = np.ones((len(pts), 1), dtype=np.float32)
    P = np.hstack([pts, ones]) @ H.T
    return P[:, :2] / P[:, 2:3]

def convert_pixels_to_dobot(pixel_pts: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Convert pixel coordinates in the camera image to Dobot coordinates using a 4-corner homography."""
    px_corners = np.array([
        [0.0, 0.0],
        [float(img_w - 1), 0.0],
        [float(img_w - 1), float(img_h - 1)],
        [0.0, float(img_h - 1)],
    ], dtype=float)
    dobot_corners = np.asarray(DOBOT_CORNERS, dtype=float)
    H = homography_from_4pt(px_corners, dobot_corners)
    return apply_homography(H, pixel_pts)

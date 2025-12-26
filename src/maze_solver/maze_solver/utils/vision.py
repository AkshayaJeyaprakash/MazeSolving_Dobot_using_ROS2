#!/usr/bin/env python3
"""
Low-level computer vision primitives for maze detection and warping.
"""

import time
import numpy as np
import cv2
from .config import OUT_SIZE, MARGIN_PX

def _order_quad_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    """Order a quadrilateral's points as top-left, top-right, bottom-right, bottom-left."""
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def _quad_score_rectangularity(quad: np.ndarray) -> float:
    """Score how rectangular a quad is using corner angle cosines."""
    def angle_cos(p0, p1, p2):
        v1 = p0 - p1
        v2 = p2 - p1
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        return abs(np.dot(v1, v2) / denom)
    cos_sum = 0.0
    for i in range(4):
        cos_sum += angle_cos(quad[(i - 1) % 4], quad[i], quad[(i + 1) % 4])
    return 1.0 / (1.0 + cos_sum)

def detect_maze_quad(gray: np.ndarray):
    """Detect the maze board quadrilateral and return ordered corners and a quality score."""
    H, W = gray.shape[:2]
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    thr = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 7)
    thr = cv2.medianBlur(thr, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)
    edges = cv2.Canny(thr, 60, 180)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0.0
    best = (None, 0.0)
    img_area = float(H * W)
    for c in cnts:
        if cv2.contourArea(c) < 0.02 * img_area:
            continue
        eps = 0.02 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        quad = approx.reshape(-1, 2).astype(np.float32)
        rect_score = _quad_score_rectangularity(quad)
        area_score = min(1.0, cv2.contourArea(approx) / img_area / 0.6)
        x, y, w, h = cv2.boundingRect(approx)
        aspect = max(w, h) / max(1, min(w, h))
        asp_score = 1.0 if 0.7 <= aspect <= 1.4 else max(0.0, 1.4 / aspect - 0.5)
        score = 0.55 * rect_score + 0.35 * area_score + 0.10 * asp_score
        if score > best[1]:
            best = (quad, score)
    if best[0] is None or best[1] < 0.35:
        return None, 0.0
    return _order_quad_tl_tr_br_bl(best[0]), float(best[1])

def warp_square(img, src_pts, out_size: int = OUT_SIZE, inner_margin: int = MARGIN_PX):
    """Warp a quadrilateral to a square output with optional inner margin scaling."""
    dst = np.array([
        [0, 0],
        [out_size - 1, 0],
        [out_size - 1, out_size - 1],
        [0, out_size - 1]
    ], dtype=np.float32)
    H0 = cv2.getPerspectiveTransform(src_pts, dst)
    warped = cv2.warpPerspective(img, H0, (out_size, out_size), flags=cv2.INTER_LINEAR)
    if inner_margin > 0:
        m = inner_margin
        warped = warped[m:out_size - m, m:out_size - m]
        warped = cv2.resize(warped, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
        s = out_size / float(out_size - 2 * m)
        A = np.array([[s, 0, -s * m],
                      [0, s, -s * m],
                      [0, 0, 1]], dtype=np.float32)
        H_total = A @ H0
    else:
        H_total = H0
    return warped, H_total

class StabilityLock:
    """
    Temporal lock that measures how long a detected quadrilateral remains within a pixel tolerance.

    The lock returns accumulated stable time when the quad remains within tolerance, and resets
    if drift exceeds tolerance or the quad disappears.
    """

    def __init__(self, hold_sec: float = 3.0, px_tol: float = 8.0):
        """Initialize the stability lock with hold-time and pixel tolerance."""
        self.hold_sec = hold_sec
        self.px_tol = px_tol
        self._last = None
        self._since = None

    def update(self, quad):
        """Update the lock with the current quad and return the stable time in seconds."""
        now = time.time()
        if quad is None:
            self._last = None
            self._since = None
            return 0.0
        if self._last is None:
            self._last = quad
            self._since = now
            return 0.0
        d = np.linalg.norm((quad - self._last), axis=1).mean()
        if d < self.px_tol:
            return now - (self._since or now)
        else:
            self._last = quad
            self._since = now
            return 0.0

    def reset(self):
        """Reset the stability measurement and last quad."""
        self._last = None
        self._since = None

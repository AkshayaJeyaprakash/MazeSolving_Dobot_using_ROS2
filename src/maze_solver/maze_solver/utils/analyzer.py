#!/usr/bin/env python3
"""
High-level analyzer that builds a grid graph from the warped maze and finds a shortest path.
"""

import numpy as np
import cv2
from heapq import heappush, heappop
from .geometry import project_points
from .io_utils import save_debug_image
from .config import OUT_SIZE

def _snap_extend(lines, length, snap_tol, extend_tol):
    """Extend and snap detected grid lines to image boundaries and deduplicate near-equals."""
    if not lines:
        return []
    lines = sorted(int(x) for x in lines)
    if lines[0] < snap_tol:
        lines[0] = 0
    if (length - 1) - lines[-1] < snap_tol:
        lines[-1] = length - 1
    dedup = []
    for v in lines:
        if not dedup or abs(v - dedup[-1]) > 1:
            dedup.append(v)
    if dedup[0] > extend_tol:
        dedup = [0] + dedup
    if (length - 1) - dedup[-1] > extend_tol:
        dedup = dedup + [length - 1]
    return dedup

class LiveMazeAnalyzer:
    """
    Analyzer that extracts grid lines, determines start and end cells, builds a cell graph,
    solves via A*, verifies collisions, and returns artifacts for visualization.
    """

    def __init__(self):
        """Initialize internal state."""
        self.last_analysis = None

    def _cluster_lines(self, positions, min_distance=10):
        """Cluster near-by line indices into representative centers."""
        if not positions:
            return []
        positions = sorted(positions)
        clusters, cur = [], [positions[0]]
        for p in positions[1:]:
            if p - cur[-1] < min_distance:
                cur.append(p)
            else:
                clusters.append(int(np.mean(cur)))
                cur = [p]
        clusters.append(int(np.mean(cur)))
        return clusters

    def detect_grid_size(self, binary_image):
        """Estimate grid rows and columns and return debug overlays."""
        h, w = binary_image.shape
        den = cv2.medianBlur(binary_image, 3)
        pad = 4
        den_pad = cv2.copyMakeBorder(den, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        kx = max(5, (w // 22) | 1)
        ky = max(5, (h // 22) | 1)
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 3))
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, ky))
        horiz_bands = cv2.morphologyEx(den_pad, cv2.MORPH_OPEN, horiz_kernel)[pad:-pad, pad:-pad]
        vert_bands = cv2.morphologyEx(den_pad, cv2.MORPH_OPEN, vert_kernel)[pad:-pad, pad:-pad]
        hp = np.sum(horiz_bands > 0, axis=1).astype(np.float32)
        vp = np.sum(vert_bands > 0, axis=0).astype(np.float32)
        hp = cv2.blur(hp.reshape(-1, 1), (21, 1)).ravel()
        vp = cv2.blur(vp.reshape(1, -1), (1, 21)).ravel()
        h_thr = max(np.percentile(hp, 70) * 0.9, 0.2 * (hp.max() if hp.max() > 0 else 1.0))
        v_thr = max(np.percentile(vp, 70) * 0.9, 0.2 * (vp.max() if vp.max() > 0 else 1.0))
        h_idx = np.where(hp >= h_thr)[0].tolist()
        v_idx = np.where(vp >= v_thr)[0].tolist()
        h_lines = self._cluster_lines(h_idx, min_distance=max(2, h // 18))
        v_lines = self._cluster_lines(v_idx, min_distance=max(2, w // 18))
        snap_tol_h = max(2, h // 60)
        snap_tol_v = max(2, w // 60)
        est_h_gap = np.median(np.diff(sorted(h_lines))) if len(h_lines) >= 2 else max(1, h // 5)
        est_v_gap = np.median(np.diff(sorted(v_lines))) if len(v_lines) >= 2 else max(1, w // 5)
        extend_tol_h = int(0.60 * est_h_gap)
        extend_tol_v = int(0.60 * est_v_gap)
        h_lines = _snap_extend(h_lines, h, snap_tol_h, extend_tol_h)
        v_lines = _snap_extend(v_lines, w, snap_tol_v, extend_tol_v)
        if len(h_lines) - 1 <= 4 and len(h_lines) >= 2:
            gaps = np.diff(sorted(h_lines))
            med = np.median(gaps) if len(gaps) else max(1, h // 5)
            ins = []
            for i, g in enumerate(gaps):
                if g > 1.75 * med:
                    ins.append(int(h_lines[i] + round(g / 2)))
            if ins:
                h_lines = sorted(h_lines + ins)
        if len(v_lines) - 1 <= 4 and len(v_lines) >= 2:
            gaps = np.diff(sorted(v_lines))
            med = np.median(gaps) if len(gaps) else max(1, w // 5)
            ins = []
            for i, g in enumerate(gaps):
                if g > 1.75 * med:
                    ins.append(int(v_lines[i] + round(g / 2)))
            if ins:
                v_lines = sorted(v_lines + ins)
        rows = len(h_lines) - 1 if len(h_lines) > 1 else 0
        cols = len(v_lines) - 1 if len(v_lines) > 1 else 0
        rows = max(3, min(10, rows))
        cols = max(3, min(10, cols))
        grid_debug = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        for yy in h_lines:
            cv2.line(grid_debug, (0, yy), (w - 1, yy), (0, 255, 0), 2)
        for xx in v_lines:
            cv2.line(grid_debug, (xx, 0), (xx, h - 1), (255, 0, 0), 2)
        edges_full = cv2.Canny(binary_image, 50, 150, apertureSize=3)
        return (rows, cols, grid_debug, edges_full)

    def extract_grid_lines(self, binary_image):
        """Return grid line indices along each axis for graph construction."""
        h, w = binary_image.shape
        den = cv2.medianBlur(binary_image, 3)
        pad = 4
        den_pad = cv2.copyMakeBorder(den, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        kx = max(5, (w // 22) | 1)
        ky = max(5, (h // 22) | 1)
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 3))
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, ky))
        hb = cv2.morphologyEx(den_pad, cv2.MORPH_OPEN, horiz_kernel)[pad:-pad, pad:-pad]
        vb = cv2.morphologyEx(den_pad, cv2.MORPH_OPEN, vert_kernel)[pad:-pad, pad:-pad]
        hp = cv2.blur(np.sum(hb > 0, axis=1).astype(np.float32).reshape(-1, 1), (21, 1)).ravel()
        vp = cv2.blur(np.sum(vb > 0, axis=0).astype(np.float32).reshape(1, -1), (1, 21)).ravel()
        h_thr = max(np.percentile(hp, 70) * 0.9, 0.2 * (hp.max() if hp.max() > 0 else 1.0))
        v_thr = max(np.percentile(vp, 70) * 0.9, 0.2 * (vp.max() if vp.max() > 0 else 1.0))
        h_idx = np.where(hp >= h_thr)[0].tolist()
        v_idx = np.where(vp >= v_thr)[0].tolist()
        h_lines = self._cluster_lines(h_idx, min_distance=max(2, h // 18))
        v_lines = self._cluster_lines(v_idx, min_distance=max(2, w // 18))
        snap_tol_h = max(2, h // 60)
        snap_tol_v = max(2, w // 60)
        est_h_gap = np.median(np.diff(sorted(h_lines))) if len(h_lines) >= 2 else max(1, h // 5)
        est_v_gap = np.median(np.diff(sorted(v_lines))) if len(v_lines) >= 2 else max(1, w // 5)
        extend_tol_h = int(0.60 * est_h_gap)
        extend_tol_v = int(0.60 * est_v_gap)
        h_lines = _snap_extend(h_lines, h, snap_tol_h, extend_tol_h)
        v_lines = _snap_extend(v_lines, w, snap_tol_v, extend_tol_v)
        if len(h_lines) - 1 <= 4 and len(h_lines) >= 2:
            gaps = np.diff(sorted(h_lines))
            med = np.median(gaps) if len(gaps) else max(1, h // 5)
            ins = []
            for i, g in enumerate(gaps):
                if g > 1.75 * med:
                    ins.append(int(h_lines[i] + round(g / 2)))
            if ins:
                h_lines = sorted(h_lines + ins)
        if len(v_lines) - 1 <= 4 and len(v_lines) >= 2:
            gaps = np.diff(sorted(v_lines))
            med = np.median(gaps) if len(gaps) else max(1, w // 5)
            ins = []
            for i, g in enumerate(gaps):
                if g > 1.75 * med:
                    ins.append(int(v_lines[i] + round(g / 2)))
            if ins:
                v_lines = sorted(v_lines + ins)
        return h_lines, v_lines

    def detect_entrance_exit(self, image_bgr):
        """Locate green and red dots and return their centers."""
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
        red1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
        red2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red1, red2)
        def _center(mask):
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                M = cv2.moments(c)
                if M['m00'] != 0:
                    return (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
            return None
        return _center(green_mask), _center(red_mask)

    def _carve_dots_from_binary(self, warped_bgr, binary):
        """Remove colored dot areas from a binary maze to avoid false walls."""
        hsv = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2HSV)
        g = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
        r1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
        r2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
        dots = cv2.bitwise_or(g, cv2.bitwise_or(r1, r2))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dots = cv2.dilate(dots, kernel, iterations=1)
        out = binary.copy()
        out[dots > 0] = 0
        return out

    def _clean_binary_for_graph(self, binary):
        """Denoise and close small holes to yield a stable graph extraction."""
        b = cv2.medianBlur(binary, 3)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k, iterations=1)
        return b

    def _cell_of_point(self, pt, h_lines, v_lines):
        """Map a pixel point to a grid cell index."""
        if pt is None:
            return None
        x, y = pt
        def idx(val, lines):
            val = max(lines[0] + 1, min(val, lines[-1] - 2))
            j = int(np.searchsorted(lines, val, side="right")) - 1
            return max(0, min(j, len(lines) - 2))
        r = idx(y, h_lines)
        c = idx(x, v_lines)
        return (r, c)

    def _cell_center(self, r, c, h_lines, v_lines):
        """Return the pixel center of a grid cell."""
        cx = (v_lines[c] + v_lines[c + 1]) // 2
        cy = (h_lines[r] + h_lines[r + 1]) // 2
        return (cx, cy)

    def _neighbors_open(self, binary, r, c, h_lines, v_lines):
        """Evaluate whether up, down, left, right neighbors are reachable."""
        H, W = binary.shape
        cx = (v_lines[c] + v_lines[c + 1]) // 2
        cy = (h_lines[r] + h_lines[r + 1]) // 2
        stripe_w = max(1, (v_lines[c + 1] - v_lines[c]) // 20)
        stripe_h = max(1, (h_lines[r + 1] - h_lines[r]) // 20)
        band_t = 3
        def is_free(mean_val):
            return mean_val < 40.0
        opens = {}
        if r > 0:
            yb = h_lines[r]
            xs = slice(max(0, cx - stripe_w), min(W, cx + stripe_w + 1))
            ys_a = slice(max(0, yb - band_t - 1), max(0, yb - 1))
            ys_m = slice(max(0, yb - band_t // 2), min(H, yb + band_t // 2 + 1))
            ys_b = slice(min(H, yb + 1), min(H, yb + band_t + 1))
            opens['U'] = all([is_free(np.mean(binary[ys_a, xs])),
                              is_free(np.mean(binary[ys_m, xs])),
                              is_free(np.mean(binary[ys_b, xs]))])
        if r < len(h_lines) - 2:
            yb = h_lines[r + 1]
            xs = slice(max(0, cx - stripe_w), min(W, cx + stripe_w + 1))
            ys_a = slice(max(0, yb - band_t - 1), max(0, yb - 1))
            ys_m = slice(max(0, yb - band_t // 2), min(H, yb + band_t // 2 + 1))
            ys_b = slice(min(H, yb + 1), min(H, yb + band_t + 1))
            opens['D'] = all([is_free(np.mean(binary[ys_a, xs])),
                              is_free(np.mean(binary[ys_m, xs])),
                              is_free(np.mean(binary[ys_b, xs]))])
        if c > 0:
            xb = v_lines[c]
            ys = slice(max(0, cy - stripe_h), min(H, cy + stripe_h + 1))
            xs_a = slice(max(0, xb - band_t - 1), max(0, xb - 1))
            xs_m = slice(max(0, xb - band_t // 2), min(W, xb + band_t // 2 + 1))
            xs_b = slice(min(W, xb + 1), min(W, xb + band_t + 1))
            opens['L'] = all([is_free(np.mean(binary[ys, xs_a])),
                              is_free(np.mean(binary[ys, xs_m])),
                              is_free(np.mean(binary[ys, xs_b]))])
        if c < len(v_lines) - 2:
            xb = v_lines[c + 1]
            ys = slice(max(0, cy - stripe_h), min(H, cy + stripe_h + 1))
            xs_a = slice(max(0, xb - band_t - 1), max(0, xb - 1))
            xs_m = slice(max(0, xb - band_t // 2), min(W, xb + band_t // 2 + 1))
            xs_b = slice(min(W, xb + 1), min(W, xb + band_t + 1))
            opens['R'] = all([is_free(np.mean(binary[ys, xs_a])),
                              is_free(np.mean(binary[ys, xs_m])),
                              is_free(np.mean(binary[ys, xs_b]))])
        return opens

    def _build_graph(self, binary, h_lines, v_lines):
        """Construct an adjacency list graph from the grid openings."""
        rows = len(h_lines) - 1
        cols = len(v_lines) - 1
        graph = {(r, c): [] for r in range(rows) for c in range(cols)}
        for r in range(rows):
            for c in range(cols):
                op = self._neighbors_open(binary, r, c, h_lines, v_lines)
                if op.get('U'):
                    graph[(r, c)].append((r - 1, c))
                if op.get('D'):
                    graph[(r, c)].append((r + 1, c))
                if op.get('L'):
                    graph[(r, c)].append((r, c - 1))
                if op.get('R'):
                    graph[(r, c)].append((r, c + 1))
        return graph

    def _astar_shortest_path(self, graph, start, goal):
        """Find a shortest path using A* on grid-like coordinates."""
        if start is None or goal is None:
            return None
        def heuristic(a, b):
            try:
                (x1, y1) = a
                (x2, y2) = b
                return abs(x1 - x2) + abs(y1 - y2)
            except Exception:
                return 0
        open_set = []
        heappush(open_set, (0, start))
        parent = {start: None}
        g_score = {start: 0}
        while open_set:
            _, u = heappop(open_set)
            if u == goal:
                path = []
                cur = u
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
            for v in graph.get(u, []):
                tentative_g = g_score[u] + 1
                if v not in g_score or tentative_g < g_score[v]:
                    g_score[v] = tentative_g
                    parent[v] = u
                    f_score = tentative_g + heuristic(v, goal)
                    heappush(open_set, (f_score, v))
        return None

    def _verify_solution_warped(self, binary_inv_255, path_pts_warped):
        """Verify that the path does not collide with walls by sampling along segments."""
        if not path_pts_warped or len(path_pts_warped) < 2:
            return {"ok": False, "reason": "no_path"}
        collisions = 0
        samples = 0
        for (x0, y0), (x1, y1) in zip(path_pts_warped[:-1], path_pts_warped[1:]):
            N = max(8, int(np.hypot(x1 - x0, y1 - y0) // 3))
            xs = np.linspace(x0, x1, N).astype(int)
            ys = np.linspace(y0, y1, N).astype(int)
            ys = np.clip(ys, 0, binary_inv_255.shape[0] - 1)
            xs = np.clip(xs, 0, binary_inv_255.shape[1] - 1)
            vals = binary_inv_255[ys, xs]
            samples += N
            collisions += int((vals > 120).sum())
        coll_ratio = collisions / max(1, samples)
        return {"ok": coll_ratio < 0.05, "coll_ratio": float(coll_ratio), "samples": int(samples)}

    def analyze_snapshot(self, warped_frame, H_total, original_frame, start_color: str, debug_dir=None):
        """
        Analyze a warped maze image to produce a path and artifacts for visualization.
        """
        gray = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 31, 5)
        if debug_dir:
            save_debug_image(debug_dir, "01_binary_initial.jpg", binary)
        green_center, red_center = self.detect_entrance_exit(warped_frame)
        if debug_dir:
            dot_detection_img = warped_frame.copy()
            if green_center:
                cv2.circle(dot_detection_img, green_center, 20, (0, 255, 0), 3)
                cv2.putText(dot_detection_img, "GREEN", (green_center[0] - 30, green_center[1] - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if red_center:
                cv2.circle(dot_detection_img, red_center, 20, (0, 0, 255), 3)
                cv2.putText(dot_detection_img, "RED", (red_center[0] - 20, red_center[1] - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            save_debug_image(debug_dir, "02_dot_detection.jpg", dot_detection_img)
        binary = self._carve_dots_from_binary(warped_frame, binary)
        binary = self._clean_binary_for_graph(binary)
        if debug_dir:
            save_debug_image(debug_dir, "03_binary_dots_carved.jpg", binary)
        rows, cols, grid_debug, edges_full = self.detect_grid_size(binary)
        if debug_dir:
            save_debug_image(debug_dir, "04_grid_detection.jpg", grid_debug)
            save_debug_image(debug_dir, "05_edges.jpg", edges_full)
        sc = (start_color or 'green').strip().lower()
        if sc.startswith('r'):
            entrance, exit_point = red_center, green_center
            chosen = 'red'
        else:
            entrance, exit_point = green_center, red_center
            chosen = 'green'
        result_image = warped_frame.copy()
        if entrance:
            cv2.circle(result_image, entrance, 15, (0, 255, 0), 3)
            cv2.putText(result_image, "START", (entrance[0] - 30, entrance[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if exit_point:
            cv2.circle(result_image, exit_point, 15, (0, 0, 255), 3)
            cv2.putText(result_image, "END", (exit_point[0] - 20, exit_point[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(result_image, f"Grid: {rows}x{cols}  Start: {chosen}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        h_lines, v_lines = self.extract_grid_lines(binary)
        graph = self._build_graph(binary, h_lines, v_lines)
        start_cell = self._cell_of_point(entrance, h_lines, v_lines)
        goal_cell = self._cell_of_point(exit_point, h_lines, v_lines)
        path_cells = self._astar_shortest_path(graph, start_cell, goal_cell)
        overlay = warped_frame.copy()
        path_pts_warped = []
        if path_cells:
            path_pts_warped = [self._cell_center(r, c, h_lines, v_lines) for (r, c) in path_cells]
            for i in range(len(path_pts_warped) - 1):
                cv2.line(overlay, path_pts_warped[i], path_pts_warped[i + 1], (255, 0, 255), 4, cv2.LINE_AA)
            if path_pts_warped:
                cv2.circle(overlay, path_pts_warped[0], 8, (0, 255, 0), -1)
                cv2.circle(overlay, path_pts_warped[-1], 8, (0, 0, 255), -1)
        if debug_dir:
            save_debug_image(debug_dir, "06_solved_warped.jpg", overlay)
        H_inv = np.linalg.inv(H_total)
        waypoints_original = []
        if path_pts_warped:
            waypoints_original = project_points(H_inv, path_pts_warped).astype(int).tolist()
        entry_original = project_points(H_inv, [entrance]).astype(int).tolist()[0] if entrance else None
        exit_original = project_points(H_inv, [exit_point]).astype(int).tolist()[0] if exit_point else None
        solved_on_camera = original_frame.copy()
        if waypoints_original:
            for i in range(len(waypoints_original) - 1):
                cv2.line(solved_on_camera, tuple(waypoints_original[i]),
                         tuple(waypoints_original[i + 1]), (255, 0, 255), 4, cv2.LINE_AA)
            cv2.circle(solved_on_camera, tuple(waypoints_original[0]), 8, (0, 255, 0), -1)
            cv2.circle(solved_on_camera, tuple(waypoints_original[-1]), 8, (0, 0, 255), -1)
        if debug_dir:
            save_debug_image(debug_dir, "07_solved_unwarped.jpg", solved_on_camera)
            waypoints_annotated = original_frame.copy()
            for i, (x, y) in enumerate(waypoints_original):
                cv2.circle(waypoints_annotated, (int(x), int(y)), 6, (255, 0, 0), -1, cv2.LINE_AA)
                cv2.putText(waypoints_annotated, str(i), (int(x) + 8, int(y) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
            if waypoints_original:
                cv2.circle(waypoints_annotated, tuple(waypoints_original[0]), 12, (0, 255, 0), 3, cv2.LINE_AA)
                cv2.circle(waypoints_annotated, tuple(waypoints_original[-1]), 12, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(waypoints_annotated, f"Waypoints: {len(waypoints_original)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
            save_debug_image(debug_dir, "08_waypoints_annotated.jpg", waypoints_annotated)
        verify = self._verify_solution_warped(binary, path_pts_warped)
        self.last_analysis = {
            'entrance': entrance,
            'exit': exit_point,
            'grid_size': (rows, cols),
            'result_image': result_image,
            'binary_image': binary,
            'original_warped': warped_frame,
            'grid_debug': grid_debug,
            'edges': edges_full,
            'solved_overlay': overlay,
            'entry_original': tuple(entry_original) if entry_original else None,
            'exit_original': tuple(exit_original) if exit_original else None,
            'waypoints_original': [tuple(p) for p in waypoints_original],
            'solved_on_camera': solved_on_camera,
            'camera_frame': original_frame.copy(),
            'path_cells': path_cells,
            'verify': verify
        }
        return self.last_analysis

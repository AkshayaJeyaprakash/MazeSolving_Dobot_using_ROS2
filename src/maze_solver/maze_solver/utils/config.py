#!/usr/bin/env python3
"""
Centralized configuration values used across the perception pipeline.
"""

from pathlib import Path

CAM_INDEX = 2
CAP_WIDTH = 640
CAP_HEIGHT = 480
OUT_SIZE = 600
MARGIN_PX = 2
DRAW_GUIDES = True

SHARED_DIR = Path("/tmp/maze_solver_stream")
SHARED_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE = SHARED_DIR / "state.json"
LIVE_FEED_FILE = SHARED_DIR / "live_feed.jpg"
WAYPOINTS_FILE = SHARED_DIR / "waypoints.jpg"
SOLUTION_FILE = SHARED_DIR / "solution.jpg"
ANALYSIS_FILE = SHARED_DIR / "analysis.jpg"

DEBUG_BASE_DIR = Path.cwd() / "debug"
DEBUG_BASE_DIR.mkdir(parents=True, exist_ok=True)

TARGET_PREVIEW_FPS = 6
MARKER_HOLD_SEC = 3.0
QUAD_SCORE_THRESHOLD = 0.45
PIXEL_STABILITY_TOL = 8.0

DOBOT_CORNERS = [
    (187, -165),
    (187, 155),
    (407, 155),
    (407, -165),
]

FIXED_Z = -25.0

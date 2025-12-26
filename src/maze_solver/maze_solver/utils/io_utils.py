#!/usr/bin/env python3
"""
File I/O helpers for debug assets and coordinate exports.
"""

from datetime import datetime
from pathlib import Path
import cv2

from .config import DEBUG_BASE_DIR

def create_debug_run_directory():
    """Create a timestamped debug directory for the current run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_dir = DEBUG_BASE_DIR / f"run_{timestamp}"
    debug_dir.mkdir(parents=True, exist_ok=True)
    return debug_dir

def save_debug_image(debug_dir: Path, filename: str, image) -> bool:
    """Persist an image under the debug directory and return success."""
    try:
        filepath = debug_dir / filename
        cv2.imwrite(str(filepath), image)
        return True
    except Exception:
        return False

def save_coordinates_csv(debug_dir: Path, filename: str, coords_dict) -> bool:
    """Write pixel and Dobot coordinate sequences to a CSV file and return success."""
    try:
        import csv
        filepath = debug_dir / filename
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['type', 'index', 'x', 'y', 'z'])
            for coord_type, coords_list in coords_dict.items():
                for i, coord in enumerate(coords_list):
                    if len(coord) == 2:
                        writer.writerow([coord_type, i, coord[0], coord[1], ''])
                    elif len(coord) == 3:
                        writer.writerow([coord_type, i, coord[0], coord[1], coord[2]])
        return True
    except Exception:
        return False

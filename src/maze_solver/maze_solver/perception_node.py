#!/usr/bin/env python3
"""
ROS 2 Perception Node for live maze solving and Dobot waypoint publishing.

This node captures camera frames, detects the maze board quadrilateral, auto-snapshots
when stable, solves the maze using utilities in utils/, generates display assets,
and publishes Dobot waypoints. All vision, geometry, I/O, and analysis logic resides
in utils/*. Only ROS-level orchestration is defined here.
"""

import json
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from typing import Iterable
from std_msgs.msg import String, Float32MultiArray

from utils.config import (
    CAM_INDEX, CAP_WIDTH, CAP_HEIGHT, OUT_SIZE, MARGIN_PX, DRAW_GUIDES,
    SHARED_DIR, STATE_FILE, LIVE_FEED_FILE, WAYPOINTS_FILE, SOLUTION_FILE,
    ANALYSIS_FILE, DEBUG_BASE_DIR, TARGET_PREVIEW_FPS, MARKER_HOLD_SEC,
    QUAD_SCORE_THRESHOLD, PIXEL_STABILITY_TOL, FIXED_Z
)
from utils.vision import detect_maze_quad, warp_square, StabilityLock
from utils.analyzer import LiveMazeAnalyzer
from utils.geometry import convert_pixels_to_dobot
from utils.io_utils import create_debug_run_directory, save_debug_image, save_coordinates_csv


class PerceptionNode(Node):
    """
    ROS 2 node that manages camera capture, auto-snapshot stability logic, maze solving,
    and Dobot waypoint publication. It writes lightweight state for external dashboards.
    """

    def __init__(self):
        """Initialize publishers, subscriptions, internal state, and external state file."""
        super().__init__('perception_node')
        self.status_pub = self.create_publisher(String, '/maze/status', 10)
        self.waypoints_pub = self.create_publisher(Float32MultiArray, '/maze/waypoints_dobot', 10)
        self.info_pub = self.create_publisher(String, '/maze/info', 10)
        self.command_sub = self.create_subscription(String, '/maze/command', self.command_callback, 10)
        self.running = False
        self.auto_capture_enabled = False
        self.start_color = 'green'
        self.camera = None
        self.analyzer = None
        self.lock = None
        self.get_logger().info('Perception Node initialized')
        self.update_state({'status': 'idle', 'has_results': False})

    def update_state(self, state_dict):
        """Persist minimal state used by external visualizers."""
        try:
            with open(STATE_FILE, 'w') as f:
                json.dump(state_dict, f)
        except Exception as e:
            self.get_logger().warn(f'Could not update state: {e}')

    def publish_status(self, status: str):
        """Publish a human-readable run status and update external state."""
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
        self.update_state({'status': status, 'has_results': status == 'solved'})

    def command_callback(self, msg: String):
        """Handle textual commands to set start color, start/stop/reset, and control auto-capture."""
        command = msg.data.lower()
        self.get_logger().info(f'Received command: {command}')
        if 'green' in command:
            self.start_color = 'green'
        elif 'red' in command:
            self.start_color = 'red'
        self.clean_shared_dir(except_paths=(LIVE_FEED_FILE,))
        if any(word in command for word in ['start', 'solve', 'begin', 'run']):
            if not self.running:
                self.start_detection()
            else:
                self.auto_capture_enabled = True
                self.lock.reset()
                self.get_logger().info(f'Auto-capture enabled from {self.start_color.upper()}')
        elif 'stop' in command or 'abort' in command:
            self.stop_detection()
        elif 'reset' in command:
            self.reset()

    def clean_shared_dir(self, except_paths: Iterable = ()):
        """Remove files in the shared directory except explicit paths to keep."""
        keep = {p.resolve() for p in except_paths}
        try:
            for p in SHARED_DIR.iterdir():
                if p.is_file() and p.resolve() not in keep:
                    try:
                        p.unlink()
                    except Exception as e:
                        self.get_logger().warn(f"Could not remove {p.name}: {e}")
        except Exception as e:
            self.get_logger().warn(f"Cleanup failed in {SHARED_DIR}: {e}")

    def start_detection(self):
        """Open the camera, initialize analysis utilities, and start the preview timer."""
        if self.running:
            self.get_logger().info('Camera already running')
            return
        self.get_logger().info(f'Starting detection with start_color={self.start_color}')
        self.running = True
        self.auto_capture_enabled = True
        try:
            self.camera = cv2.VideoCapture(CAM_INDEX)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
            if not self.camera.isOpened():
                self.publish_status('error')
                self.get_logger().error(f'Could not open camera {CAM_INDEX}')
                return
        except Exception as e:
            self.publish_status('error')
            self.get_logger().error(f'Camera error: {e}')
            return
        self.analyzer = LiveMazeAnalyzer()
        self.lock = StabilityLock(hold_sec=MARKER_HOLD_SEC, px_tol=PIXEL_STABILITY_TOL)
        self.publish_status('running')
        self.timer = self.create_timer(1.0 / TARGET_PREVIEW_FPS, self.camera_loop)
        self.get_logger().info('Camera started')

    def camera_loop(self):
        """Capture frames, detect the maze quad, manage stability, and trigger processing."""
        if not self.running or self.camera is None:
            return
        ret, frame = self.camera.read()
        if not ret:
            return
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        quad, score = detect_maze_quad(gray)
        view = frame.copy()
        if quad is not None:
            if DRAW_GUIDES:
                for i in range(4):
                    p1 = tuple(quad[i].astype(int))
                    p2 = tuple(quad[(i + 1) % 4].astype(int))
                    cv2.line(view, p1, p2, (0, 255, 0), 3)
            if self.auto_capture_enabled and score >= QUAD_SCORE_THRESHOLD:
                stable_time = self.lock.update(quad)
                remaining = max(0.0, MARKER_HOLD_SEC - stable_time)
                if remaining <= 0.0:
                    self.get_logger().info(f'Maze stable with score {score:.2f}')
                    self.publish_status('solving')
                    self.process_maze(frame, quad)
                    self.auto_capture_enabled = False
                    self.lock.reset()
                    self.get_logger().info('Solved. Camera continues for next command')
                    return
                else:
                    cv2.putText(view, f"Quad score {score:.2f} - snapshot in {remaining:.1f}s",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(view, f"Start color: {self.start_color.upper()}",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            else:
                self.lock.reset()
                if not self.auto_capture_enabled:
                    cv2.putText(view, f"Quad detected (score: {score:.2f}) - waiting for solve",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
                else:
                    cv2.putText(view, f"Quad score {score:.2f} (< {QUAD_SCORE_THRESHOLD}) - hold steady",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2, cv2.LINE_AA)
        else:
            self.lock.reset()
            cv2.putText(view, "Show the maze board; seeking rectangular quad",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        if self.auto_capture_enabled:
            cv2.putText(view, "ROS Maze Solver (auto-capture enabled)",
                        (10, view.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(view, f"Command: solve from {self.start_color.upper()}",
                        (10, view.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(view, "Waiting for next solve command",
                        (10, view.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
            cv2.putText(view, "Camera in preview mode",
                        (10, view.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
        cv2.imwrite(str(LIVE_FEED_FILE), view)

    def process_maze(self, frame, quad):
        """Warp the maze, analyze and solve, verify, persist artifacts, and publish Dobot waypoints."""
        self.get_logger().info('Processing maze')
        debug_dir = create_debug_run_directory()
        self.get_logger().info(f'Debug directory: {debug_dir}')
        save_debug_image(debug_dir, "00_captured_maze.jpg", frame)
        warped, H_warp = warp_square(frame, quad)
        save_debug_image(debug_dir, "00_warped_maze.jpg", warped)
        result = self.analyzer.analyze_snapshot(warped, H_warp, frame, self.start_color, debug_dir)
        if not result or not result.get('waypoints_original'):
            self.get_logger().error('Failed to solve maze')
            self.publish_status('error')
            return
        waypoints_camera = np.array(result['waypoints_original'], dtype=np.float32)
        dobot_waypoints = convert_pixels_to_dobot(waypoints_camera, CAP_WIDTH, CAP_HEIGHT)
        coords_dict = {
            'pixel': [(int(x), int(y)) for x, y in waypoints_camera],
            'dobot': [(float(x), float(y), FIXED_Z) for x, y in dobot_waypoints]
        }
        save_coordinates_csv(debug_dir, "coordinates.csv", coords_dict)
        try:
            summary_file = debug_dir / "summary.txt"
            with open(summary_file, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("MAZE SOLVER RUN SUMMARY\n")
                f.write("=" * 60 + "\n\n")
                from datetime import datetime
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Start Color: {self.start_color.upper()}\n")
                f.write(f"Grid Size: {result['grid_size'][0]}x{result['grid_size'][1]}\n")
                f.write(f"Path Length: {len(result.get('path_cells', []))} cells\n")
                f.write(f"Waypoints: {len(waypoints_camera)}\n\n")
                f.write("Pixel Coordinates:\n")
                f.write("-" * 40 + "\n")
                for i, (x, y) in enumerate(waypoints_camera):
                    f.write(f"  [{i:2d}] ({x:4.0f}, {y:4.0f})\n")
                f.write("\n")
                f.write("Dobot Coordinates:\n")
                f.write("-" * 40 + "\n")
                for i, (x, y) in enumerate(dobot_waypoints):
                    f.write(f"  [{i:2d}] ({x:7.2f}, {y:7.2f}, {FIXED_Z:7.2f})\n")
                f.write("\n")
                verify = result.get('verify', {})
                f.write("Path Verification:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Valid: {verify.get('ok', False)}\n")
                f.write(f"  Collision Ratio: {verify.get('coll_ratio', 0):.4f}\n")
                f.write(f"  Samples: {verify.get('samples', 0)}\n")
                f.write("\n")
                f.write("Files Saved:\n")
                f.write("-" * 40 + "\n")
                f.write("  00_captured_maze.jpg\n")
                f.write("  00_warped_maze.jpg\n")
                f.write("  01_binary_initial.jpg\n")
                f.write("  02_dot_detection.jpg\n")
                f.write("  03_binary_dots_carved.jpg\n")
                f.write("  04_grid_detection.jpg\n")
                f.write("  05_edges.jpg\n")
                f.write("  06_solved_warped.jpg\n")
                f.write("  07_solved_unwarped.jpg\n")
                f.write("  08_waypoints_annotated.jpg\n")
                f.write("  coordinates.csv\n")
                f.write("  summary.txt\n")
                f.write("\n")
                f.write("=" * 60 + "\n")
            self.get_logger().info(f'Summary saved to: {summary_file}')
        except Exception as e:
            self.get_logger().warn(f'Could not save summary: {e}')
        self.publish_dobot_waypoints(dobot_waypoints)
        grid_size = result['grid_size']
        path_cells = result.get('path_cells', [])
        info = f"Grid: {grid_size[0]}x{grid_size[1]}, Path: {len(path_cells)} cells, Waypoints: {len(dobot_waypoints)}"
        msg = String()
        msg.data = info
        self.info_pub.publish(msg)
        self.generate_display_images(result, waypoints_camera, frame)
        self.publish_status('solved')
        self.get_logger().info(f"Solved {grid_size[0]}x{grid_size[1]} with {len(dobot_waypoints)} waypoints")
        self.get_logger().info(f"Debug files saved to: {debug_dir.absolute()}")

    def generate_display_images(self, result, waypoints_camera, frame):
        """Render and persist numbered waypoints, solution overlay, and warped analysis images."""
        waypoints_img = frame.copy()
        for i, (x, y) in enumerate(waypoints_camera):
            cv2.circle(waypoints_img, (int(x), int(y)), 6, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(waypoints_img, str(i), (int(x) + 8, int(y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        if len(waypoints_camera) > 0:
            start_pt = waypoints_camera[0]
            end_pt = waypoints_camera[-1]
            cv2.circle(waypoints_img, (int(start_pt[0]), int(start_pt[1])), 12, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.circle(waypoints_img, (int(end_pt[0]), int(end_pt[1])), 12, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.putText(waypoints_img, f"Waypoints: {len(waypoints_camera)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
        solution_img = result.get('solved_on_camera', frame.copy())
        cv2.putText(solution_img, f"Started from {self.start_color.upper()}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
        analysis_img = result.get('original_warped', np.zeros((OUT_SIZE, OUT_SIZE, 3), dtype=np.uint8)).copy()
        entrance = result.get('entrance', None)
        exit_point = result.get('exit', None)
        if entrance:
            cv2.circle(analysis_img, entrance, 15, (0, 0, 0), 3)
            cv2.putText(analysis_img, "START", (entrance[0] - 30, entrance[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        if exit_point:
            cv2.circle(analysis_img, exit_point, 15, (0, 0, 0), 3)
            cv2.putText(analysis_img, "END", (exit_point[0] - 20, exit_point[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.imwrite(str(WAYPOINTS_FILE), waypoints_img)
        cv2.imwrite(str(SOLUTION_FILE), solution_img)
        cv2.imwrite(str(ANALYSIS_FILE), analysis_img)
        self.get_logger().info('Display images saved')

    def publish_dobot_waypoints(self, dobot_pts):
        """Publish Dobot waypoints as Float32MultiArray with fixed Z appended."""
        msg = Float32MultiArray()
        msg.data = [float(coord) for pt in dobot_pts for coord in [pt[0], pt[1], FIXED_Z]]
        self.waypoints_pub.publish(msg)
        self.get_logger().info(f'Published {len(dobot_pts)} waypoints to Dobot')

    def stop_detection(self):
        """Stop the camera loop, release the device, and set idle status."""
        self.running = False
        if hasattr(self, 'timer'):
            self.timer.cancel()
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.publish_status('idle')
        self.get_logger().info('Camera stopped')

    def reset(self):
        """Reset the node, clear display artifacts, and disable auto-capture."""
        self.stop_detection()
        for f in [LIVE_FEED_FILE, WAYPOINTS_FILE, SOLUTION_FILE, ANALYSIS_FILE]:
            if f.exists():
                f.unlink()
        if self.lock:
            self.lock.reset()
        self.auto_capture_enabled = False
        self.get_logger().info('Node reset')


def main(args=None):
    """Initialize rclpy, run the node event loop, and clean up on shutdown."""
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_detection()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

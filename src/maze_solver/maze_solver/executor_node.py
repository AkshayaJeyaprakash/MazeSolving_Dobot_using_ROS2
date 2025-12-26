#!/usr/bin/env python3
"""
ROS 2 Node: Dobot Executor with Auto-Home

This module defines a ROS 2 node that executes a sequence of (X, Y) waypoints on a Dobot arm
at a fixed Z and R orientation. It supports both real-hardware and simulation modes, auto-starts
execution upon receiving /maze/waypoints_dobot, publishes progress and status topics, and
returns the robot to home after successful completion. Services are defined for execution control
(start via subscription, stop, and home).
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32, Int32, Float32MultiArray
import numpy as np
import time

try:
    from pydobot.dobot import MODE_PTP
    import pydobot
    PYDOBOT_AVAILABLE = True
    print("✓ pydobot imported successfully!")
except ImportError as e:
    PYDOBOT_AVAILABLE = False
    print(f"✗ pydobot import failed: {e}")
except Exception as e:
    PYDOBOT_AVAILABLE = False
    print(f"✗ pydobot error: {e}")

DOBOT_PORT = "/dev/ttyACM0"
FIXED_Z = -25.0
FIXED_R = 0.0
SPEED_LIN = 10
SPEED_ANG = 10


class ExecutorNode(Node):
    """
    Executes Dobot waypoint paths with progress/status publishing and auto-home.

    This node:
    - Subscribes to /maze/waypoints_dobot (Float32MultiArray) and auto-runs received paths.
    - Publishes /dobot/status (String), /dobot/progress (Float32), and /dobot/current_waypoint (Int32).
    - Attempts to control a real Dobot via pydobot; if unavailable or connection fails, runs in simulation.
    - After successful execution, commands the Dobot to return to home.
    - Exposes stop and home control methods intended to be bound to services externally.
    """

    def __init__(self):
        """Initialize publishers, subscriber, hardware connection, and initial status."""
        super().__init__('executor_node')
        self.status_pub = self.create_publisher(String, '/dobot/status', 10)
        self.progress_pub = self.create_publisher(Float32, '/dobot/progress', 10)
        self.waypoint_pub = self.create_publisher(Int32, '/dobot/current_waypoint', 10)
        self.waypoints_sub = self.create_subscription(
            Float32MultiArray, '/maze/waypoints_dobot', self.waypoints_callback, 10
        )
        self.device = None
        self.connected = False
        self.executing = False
        self.should_stop = False
        self.simulation_mode = False
        if PYDOBOT_AVAILABLE:
            self.connect_dobot()
        else:
            self.get_logger().warn('pydobot not available - running in SIMULATION mode')
            self.simulation_mode = True
        self.get_logger().info('Executor Node initialized')
        if self.simulation_mode:
            self.get_logger().info('SIMULATION MODE: Will print waypoints instead of moving')
        self.publish_status('idle')

    def publish_status(self, status: str):
        """Publish a human-readable Dobot status string on /dobot/status."""
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
        self.get_logger().info(f'Dobot status: {status}')

    def publish_progress(self, progress: float):
        """Publish a normalized execution progress in [0.0, 1.0] on /dobot/progress."""
        msg = Float32()
        msg.data = float(progress)
        self.progress_pub.publish(msg)

    def publish_current_waypoint(self, index: int):
        """Publish the zero-based index of the current waypoint on /dobot/current_waypoint."""
        msg = Int32()
        msg.data = index
        self.waypoint_pub.publish(msg)

    def connect_dobot(self) -> bool:
        """Attempt to connect to the Dobot and configure speed; fall back to simulation on failure."""
        if not PYDOBOT_AVAILABLE:
            self.get_logger().warn('pydobot not available - running in simulation mode')
            self.simulation_mode = True
            return False
        try:
            self.device = pydobot.Dobot(port=DOBOT_PORT)
            self.device.speed(SPEED_LIN, SPEED_ANG)
            self.connected = True
            self.simulation_mode = False
            self.get_logger().info(f'Connected to Dobot on {DOBOT_PORT}')
            return True
        except Exception as e:
            self.get_logger().warn(f'Could not connect to Dobot: {e}')
            self.get_logger().info('Switching to SIMULATION mode')
            self.connected = False
            self.simulation_mode = True
            return False

    def disconnect_dobot(self):
        """Close the Dobot connection if open and reset connection state."""
        if self.device is not None:
            try:
                self.device.close()
                self.get_logger().info('Dobot disconnected')
            except Exception as e:
                self.get_logger().warn(f'Error disconnecting: {e}')
            finally:
                self.device = None
                self.connected = False

    def waypoints_callback(self, msg: Float32MultiArray):
        """Handle received Dobot waypoints and trigger execution if not already running."""
        if self.executing:
            self.get_logger().warn('Already executing, ignoring new waypoints')
            return
        num_waypoints = len(msg.data) // 3
        self.get_logger().info(f'Received {num_waypoints} Dobot waypoints')
        data = np.array(msg.data).reshape(-1, 3)
        waypoints = data[:, :2]
        self.execute_waypoints(waypoints, FIXED_Z, SPEED_LIN)

    def execute_waypoints(self, waypoints: np.ndarray, z_height: float, speed: float) -> bool:
        """
        Execute a sequence of (X, Y) waypoints at fixed Z and R, publishing progress and status.

        Args:
            waypoints: Array of shape (N, 2) containing (X, Y) coordinates.
            z_height: Fixed Z height used for all moves.
            speed: Linear speed parameter (applied at connection time for hardware).

        Returns:
            True on successful completion (including auto-home), False otherwise.
        """
        if self.executing:
            self.get_logger().warn('Already executing')
            return False
        self.executing = True
        self.should_stop = False
        self.publish_status('moving')
        self.publish_progress(0.0)
        start_time = time.time()
        num_waypoints = len(waypoints)
        completed = 0
        if self.simulation_mode:
            self.get_logger().info('=' * 60)
            self.get_logger().info('SIMULATION MODE - Waypoint Execution')
            self.get_logger().info('=' * 60)
        try:
            for i, (x, y) in enumerate(waypoints):
                if self.should_stop:
                    self.get_logger().warn('Execution stopped by user')
                    break
                if self.simulation_mode:
                    self.get_logger().info(f'[{i+1}/{num_waypoints}] Moving to: X={x:7.2f}, Y={y:7.2f}, Z={z_height:7.2f}')
                    time.sleep(0.3)
                    completed += 1
                else:
                    self.get_logger().info(f'Moving to waypoint {i+1}/{num_waypoints}: ({x:.2f}, {y:.2f})')
                    try:
                        qid = self.device.move_to(
                            mode=int(MODE_PTP.MOVJ_XYZ),
                            x=float(x),
                            y=float(y),
                            z=z_height,
                            r=FIXED_R
                        )
                        self.device.wait_for_cmd(qid)
                        completed += 1
                    except KeyboardInterrupt:
                        self.get_logger().warn('Interrupted by keyboard')
                        if not self.simulation_mode:
                            self.device.force_stop_and_go()
                        break
                    except Exception as e:
                        self.get_logger().error(f'Error at waypoint {i}: {e}')
                        break
                progress = (i + 1) / num_waypoints
                self.publish_progress(progress)
                self.publish_current_waypoint(i)
            execution_time = time.time() - start_time
            if completed == num_waypoints:
                self.publish_status('complete')
                self.publish_progress(1.0)
                if self.simulation_mode:
                    self.get_logger().info('=' * 60)
                    self.get_logger().info(f'Simulation complete! {completed}/{num_waypoints} waypoints in {execution_time:.2f}s')
                    self.get_logger().info('=' * 60)
                else:
                    self.get_logger().info(f'Execution complete! {completed}/{num_waypoints} waypoints in {execution_time:.2f}s')
                self.get_logger().info('Returning to home position...')
                time.sleep(0.5)
                home_success = self.go_home()
                if home_success:
                    if self.simulation_mode:
                        self.get_logger().info('[Simulation] Successfully returned to home position')
                    else:
                        self.get_logger().info('Successfully returned to home position')
                    self.publish_status('idle')
                else:
                    self.get_logger().warn('Could not return to home position')
                return True
            else:
                self.publish_status('error')
                self.get_logger().error(f'Execution incomplete: {completed}/{num_waypoints}')
                return False
        except Exception as e:
            self.get_logger().error(f'Execution failed: {e}')
            self.publish_status('error')
            return False
        finally:
            self.executing = False

    def stop_execution(self) -> bool:
        """Issue an emergency stop; halts current motion and publishes idle status."""
        if not self.executing:
            self.get_logger().warn('Not executing, nothing to stop')
            return False
        self.should_stop = True
        if self.connected and self.device is not None and not self.simulation_mode:
            try:
                self.device.force_stop_and_go()
                self.get_logger().info('Emergency stop executed')
            except Exception as e:
                self.get_logger().error(f'Could not stop robot: {e}')
                return False
        else:
            self.get_logger().info('[Simulation] Emergency stop executed')
        self.publish_status('idle')
        return True

    def go_home(self) -> bool:
        """Return the robot to its home position, using device.home() if connected."""
        home_x, home_y = 200.0, 0.0
        if self.simulation_mode:
            self.get_logger().info('[Simulation] Moving to home')
            time.sleep(0.3)
            self.get_logger().info('[Simulation] Home position reached')
            return True
        if not self.connected or self.device is None:
            self.get_logger().error('Dobot not connected')
            return False
        self.get_logger().info(f'Moving to home position ({home_x}, {home_y})...')
        try:
            qid = self.device.home()
            self.device.wait_for_cmd(qid)
            self.get_logger().info('✓ Home position reached')
            return True
        except Exception as e:
            self.get_logger().error(f'Could not move home: {e}')
            return False

    def __del__(self):
        """Ensure device connection is closed on object destruction."""
        self.disconnect_dobot()


def main(args=None):
    """Initialize rclpy and spin the ExecutorNode until shutdown or interrupt."""
    rclpy.init(args=args)
    node = ExecutorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.disconnect_dobot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

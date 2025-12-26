#!/usr/bin/env python3
"""
Streamlit Frontend for Maze Solver
Displays images saved by the ROS2 perception node with live annotations

Run with: streamlit run streamlit_display.py
"""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import time
import json

SHARED_DIR = Path("/tmp/maze_solver_stream")
STATE_FILE = SHARED_DIR / "state.json"
LIVE_FEED_FILE = SHARED_DIR / "live_feed.jpg"
WAYPOINTS_FILE = SHARED_DIR / "waypoints.jpg"
SOLUTION_FILE = SHARED_DIR / "solution.jpg"
ANALYSIS_FILE = SHARED_DIR / "analysis.jpg"

def create_placeholder_image(text, width=640, height=480):
    """Create placeholder image with text"""
    img = np.ones((height, width, 3), dtype=np.uint8) * 50
    lines = text.split('\n')
    y_offset = height // 2 - (len(lines) * 30) // 2
    
    for i, line in enumerate(lines):
        text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        x_pos = (width - text_size[0]) // 2
        y_pos = y_offset + i * 40
        cv2.putText(img, line, (x_pos, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
    
    return img

def load_image(filepath, placeholder_text="Waiting..."):
    """Load image or return placeholder"""
    if filepath.exists():
        try:
            img = cv2.imread(str(filepath))
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            st.error(f"Error loading {filepath.name}: {e}")
    return create_placeholder_image(placeholder_text)

def get_state():
    """Get current state from ROS2 node"""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Error reading state: {e}")
    return {'status': 'waiting', 'has_results': False}

def get_status_emoji(status):
    """Get emoji for status"""
    status_emojis = {
        'idle': '🟢',
        'running': '🟢',
        'solving': '🟡',
        'solved': '✅',
        'error': '❌',
        'waiting': '⚪'
    }
    return status_emojis.get(status, '⚪')

def get_status_message(status):
    """Get friendly status message"""
    messages = {
        'idle': 'Idle - Waiting for command',
        'running': 'Camera running - Waiting for maze detection',
        'solving': 'Solving maze...',
        'solved': 'Maze solved successfully!',
        'error': 'Error occurred',
        'waiting': 'Waiting for ROS2 node to start...'
    }
    return messages.get(status, 'Unknown status')

TARGET_DISPLAY_HEIGHT = 480

def resize_to_height(img, target_h=TARGET_DISPLAY_HEIGHT):
    if img is None:
        return None
    h, w = img.shape[:2]
    if h == 0:
        return img
    if h == target_h:
        return img
    scale = float(target_h) / float(h)
    new_w = max(1, int(round(w * scale)))
    # INTER_AREA for downscale, INTER_CUBIC for upscale
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(img, (new_w, target_h), interpolation=interp)


def main():
    st.set_page_config(
        layout="wide", 
        page_title="Maze Solver DoLi",
        page_icon="🤖"
    )
    
    st.markdown("""
        <h1 style='text-align: center; color: #4CAF50;'>
            🤖 Maze Solver - Live Perception System
        </h1>
    """, unsafe_allow_html=True)
    
    SHARED_DIR.mkdir(parents=True, exist_ok=True)
    
    state = get_state()
    status = state.get('status', 'waiting')
    has_results = state.get('has_results', False)
    
    emoji = get_status_emoji(status)
    message = get_status_message(status)
    
    status_colors = {
        'idle': 'info',
        'running': 'info',
        'solving': 'warning',
        'solved': 'success',
        'error': 'error',
        'waiting': 'info'
    }
    
    status_color = status_colors.get(status, 'info')
    
    if status_color == 'info':
        st.info(f"{emoji} **Status:** {message}")
    elif status_color == 'warning':
        st.warning(f"{emoji} **Status:** {message}")
    elif status_color == 'success':
        st.success(f"{emoji} **Status:** {message}")
    elif status_color == 'error':
        st.error(f"{emoji} **Status:** {message}")
    
    with st.expander("📖 How to use this system"):
        st.markdown("""
        ### Setup Instructions
        
        **1. Start the ROS2 perception node:**
        ```bash
        ros2 run maze_solver perception_node
        ```
        
        **2. Send commands to solve the maze:**
        ```bash
        # Solve starting from green dot
        ros2 topic pub /maze/command std_msgs/String "data: 'solve from green'" -1
        
        # Solve starting from red dot
        ros2 topic pub /maze/command std_msgs/String "data: 'solve from red'" -1
        ```
        
        **3. Additional commands:**
        ```bash
        # Stop the camera
        ros2 topic pub /maze/command std_msgs/String "data: 'stop'" -1
        
        # Reset the system
        ros2 topic pub /maze/command std_msgs/String "data: 'reset'" -1
        ```
        
        ### What you'll see:
        - **Live Camera Feed**: Real-time view with maze detection overlay
        - **Waypoints**: Numbered path points for robot navigation
        - **Solution Path**: Complete path drawn on the camera view
        - **Maze Analysis**: Warped maze view with start/end markers
        
        ### Annotations in Live Feed:
        - **Green outline**: Detected maze boundary
        - **Countdown timer**: Shows when auto-capture will trigger
        - **Status messages**: Current operation status
        - **Start color**: Which dot (red/green) is the starting point
        """)
    
    st.markdown("---")
    
    if not has_results:
        st.info("🎥 Camera feed will appear here once the ROS2 node is running. Send a 'solve' command to begin maze solving.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🎥 Live Camera Feed")
        st.markdown("*Real-time camera view with detection overlay*")
        live_placeholder = st.empty()
    
    with col2:
        st.markdown("### 🔍 Maze Analysis")
        st.markdown("*Warped maze with start/end markers*")
        analysis_placeholder = st.empty()
    
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### ✅ Solution Path")
        st.markdown("*Complete path overlay on camera view*")
        solution_placeholder = st.empty()
    
    with col4:
        st.markdown("### 🔢 Waypoints (Numbered)")
        st.markdown("*Numbered path points for navigation*")
        waypoints_placeholder = st.empty()
    
    st.markdown("---")
    refresh_info = st.empty()
    refresh_rate = 0.1
    frame_count = 0
    
    while True:
        frame_count += 1
        
        if frame_count % 10 == 0:
            state = get_state()
            status = state.get('status', 'waiting')
        
        live_img = load_image(LIVE_FEED_FILE, "Waiting for camera feed...\nStart the ROS2 node")
        waypoints_img = load_image(WAYPOINTS_FILE, "Waypoints will appear here\nafter solving the maze")
        solution_img = load_image(SOLUTION_FILE, "Solution path will appear here\nafter solving the maze")
        analysis_img = load_image(ANALYSIS_FILE, "Maze analysis will appear here\nafter solving the maze")
        
        live_placeholder.image(live_img, channels="RGB", use_container_width=True)
        waypoints_placeholder.image(waypoints_img, channels="RGB", use_container_width=True)
        solution_placeholder.image(solution_img, channels="RGB", use_container_width=True)
        analysis_placeholder.image(analysis_img, channels="RGB", use_container_width=True)
        
        refresh_info.caption(f"🔄 Auto-refreshing at ~10 FPS | Status: {status}")
        
        time.sleep(refresh_rate)

if __name__ == '__main__':
    main()
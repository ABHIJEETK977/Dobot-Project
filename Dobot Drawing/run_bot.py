# sketch_frame_preview.py (FIXED VERSION WITH PAUSE + ORIENTATION CONTROL)
# Rewritten to handle workspace reachability constraints at different Z heights,
# provide interactive pause/resume and configurable mirroring.

import os
import sys
import json
import platform
import time

import cv2       
import numpy as np  

# Windows-only keyboard input for pause/resume
if platform.system() == "Windows":
    import msvcrt
else:
    msvcrt = None

# ============================================================================
# PATHS / SDK
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOBOT_LIB_PATH = os.path.join(BASE_DIR, "dobot_lib")
if DOBOT_LIB_PATH not in sys.path:
    sys.path.insert(0, DOBOT_LIB_PATH)

import dobot_lib.DobotDllType as dType  # Dobot SDK

# ============================================================================
# USER CONFIG - HEIGHTS (robot/world mm)
# ============================================================================
Z_SAFE = -32.0      # Safe height for moving between points (no collision risk)
Z_DRAW = -42.0      # Drawing height (RAISED from -48.0 to increase XY reachability)
R_HEAD = 0.0        # Rotation axis

# ============================================================================
# CP (Continuous Path) PARAMETERS
# ============================================================================
# These are NOT the same as PTP params. Tune as needed.
CP_PLAN_ACC     = 200.0
CP_JUNCTION_VEL = 15.0
CP_ACC          = 200.0

# CP_PLAN_ACC     = 800.0     # path planning acceleration
# CP_JUNCTION_VEL = 80.0      # cornering / junction velocity
# CP_ACC          = 800.0     # execution acceleration
CP_REALTIME     = 0         # 0 = queued path; 1 = realtime track (leave 0 here)

# ============================================================================
# CONNECTION CONFIG
# ============================================================================
PORT = "COM15"
SKETCH_FILE_PATH = os.path.join(BASE_DIR, "robot_commands_albert.json")
# ============================================================================
# MOTION PARAMETERS
# ============================================================================
# Standard motion parameters
VELOCITY = 350.0
ACCELERATION = 1100.0

# Reduced velocities for boundary operations (safer at workspace limits)
VELOCITY_BOUNDARY = 350.0
ACCELERATION_BOUNDARY = 1100.0

# ============================================================================
# BOUNDARY (FRAME) CONFIGURATION - WORKSPACE-OPTIMIZED
# ============================================================================
# Define the visible frame in robot coordinates.
#   "TL"     -> BOUND_X/BOUND_Y mean TOP-LEFT of the visible frame
#   "CENTER" -> BOUND_X/BOUND_Y mean CENTER of the visible frame
BOUND_ORIGIN_MODE = "tl"  # "TL" or "CENTER"

# If your robot +Y is opposite the image/page +Y, flip the Y extent of the frame.
Y_AXIS_FLIP = False  # set True if your frame goes the "wrong" way in Y

# Frame parameters (already tuned for reachability)
BOUND_X = 170.0
BOUND_Y = -70.0
BOUND_W = 100.0
BOUND_H = 140.0
INNER_MARGIN = 8.0

# BOUND_X = 200.0     # top-left X or center X (see BOUND_ORIGIN_MODE)
# BOUND_Y = -90.0     # CHANGED from -50.0 to -20.0 (moved closer in Y direction)
# BOUND_W = 120.0     # visible frame width (mm)
# BOUND_H = 180.0     # CHANGED from 150.0 to 120.0 (reduced height for safety margin)

# Sketch must stay this far inside the frame line
# INNER_MARGIN = 5.0

# Allow uniform auto-scaling to fit inside the inner rectangle
ALLOW_SCALING = True

# Preview each corner of the frame at Z_SAFE before drawing it
PREVIEW_CORNERS = True

# Enable workspace mapping and diagnostics
ENABLE_WORKSPACE_DIAGNOSTICS = False  # Set True to map workspace before drawing

# ============================================================================
# SKETCH ORIENTATION CONTROL (FIX FOR MIRROR/INVERTED DRAWING)
# ============================================================================
# These are applied to the commands BEFORE planning the transform.
MIRROR_X = False
MIRROR_Y = True

# ============================================================================
# PAUSE / RESUME CONTROL
# ============================================================================
# Interactive pause/resume in the console:
#   - Press 'P' to pause after the current command finishes.
#   - Robot will lift vertically to Z_SAFE at the current XY.
#   - Press 'R' to resume from that same XY.
#   - Press 'Q' to abort.
ENABLE_KEYBOARD_PAUSE = True

# ============================================================================
# COARSE WORKSPACE GUARD (optional)
# ============================================================================
WORK_MIN_X, WORK_MAX_X = -75.0, 350.0
WORK_MIN_Y, WORK_MAX_Y = -120.0, 200.0

# ============================================================================
# NAME BOX CONFIG
# ============================================================================
ENABLE_NAME_BOX = True       # set False if you want to disable name feature
NAME_BOX_HEIGHT = 25.0       # height of the rectangle under the frame (mm)
NAME_TEXT_MARGIN = 4.0       # inner margin inside the name box (mm)

API = None

# ============================================================================
# UTILITIES - COMMAND QUEUEING
# ============================================================================
def wait_for_command(last_idx):
    """Wait until the queued command index is reached."""
    if API is None:
        return
    cur = dType.GetQueuedCmdCurrentIndex(API)[0]
    while cur < last_idx:
        dType.dSleep(100)
        cur = dType.GetQueuedCmdCurrentIndex(API)[0]

def get_current_position():
    """Get current robot pose."""
    if API is None:
        return None
    pose = dType.GetPose(API)
    return pose[0], pose[1], pose[2], pose[3]  # x, y, z, r

# ============================================================================
# CONNECTION & SETUP
# ============================================================================
def connect_to_dobot():
    """Connect to Dobot robot."""
    global API
    if platform.system() != "Windows":
        print("Error: Only Windows DLL path configured.")
        return None

    dll_path = os.path.join(DOBOT_LIB_PATH, "DobotDll.dll")
    if not os.path.exists(dll_path):
        print(f"Error: DLL not found at {dll_path}")
        return None

    API = dType.load()
    if not API:
        print("Error: Failed to load DobotDll.dll")
        return None

    res = dType.ConnectDobot(API, PORT, 115200)[0]
    if res == dType.DobotConnect.DobotConnect_NoError:
        dType.SetQueuedCmdClear(API)
        dType.SetQueuedCmdStartExec(API)
        return API

    if res == dType.DobotConnect.DobotConnect_NotFound:
        print(f"Error: Dobot not found on {PORT}")
    elif res == dType.DobotConnect.DobotConnect_Occupied:
        print(f"Error: Port {PORT} is occupied")
    else:
        print(f"Error: Connect code {res}")
    API = None
    return None

def setup_dobot(api):
    """Initialize robot, home, and set motion parameters."""
    dType.ClearAllAlarmsState(api)

    # Set joint motion parameters
    dType.SetPTPJointParams(
        api,
        VELOCITY, ACCELERATION,
        VELOCITY, ACCELERATION,
        VELOCITY, ACCELERATION,
        VELOCITY, ACCELERATION, 0
    )

    # Set Cartesian motion parameters
    dType.SetPTPCoordinateParams(api, VELOCITY, ACCELERATION, VELOCITY, ACCELERATION, 0)

    # Set CP params
    dType.SetCPParams(api, CP_PLAN_ACC, CP_JUNCTION_VEL, CP_ACC, CP_REALTIME, 0)
    
    # Home the robot
    last = dType.SetHOMECmd(api, 0, 1)[0]
    wait_for_command(last)

    pose = dType.GetPose(api)
    print(f"✓ Homed. Pose: x={pose[0]:.2f} y={pose[1]:.2f} z={pose[2]:.2f} r={pose[3]:.2f}")
    return pose[0], pose[1], pose[2], pose[3]

# ============================================================================
# GEOMETRY HELPERS
# ============================================================================
def compute_bbox(commands):
    """Return (xmin, xmax, ymin, ymax) from JSON commands."""
    xs, ys = [], []
    for c in commands:
        if not c:
            continue
        x = c.get("x", None)
        y = c.get("y", None)
        if x is None or y is None:
            continue
        xs.append(float(x))
        ys.append(float(y))
    if not xs:
        raise ValueError("No valid x/y points in JSON.")
    return min(xs), max(xs), min(ys), max(ys)

def make_boundary_rect(bound_x, bound_y, bound_w, bound_h):
    """
    Return a top-left based rectangle (x0, y0, w, h) using BOUND_ORIGIN_MODE and Y_AXIS_FLIP.
    If Y_AXIS_FLIP=True, the rectangle extends downward/upward in the opposite Y direction
    while keeping the same top edge.
    """
    mode = BOUND_ORIGIN_MODE.upper()
    if mode == "CENTER":
        x0 = bound_x - 0.5 * bound_w
        y0 = bound_y - 0.5 * bound_h
    else:  # "TL"
        x0 = bound_x
        y0 = bound_y

    # If we must flip Y, reflect the vertical extent so the "top" edge stays at y0.
    if Y_AXIS_FLIP:
        y0 = y0 - bound_h

    return x0, y0, bound_w, bound_h

def boundary_corners_top_left(x0, y0, w, h):
    """Top-left origin to 4 corners in CCW order: TL, TR, BR, BL."""
    tl = (x0,       y0)
    tr = (x0 + w,   y0)
    br = (x0 + w,   y0 + h)
    bl = (x0,       y0 + h)
    return tl, tr, br, bl

def plan_transform_autofit(commands, inner_x0, inner_y0, inner_w, inner_h):
    """
    Uniform scale s and translation (tx, ty) so that:
      [x_r, y_r]^T = s * [x_d, y_d]^T + [tx, ty]^T
    The mapped drawing fits inside the inner rectangle (top-left at inner_x0/inner_y0).
    If ALLOW_SCALING=False, uses s=1 and raises if it doesn't fit.
    """
    xmin, xmax, ymin, ymax = compute_bbox(commands)
    dw = xmax - xmin
    dh = ymax - ymin
    if dw <= 0 or dh <= 0:
        raise ValueError("Degenerate drawing bbox (zero area).")
    if inner_w <= 0 or inner_h <= 0:
        raise ValueError("Inner rectangle invalid (non-positive size).")

    if ALLOW_SCALING:
        s = min(inner_w / dw, inner_h / dh)
        if s <= 0:
            raise ValueError("Non-positive scale; inner rect too small.")
    else:
        s = 1.0
        if dw > inner_w or dh > inner_h:
            raise ValueError(
                "Drawing larger than inner boundary and scaling disabled. "
                "Enable ALLOW_SCALING or enlarge boundary."
            )

    # center inside inner rectangle
    cx_d = 0.5 * (xmin + xmax)
    cy_d = 0.5 * (ymin + ymax)
    cx_i = inner_x0 + 0.5 * inner_w
    cy_i = inner_y0 + 0.5 * inner_h

    tx = cx_i - s * cx_d
    ty = cy_i - s * cy_d

    # final bbox in robot space (sanity)
    Xmin = s * xmin + tx
    Xmax = s * xmax + tx
    Ymin = s * ymin + ty
    Ymax = s * ymax + ty

    # Allow for tiny floating-point noise
    eps = 1e-6
    ok = (inner_x0 - eps <= Xmin <= Xmax <= inner_x0 + inner_w + eps and
          inner_y0 - eps <= Ymin <= Ymax <= inner_y0 + inner_h + eps)

    if not ok:
        print("✗ Planned transform does not fit inside the inner rectangle.")
        print(f"  inner_x0={inner_x0:.6f}, inner_w={inner_w:.6f}, "
              f"Xmin={Xmin:.6f}, Xmax={Xmax:.6f}")
        print(f"  inner_y0={inner_y0:.6f}, inner_h={inner_h:.6f}, "
              f"Ymin={Ymin:.6f}, Ymax={Ymax:.6f}")
        raise RuntimeError("Planned transform does not fit inside the inner rectangle.")

    return s, tx, ty, (Xmin, Xmax, Ymin, Ymax)

def guard_workspace(bbox_robot):
    """Check if bbox is within coarse workspace limits."""
    Xmin, Xmax, Ymin, Ymax = bbox_robot
    return (WORK_MIN_X <= Xmin <= WORK_MAX_X and
            WORK_MIN_X <= Xmax <= WORK_MAX_X and
            WORK_MIN_Y <= Ymin <= WORK_MAX_Y and
            WORK_MIN_Y <= Ymax <= WORK_MAX_Y)

# ============================================================================
# TEXT TO DRAWING COMMANDS (USING OPENCV FONT)
# ============================================================================
def build_name_commands_from_text(text):
    """
    Render 'text' into an image using OpenCV's Hershey font, then convert the
    contours into MOVE / PEN_DOWN / DRAW commands in a local coordinate system.
    The coordinates are in pixel units; later we use plan_transform_autofit to
    fit them into the name box in mm.
    """
    text = text.strip()
    if not text:
        return []

    # Create a blank white image
    img_h, img_w = 200, 600
    img = np.full((img_h, img_w), 255, np.uint8)

    # Put black text
    # Adjust fontScale & thickness if you want a different style
    cv2.putText(
        img,
        text,
        org=(20, 150),  # x, y (baseline)
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=3.0,
        color=0,
        thickness=5,
        lineType=cv2.LINE_AA,
    )

    # Convert to binary (white background, black text)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours of the text
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    commands = []
    for cnt in contours:
        if len(cnt) < 10:
            continue

        pts = cnt[:, 0, :]  # shape (N,2)

        # Downsample so we don't get thousands of points per contour
        step = max(1, len(pts) // 200)
        pts = pts[::step]

        x0, y0 = float(pts[0][0]), float(pts[0][1])
        commands.append({"action": "MOVE",     "x": x0, "y": y0, "pen_state": "UP"})
        commands.append({"action": "PEN_DOWN", "x": x0, "y": y0, "pen_state": "DOWN"})
        for p in pts[1:]:
            x, y = float(p[0]), float(p[1])
            commands.append({"action": "DRAW", "x": x, "y": y, "pen_state": "DOWN"})

    return commands

# ============================================================================
# ORIENTATION PRE-PROCESSING (FIX MIRRORING)
# ============================================================================
def preprocess_commands_for_orientation(commands):
    """
    Apply MIRROR_X / MIRROR_Y to commands BEFORE planning.
    This keeps bbox, scaling and fit consistent with the final drawing.
    """
    if not (MIRROR_X or MIRROR_Y):
        return commands

    processed = []
    for c in commands:
        if not c:
            processed.append(c)
            continue
        nc = dict(c)
        if "x" in nc and nc["x"] is not None:
            x = float(nc["x"])
            if MIRROR_X:
                x = -x
            nc["x"] = x
        if "y" in nc and nc["y"] is not None:
            y = float(nc["y"])
            if MIRROR_Y:
                y = -y
            nc["y"] = y
        processed.append(nc)
    return processed

# ============================================================================
# WORKSPACE REACHABILITY CHECKING
# ============================================================================
def is_point_reachable(api, x, y, z, tolerance=5.0):
    """
    Check if a point is reachable at a specific Z height.
    This uses a non-blocking probe to test reachability without committing movement.
    """
    try:
        pose = get_current_position()
        if pose is None:
            return False

        # Mode 0 = PTPMOVJXYZMode, isQueued=0 for immediate test
        result = dType.SetPTPCmd (api, dType.PTPMode.PTPMOVJXYZMode,
                                 x, y, z, R_HEAD, 0)

        if result and result[0] >= 0:
            return True
        return False
    except Exception as e:
        print(f"Reachability check error at ({x:.1f},{y:.1f},{z:.1f}): {e}")
        return False

def validate_points_reachable(api, points, z_height, point_names=None):
    """
    Validate that all points are reachable at a specific Z height.
    """
    if point_names is None:
        point_names = [f"Point_{i}" for i in range(len(points))]

    reachability = []
    all_reachable = True

    print(f"\n{'='*70}")
    print(f"Validating reachability at Z={z_height:.2f}mm")
    print(f"{'='*70}")

    for (x, y), name in zip(points, point_names):
        reachable = is_point_reachable(api, x, y, z_height)
        reachability.append(reachable)
        status = "✓ REACHABLE" if reachable else "✗ UNREACHABLE"
        print(f"  {name:15s} ({x:7.2f}, {y:7.2f}) : {status}")

        if not reachable:
            all_reachable = False

    print(f"{'='*70}\n")
    return all_reachable, reachability

def map_workspace_at_z(api, z_height, x_range=(150, 320), y_range=(-100, 150),
                       resolution=20, verbose=False):
    """
    Map the reachable workspace at a specific Z height by testing a grid of points.
    """
    reachable = []
    unreachable = []

    print(f"\nMapping workspace at Z={z_height:.2f}mm...")
    print(f"Testing from X=[{x_range[0]}, {x_range[1]}], Y=[{y_range[0]}, {y_range[1]}]")
    print(f"Grid resolution: {resolution}mm\n")

    test_count = 0
    for x in range(int(x_range[0]), int(x_range[1]), resolution):
        for y in range(int(y_range[0]), int(y_range[1]), resolution):
            test_count += 1
            if is_point_reachable(api, x, y, z_height):
                reachable.append((x, y))
                if verbose:
                    print(f"  ✓ ({x:3d}, {y:4d})")
            else:
                unreachable.append((x, y))
                if verbose:
                    print(f"  ✗ ({x:3d}, {y:4d})")

            dType.dSleep(10)

    print(f"\nWorkspace mapping complete:")
    print(f"  Total tested: {test_count} points")
    print(f"  Reachable:    {len(reachable)} points")
    print(f"  Unreachable:  {len(unreachable)} points")
    print(f"  Coverage:     {100.0*len(reachable)/test_count:.1f}%\n")

    return reachable, unreachable

# ============================================================================
# MOTION HELPERS
# ============================================================================
def move_safe(api, x, y, velocity=None, acceleration=None):
    """Jog to (x,y) at Z_SAFE."""
    if velocity is None:
        velocity = VELOCITY
    if acceleration is None:
        acceleration = ACCELERATION

    # Update motion params if using custom values
    if velocity != VELOCITY or acceleration != ACCELERATION:
        dType.SetPTPCoordinateParams(api, velocity, acceleration, velocity, acceleration, 0)

    return dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, x, y, Z_SAFE, R_HEAD, 1)[0]

def move_draw(api, x, y, velocity=None, acceleration=None):
    """Linear to (x,y) at Z_DRAW."""
    if velocity is None:
        velocity = VELOCITY
    if acceleration is None:
        acceleration = ACCELERATION

    # Update motion params if using custom values
    if velocity != VELOCITY or acceleration != ACCELERATION:
        dType.SetPTPCoordinateParams(api, velocity, acceleration, velocity, acceleration, 0)

    return dType.SetPTPCmd (api, dType.PTPMode.PTPMOVLXYZMode, x, y, Z_DRAW, R_HEAD, 1)[0]

def cp_draw_point(api, x, y, z=Z_DRAW, velocity=None):
    """
    Enqueue one CP absolute point at drawing height.
    Returns queued index.
    """
    if velocity is None:
        velocity = VELOCITY
    return dType.SetCPCmd(api, dType.ContinuousPathMode.CPAbsoluteMode, x, y, z, velocity, 1)[0]

def lift_safe_here(api):
    """Lift current position to Z_SAFE."""
    pose = get_current_position()
    return dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, pose[0], pose[1], Z_SAFE, pose[3], 1)[0]

# ============================================================================
# PAUSE HANDLER (LIFT TO Z_SAFE, RESUME FROM SAME XY)
# ============================================================================
def handle_pause(api, cur_z):
    """
    Check keyboard for pause/resume/quit commands.
    If 'P' is pressed:
      - Finish the current queued move,
      - Lift vertically to Z_SAFE (if not already there),
      - Wait until 'R' to resume or 'Q' to abort.
    Returns the updated cur_z.
    """
    if not ENABLE_KEYBOARD_PAUSE or msvcrt is None:
        return cur_z

    if not msvcrt.kbhit():
        return cur_z

    ch = msvcrt.getch()
    # Allow immediate abort with Q even without prior pause
    if ch in (b'q', b'Q'):
        print("[ABORT REQUESTED]")
        raise KeyboardInterrupt

    if ch not in (b'p', b'P'):
        return cur_z

    print("\n[PAUSE REQUESTED] Finishing current move and lifting to Z_SAFE...")

    # Ensure Z_SAFE physically and in our state
    if cur_z != Z_SAFE:
        wait_for_command(lift_safe_here(api))
        cur_z = Z_SAFE

    print("[PAUSED] Press 'R' to resume or 'Q' to abort.")
    while True:
        if msvcrt.kbhit():
            ch2 = msvcrt.getch()
            if ch2 in (b'r', b'R'):
                print("[RESUME]\n")
                return cur_z
            elif ch2 in (b'q', b'Q'):
                print("[ABORT REQUESTED]")
                raise KeyboardInterrupt
        dType.dSleep(100)

# ============================================================================
# PREVIEW & CORNER VALIDATION
# ============================================================================
def preview_corners(api, corners):
    """Jog above each corner at Z_SAFE and pause for confirmation."""
    print("\n" + "="*70)
    print("PREVIEWING FRAME CORNERS AT Z_SAFE")
    print("="*70)

    pose = get_current_position()
    wait_for_command(dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode,
                                     pose[0], pose[1], Z_SAFE, R_HEAD, 1)[0])

    names = ("TL (Top-Left)", "TR (Top-Right)", "BR (Bottom-Right)", "BL (Bottom-Left)")
    for i, (name, (x, y)) in enumerate(zip(names, corners), 1):
        print(f"\n  [{i}/4] Moving to {name}")
        print(f"        Position: ({x:.2f}, {y:.2f}) at Z_SAFE={Z_SAFE:.2f}mm")
        wait_for_command(move_safe(api, x, y))
        print(f"        ✓ Reached. Pen should be pointing at this corner.")
        input("        Press Enter to continue to next corner...")

    print("\n" + "="*70 + "\n")

# ============================================================================
# FRAME DRAWING - SEGMENTED APPROACH
# ============================================================================
def draw_boundary_frame_segmented(api, x0, y0, w, h):
    """
    Draw the visible rectangular frame at Z_DRAW using segmented approach.
    """
    tl, tr, br, bl = boundary_corners_top_left(x0, y0, w, h)
    corners = [tl, tr, br, bl]
    corner_names = ["TL", "TR", "BR", "BL"]

    print("\n" + "="*70)
    print("DRAWING FRAME - SEGMENTED APPROACH")
    print("="*70)

    # Use reduced velocity/acceleration for boundary operations
    dType.SetPTPCoordinateParams(api, VELOCITY_BOUNDARY, ACCELERATION_BOUNDARY,
                                 VELOCITY_BOUNDARY, ACCELERATION_BOUNDARY, 0)

    # Lift to safe height initially
    pose = get_current_position()
    wait_for_command(dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode,
                                     pose[0], pose[1], Z_SAFE, R_HEAD, 1)[0])
    cur_z = Z_SAFE

    # Draw each side independently
    for i in range(4):
        start_corner = corners[i]
        end_corner = corners[(i + 1) % 4]
        start_name = corner_names[i]
        end_name = corner_names[(i + 1) % 4]

        print(f"\n  Side {i+1}/4: {start_name} → {end_name}")
        print(f"    From ({start_corner[0]:.1f}, {start_corner[1]:.1f})")
        print(f"    To   ({end_corner[0]:.1f}, {end_corner[1]:.1f})")

        # Lift to safe height
        print(f"    [1/4] Lifting to Z_SAFE...")
        wait_for_command(lift_safe_here(api))
        dType.dSleep(100)
        cur_z = Z_SAFE

        # Move to start corner at safe height
        print(f"    [2/4] Moving to start corner at Z_SAFE...")
        wait_for_command(move_safe(api, start_corner[0], start_corner[1],
                                   VELOCITY_BOUNDARY, ACCELERATION_BOUNDARY))
        dType.dSleep(100)

        # Lower to draw height
        print(f"    [3/4] Lowering to Z_DRAW...")
        wait_for_command(move_draw(api, start_corner[0], start_corner[1],
                                   VELOCITY_BOUNDARY, ACCELERATION_BOUNDARY))
        dType.dSleep(100)
        cur_z = Z_DRAW

        # Draw the side
        print(f"    [4/4] Drawing side...")
        wait_for_command(move_draw(api, end_corner[0], end_corner[1],
                                   VELOCITY_BOUNDARY, ACCELERATION_BOUNDARY))
        dType.dSleep(100)

        print(f"    ✓ Side {i+1} complete")

        # Allow user to pause between sides (lift handled in handler if needed)
        cur_z = handle_pause(api, cur_z)

    # Final lift
    print(f"\n  Lifting pen to safe height...")
    wait_for_command(lift_safe_here(api))
    cur_z = Z_SAFE

    # Restore normal motion parameters
    dType.SetPTPCoordinateParams(api, VELOCITY, ACCELERATION, VELOCITY, ACCELERATION, 0)

    print("\n" + "="*70 + "\n")


# ============================================================================
# NAME BOX + TEXT DRAWING
# ============================================================================
def draw_name_box_and_text(api, frame_x0, frame_y0, frame_w, frame_h, name_text, position):
    """
    Draw a rectangle above or below the frame and write 'name_text' inside it.
    'position' is either 'above' or 'below'.
    """
    if not name_text:
        return

    box_x0 = frame_x0
    box_h  = NAME_BOX_HEIGHT
    box_w  = frame_w

    # Decide where to place the box in robot Y coordinates
    pos = position.lower()
    if pos == "above":
        # box immediately above the top edge of the frame
        box_y0 = frame_y0 + frame_h
    else:
        # default: box immediately below the bottom edge of the frame
        box_y0 = frame_y0 - box_h

    print("\n" + "="*70)
    print(f"DRAWING NAME BOX FOR '{name_text}' ({position.upper()})")
    print("="*70)

    # 1) Draw the rectangular name box
    draw_boundary_frame_segmented(api, box_x0, box_y0, box_w, box_h)

    # 2) Build local drawing commands for the text
    text_cmds = build_name_commands_from_text(name_text.upper())
    if not text_cmds:
        print("No text commands generated; skipping name.")
        return

    # Apply same orientation mirroring as the main sketch
    text_cmds = preprocess_commands_for_orientation(text_cmds)

    # 3) Inner rectangle for the text inside the name box
    inner_x0 = box_x0 + NAME_TEXT_MARGIN
    inner_y0 = box_y0 + NAME_TEXT_MARGIN
    inner_w  = box_w  - 2.0 * NAME_TEXT_MARGIN
    inner_h  = box_h  - 2.0 * NAME_TEXT_MARGIN
    if inner_w <= 0 or inner_h <= 0:
        print("Error: Name box inner rectangle invalid; adjust NAME_TEXT_MARGIN or NAME_BOX_HEIGHT.")
        return

    # 4) Plan transform to fit text into this rectangle
    s, tx, ty, bbox_robot = plan_transform_autofit(text_cmds, inner_x0, inner_y0, inner_w, inner_h)
    print(f"Name transform: scale={s:.5f}, tx={tx:.2f}, ty={ty:.2f}")
    print(f"  Name bbox in robot space: X=[{bbox_robot[0]:.2f}, {bbox_robot[1]:.2f}], "
          f"Y=[{bbox_robot[2]:.2f}, {bbox_robot[3]:.2f}]")

    if not guard_workspace(bbox_robot):
        print("⚠ Name bbox outside coarse workspace limits; skipping name drawing.")
        return

    # 5) Draw the text using existing sketch drawing pipeline
    draw_sketch(api, text_cmds, s, tx, ty)




# ============================================================================
# SKETCH DRAWING
# ============================================================================
def draw_sketch(api, commands, s, tx, ty):
    """Execute JSON drawing using the planned transform (fit inside inner frame)."""
    print("\n" + "="*70)
    print("DRAWING SKETCH")
    print("="*70)
    print(f"Transform: scale={s:.5f}, tx={tx:.2f}, ty={ty:.2f}")
    if ENABLE_KEYBOARD_PAUSE and msvcrt is not None:
        print("Controls: 'P' = pause (lift to Z_SAFE), 'R' = resume, 'Q' = abort.\n")
    else:
        print()

    pose = get_current_position()
    wait_for_command(
        dType.SetPTPCmd(
            api, dType.PTPMode.PTPMOVJXYZMode,
            pose[0], pose[1], Z_SAFE, R_HEAD, 1
        )[0]
    )

    cur_x = pose[0]
    cur_y = pose[1]
    cur_z = Z_SAFE
    in_stroke = False
    last_cp_idx = None

    total = len(commands)

    # --- timing ---
    total_start_time = time.time()      # total drawing time
    batch_start_time = total_start_time # per-200-command delta
    # ---------------

    for i, cmd in enumerate(commands, 1):
        action = cmd.get("action")
        x_rel = cmd.get("x")
        y_rel = cmd.get("y")
        pen_state = cmd.get("pen_state")

        if x_rel is not None and y_rel is not None:
            x_abs = float(s) * float(x_rel) + float(tx)
            y_abs = float(s) * float(y_rel) + float(ty)
        else:
            x_abs, y_abs = cur_x, cur_y

        if action == "MOVE" and pen_state == "UP":
            if in_stroke and last_cp_idx is not None:
                wait_for_command(last_cp_idx)
                in_stroke = False
                last_cp_idx = None
            if cur_z != Z_SAFE:
                wait_for_command(lift_safe_here(api))
                cur_z = Z_SAFE
            wait_for_command(move_safe(api, x_abs, y_abs))

        elif action == "PEN_DOWN":
            if in_stroke and last_cp_idx is not None:
                wait_for_command(last_cp_idx)
                in_stroke = False
                last_cp_idx = None
            if cur_x != x_abs or cur_y != y_abs:
                if cur_z != Z_SAFE:
                    wait_for_command(lift_safe_here(api))
                    cur_z = Z_SAFE
                wait_for_command(move_safe(api, x_abs, y_abs))
            wait_for_command(move_draw(api, x_abs, y_abs))
            cur_z = Z_DRAW

            # Start stroke: enqueue first CP point at current location
            last_cp_idx = cp_draw_point(api, x_abs, y_abs, Z_DRAW, velocity=VELOCITY)
            in_stroke = True


        elif action == "DRAW" and pen_state == "DOWN":
            if cur_z != Z_DRAW:
                wait_for_command(move_draw(api, x_abs, y_abs))
                cur_z = Z_DRAW
                last_cp_idx = cp_draw_point(api, x_abs, y_abs, Z_DRAW, velocity=VELOCITY)
                in_stroke = True
            else:
                last_cp_idx = cp_draw_point(api, x_abs, y_abs, Z_DRAW, velocity=VELOCITY)
                in_stroke = True

        elif action == "MOVE" and pen_state == "DOWN":
            # treat as drawing move with pen down
            if cur_z != Z_DRAW:
                wait_for_command(move_draw(api, x_abs, y_abs))
                cur_z = Z_DRAW
                last_cp_idx = cp_draw_point(api, x_abs, y_abs, Z_DRAW, velocity=VELOCITY)
                in_stroke = True
            else:
                last_cp_idx = cp_draw_point(api, x_abs, y_abs, Z_DRAW, velocity=VELOCITY)
                in_stroke = True

            cur_z = Z_DRAW

        cur_x, cur_y = x_abs, y_abs

        # progress log every 200 commands
        if i % 200 == 0:
            elapsed = time.time() - batch_start_time
            print(f"  {i}/{total}: {action:12s} → ({x_abs:.2f},{y_abs:.2f}) | Δt={elapsed:.2f}s")
            batch_start_time = time.time()

        # allow user to pause at any time; will lift to Z_SAFE and resume from same XY
        cur_z = handle_pause(api, cur_z)

    if in_stroke and last_cp_idx is not None:
        wait_for_command(last_cp_idx)

    wait_for_command(lift_safe_here(api))

    total_elapsed = time.time() - total_start_time
    print(f"\n✓ Sketch complete ({total} commands executed)")
    print(f"  Total drawing time: {total_elapsed:.2f} s ({total_elapsed/60.0:.2f} min)\n")

# ============================================================================
# MAIN PROGRAM
# ============================================================================
def main():
    api = None
    try:
        if not os.path.exists(SKETCH_FILE_PATH):
            print(f"Error: sketch file not found: {SKETCH_FILE_PATH}")
            return

        api = connect_to_dobot()
        if not api:
            return

        setup_dobot(api)

        # Load JSON
        print(f"\nLoading sketch from: {SKETCH_FILE_PATH}")
        with open(SKETCH_FILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw_commands = data.get("commands", [])
        if not raw_commands:
            print("Error: 'commands' empty in JSON.")
            return
        print(f"✓ Loaded {len(raw_commands)} commands from sketch")

        # Apply orientation fix BEFORE planning
        commands = preprocess_commands_for_orientation(raw_commands)

        # ====================================================================
        # BUILD FRAME RECTANGLE
        # ====================================================================
        x0, y0, w, h = make_boundary_rect(BOUND_X, BOUND_Y, BOUND_W, BOUND_H)
        tl, tr, br, bl = boundary_corners_top_left(x0, y0, w, h)

        print(f"\n{'='*70}")
        print("FRAME CONFIGURATION")
        print(f"{'='*70}")
        print(f"Frame mode: {BOUND_ORIGIN_MODE.upper()}")
        print(f"Base position: ({BOUND_X:.1f}, {BOUND_Y:.1f})")
        print(f"Frame size: {BOUND_W:.1f}mm × {BOUND_H:.1f}mm")
        print(f"\nCorners (robot coordinates in mm):")
        print(f"  TL (Top-Left):     ({tl[0]:7.2f}, {tl[1]:7.2f})")
        print(f"  TR (Top-Right):    ({tr[0]:7.2f}, {tr[1]:7.2f})")
        print(f"  BR (Bottom-Right): ({br[0]:7.2f}, {br[1]:7.2f})")
        print(f"  BL (Bottom-Left):  ({bl[0]:7.2f}, {bl[1]:7.2f})")
        print(f"{'='*70}\n")

        name_text = ""
        name_position = "below"
        if ENABLE_NAME_BOX:
            name_text = input("Enter name to write (leave blank for none): ").strip()
            if name_text:
                pos_in = input("Where should the name be? (above/below) [below]: ").strip().lower()
                if pos_in in ("above", "a", "top"):
                    name_position = "above"
                else:
                    name_position = "below"
            print()
        # ====================================================================
        # OPTIONAL: PREVIEW CORNERS AT Z_SAFE
        # ====================================================================
        if PREVIEW_CORNERS:
            preview_corners(api, (tl, tr, br, bl))

        # ====================================================================
        # VALIDATE FRAME CORNERS ARE REACHABLE AT Z_SAFE
        # ====================================================================
        corners_list = [tl, tr, br, bl]
        corner_names = ["TL", "TR", "BR", "BL"]

        all_reachable, reachability = validate_points_reachable(api, corners_list,
                                                                 Z_SAFE, corner_names)

        if not all_reachable:
            unreachable_corners = [name for name, reach in zip(corner_names, reachability)
                                   if not reach]
            print(f"✗ ERROR: The following corners are unreachable at Z_SAFE={Z_SAFE}:")
            for corner in unreachable_corners:
                print(f"  - {corner}")
            print(f"\nSOLUTIONS:")
            print(f"  1. Raise Z_DRAW from {Z_DRAW:.1f}mm to {Z_DRAW + 5:.1f}mm")
            print(f"  2. Move frame closer: reduce BOUND_Y from {BOUND_Y:.1f} to {BOUND_Y - 10:.1f}")
            print(f"  3. Reduce frame size: decrease BOUND_W or BOUND_H")
            print(f"\nAborting drawing to prevent robot collision.\n")
            return

        print(f"✓ All frame corners validated as reachable at Z_SAFE={Z_SAFE}mm\n")

        # ====================================================================
        # OPTIONAL: WORKSPACE DIAGNOSTICS
        # ====================================================================
        if ENABLE_WORKSPACE_DIAGNOSTICS:
            print("\n" + "="*70)
            print("WORKSPACE DIAGNOSTICS")
            print("="*70)

            print("\nTesting workspace at Z_SAFE:")
            safe_reach, _ = map_workspace_at_z(api, Z_SAFE, resolution=30)

            print("\nTesting workspace at Z_DRAW:")
            draw_reach, _ = map_workspace_at_z(api, Z_DRAW, resolution=30)

            print(f"Workspace shrinkage: {100.0*(len(safe_reach)-len(draw_reach))/len(safe_reach):.1f}%")
            print()

        # ====================================================================
        # DRAW FRAME
        # ====================================================================
        input("Press Enter to start drawing frame...")
        print()
        draw_boundary_frame_segmented(api, x0, y0, w, h)


        # Draw name box + text under the frame, if requested
        # Draw name box + text, if requested
        if ENABLE_NAME_BOX and name_text:
            draw_name_box_and_text(api, x0, y0, w, h, name_text, name_position)



        # ====================================================================
        # SKETCH TRANSFORM & DRAWING
        # ====================================================================
        inner_x0 = x0 + INNER_MARGIN
        inner_y0 = y0 + INNER_MARGIN
        inner_w  = w - 2.0 * INNER_MARGIN
        inner_h  = h - 2.0 * INNER_MARGIN
        if inner_w <= 0 or inner_h <= 0:
            print("Error: Inner rectangle invalid; adjust INNER_MARGIN or frame size.")
            return

        print("Planning transform for sketch...")
        s, tx, ty, bbox_robot = plan_transform_autofit(commands, inner_x0, inner_y0, inner_w, inner_h)
        print(f"✓ Transform planned: scale={s:.5f}, tx={tx:.2f}, ty={ty:.2f}")
        print(f"  Sketch bbox in robot space:")
        print(f"    X: [{bbox_robot[0]:.2f}, {bbox_robot[1]:.2f}]")
        print(f"    Y: [{bbox_robot[2]:.2f}, {bbox_robot[3]:.2f}]\n")

        if not guard_workspace(bbox_robot):
            print("⚠ Warning: planned sketch bbox outside coarse workspace limits.")
            print("  Adjust frame position or sketch file.")
            response = input("  Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return

        input("Press Enter to start drawing sketch inside frame...")
        draw_sketch(api, commands, s, tx, ty)

        print("✓ All drawing complete!")

    except KeyboardInterrupt:
        print("\n✗ Cancelled by user.")
    except Exception as e:
        print(f"✗ Critical error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if api:
            try:
                wait_for_command(lift_safe_here(api))
            except Exception:
                pass
            dType.SetQueuedCmdStopExec(api)
            dType.DisconnectDobot(api)
            print("✓ Disconnected from robot.")

if __name__ == "__main__":
    main()
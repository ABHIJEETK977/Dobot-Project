# main_out.py
import os
import time
import threading
from pathlib import Path
from dataclasses import dataclass
import subprocess, shutil
from queue import Queue, Empty
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

import signal
import numpy as np

# --- Hailo / pipeline pieces ---
from utils.detector_helper import DetectionHelper
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.detection.detection_pipeline import GStreamerDetectionApp

# --- DoBot & IO ---
from utils.dobot_lib.DoBotArm import DoBotArm as Dbt
from utils.dobot_lib.DobotDllType import SetIOMultiplexing, GetIODI  # direct DI access

# ==============================
# Tunables (edit as needed)
# ==============================
HOME = (234.6, 13.5, 116.3)
CONVEYOR_MM_S = 45.0          # >0 forward, 0 stop
stepper_index = 1             # 0 or 1 for stepper 1 or 2
SENSOR_PIN = 17               # DI15 or DI17 for port 1 or 6
COOLDOWN_S = 0.75             # min time between two pick cycles (safety)
staging_pos = (248.3, 13.5, 34.6)
tray_approach_pos = [20, -267.9, 63.4]

# Label → bin coordinates
DEFAULT_DROP_Z = 45
Drop_side = 'Left' # 'Right' or 'Left' (relative to robot front); affects Y sign of all drop coords

Drop_coord = [(-51,   -188.3),   # red apple
              (14,    -185.2),   # strawberry
              (73.8,  -180.1),   # olive
              (77.4,  -238.7),   # green apple
              (21.6,  -247.3),   # pomegranate
              (-41.9, -252.3),   # pumpkin
              (-43.5, -309.6),   # carrot
              (19.3,  -307.9),   # eggplant
              (80.9,  -299.9)]   # corn

if Drop_side == 'Right':
    tray_approach_pos = tray_approach_pos
    Drop_coord = Drop_coord
else:
    tray_approach_pos[1] = -tray_approach_pos[1]
    Drop_coord = [(x, -y) for (x, y) in Drop_coord]

DROP_ZONES = {
    "red apple":    (Drop_coord[0][0],   Drop_coord[0][1], DEFAULT_DROP_Z),
    "strawberry":   (Drop_coord[1][0],   Drop_coord[1][1], DEFAULT_DROP_Z),
    "olive":        (Drop_coord[2][0],   Drop_coord[2][1], DEFAULT_DROP_Z),
    "green apple":  (Drop_coord[3][0],   Drop_coord[3][1], DEFAULT_DROP_Z),
    "pomegranate":  (Drop_coord[4][0],   Drop_coord[4][1], DEFAULT_DROP_Z),
    "pumpkin":      (Drop_coord[5][0],   Drop_coord[5][1], DEFAULT_DROP_Z),
    "carrot":       (Drop_coord[6][0],   Drop_coord[6][1], DEFAULT_DROP_Z),
    "eggplant":     (Drop_coord[7][0],   Drop_coord[7][1], DEFAULT_DROP_Z),
    "corn":        (Drop_coord[8][0],   Drop_coord[8][1], DEFAULT_DROP_Z),
}
ALLOW_LABELS = {
    "eggplant", "carrot", "green apple", "olive",
    "pomegranate", "pumpkin", "red apple", "corn", "strawberry",
}
SHUTDOWN = threading.Event()

Default_Pick_Z = 20.0

PICK_Z_BY_LABEL = {
    "red apple":   Default_Pick_Z,
    "strawberry":  Default_Pick_Z - 5.0,
    "olive":       Default_Pick_Z - 5.0,
    "green apple": Default_Pick_Z,
    "pomegranate": Default_Pick_Z,
    "pumpkin":     Default_Pick_Z - 5.0,
    "carrot":      Default_Pick_Z - 3.0,
    "eggplant": Default_Pick_Z + 5.0,
    "corn":   Default_Pick_Z - 10.0,
}
CONF_THRESH = 0.4
MIN_BOX_AREA = 0

# =========================================================
# Homography & mapping
# =========================================================
H_PATH = Path("resources/pixel_to_table_H.npy")   # must match your calibrator's SAVE_FILE
_H = None
_H_lock = threading.Lock()

# Action re-entry guard to ensure one pick cycle at a time
ACTION_LOCK = threading.Lock()

def _load_homography_once():
    """Load the 3x3 pixel->world homography once, thread-safe."""
    global _H
    if _H is not None:
        return _H
    with _H_lock:
        if _H is not None:
            return _H
        if not H_PATH.exists():
            raise FileNotFoundError(
                f"Homography file not found at {H_PATH}. "
                f"Run your calibration script to create it."
            )
        H = np.load(str(H_PATH)).astype(np.float64)
        if H.shape != (3, 3):
            raise ValueError(f"Invalid homography shape {H.shape}; expected (3,3)")
        _H = H
    return _H

def _wait_until_pose(bot: Dbt, target_xyz, tol=(1.5, 1.5, 1.5), timeout_s=20.0, poll_s=0.05):
    """Block until robot (x,y,z) is within tol of target_xyz or timeout."""
    tx, ty, tz = float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2])
    t0 = time.time()
    while True:
        try:
            pose = bot.cur_location()
            if pose and len(pose) >= 3:
                x, y, z = float(pose[0]), float(pose[1]), float(pose[2])
                if abs(x - tx) <= tol[0] and abs(y - ty) <= tol[1] and abs(z - tz) <= tol[2]:
                    return True
        except Exception:
            pass
        if time.time() - t0 > timeout_s:
            print(f"[robot][WARN] wait_until_pose timed out after {timeout_s}s; "
                  f"last pose={pose if 'pose' in locals() else None}, target={target_xyz}")
            return False
        if SHUTDOWN.wait(poll_s):
            return False

def _apply_homography(u: int, v: int) -> tuple[float, float]:
    """Apply H to pixel (u,v) → world (X,Y) in robot table coordinates (mm)."""
    H = _load_homography_once()
    uv1 = np.array([float(u), float(v), 1.0], dtype=np.float64)
    XYw = H @ uv1
    s = XYw[2]
    if abs(s) < 1e-9:
        raise ZeroDivisionError("Homogeneous scale ~0 while applying homography; check calibration pairs.")
    return float(XYw[0] / s), float(XYw[1] / s)

def uv_to_robot_xyz(label: str, u: int, v: int) -> tuple[float, float, float]:
    """Map pixel center (u,v) -> robot (X,Y,Z) using calibrated planar homography."""
    try:
        X, Y = _apply_homography(u, v)
        Z = PICK_Z_BY_LABEL.get(label)
        print(f"[map] '{label}' (u={u}, v={v}) -> (X={X:.1f}, Y={Y:.1f}, Z={Z:.1f}) [H]")
        return X, Y, Z
    except Exception as e:
        print(f"[calib][ERROR] {e}. Falling back to rough mapping.")
        X = 100.0 + (u / 8.0)
        Y =  60.0 + (v / 8.0)
        Z = PICK_Z_BY_LABEL.get(label)
        return X, Y, Z

CLEANUP_STARTED = threading.Event()

def _best_effort_stop(bot: Dbt | None):
    """
    Stop conveyor immediately.
    Do NOT call moveHome() during Ctrl-C shutdown unless you know the robot is idle.
    """
    if CLEANUP_STARTED.is_set():
        return
    CLEANUP_STARTED.set()

    try:
        if bot:
            print("[shutdown] stopping conveyor...")
            try:
                bot.set_conveyor_speed(0.0, stepper_index)
                time.sleep(0.2)
            except Exception as e:
                print(f"[shutdown][WARN] conveyor stop failed: {e}")

            # Only home if robot is definitely idle.
            if ACTION_LOCK.acquire(blocking=False):
                try:
                    print("[shutdown] robot appears idle; moving home...")
                    bot.moveHome()
                except Exception as e:
                    print(f"[shutdown][WARN] moveHome failed: {e}")
                finally:
                    ACTION_LOCK.release()
            else:
                print("[shutdown] robot busy; skipping moveHome during shutdown")
    except Exception as e:
        print(f"[shutdown][WARN] {e}")

class TTSSpeaker:
    """
    Non-blocking TTS queue with clear diagnostics.
    Prefers Piper (if installed + PIPER_VOICE exists), else pyttsx3, else logs.
    Set PIPER_VOICE to your .onnx voice file and optionally PIPER_OUTPUT_DEVICE_CMD
    (default: 'aplay -q -f S16_LE -r 22050 -t wav -' or 'paplay -').
    """
    def __init__(self):
        self._q = Queue(maxsize=32)
        self._stop = False

        self._piper_path = shutil.which("piper")
        self._piper_voice = os.environ.get("PIPER_VOICE")
        self._piper_out = os.environ.get("PIPER_OUTPUT_DEVICE_CMD", "aplay -q -f S16_LE -r 22050 -t wav -")

        # Parse output command once (no bash)
        self._out_cmd = None
        try:
            # split simple commandline into argv; support basic tokens (no pipes here)
            self._out_cmd = self._piper_out.split()
        except Exception:
            self._out_cmd = None

        self._mode = "none"
        self._pyttsx3 = None

        if self._piper_path and self._piper_voice and os.path.exists(self._piper_voice) and self._out_cmd:
            self._mode = "piper"
        else:
            # Try pyttsx3
            try:
                import pyttsx3  # noqa: F401
                import pyttsx3 as _py
                self._pyttsx3 = _py.init()
                self._mode = "pyttsx3"
            except Exception:
                self._mode = "none"

        self._print_diagnostics()

        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def _print_diagnostics(self):
        print("[tts] init…")
        print(f"[tts] mode: {self._mode}")
        print(f"[tts] piper: {self._piper_path or 'not found'}")
        print(f"[tts] voice: {self._piper_voice or 'not set'}")
        if self._piper_voice and not os.path.exists(self._piper_voice):
            print(f"[tts][WARN] voice file not found at: {self._piper_voice}")
        print(f"[tts] output cmd: {self._piper_out!r} ({'ok' if self._out_cmd else 'invalid'})")
        if self._mode == "none":
            print("[tts][WARN] No working TTS backend. "
                  "Install Piper (apt install piper alsa-utils) and set PIPER_VOICE, "
                  "or install pyttsx3 + espeak.")

    def speak(self, text: str):
        if not text:
            return
        try:
            self._q.put_nowait(text)
        except Exception:
            try:
                _ = self._q.get_nowait()
            except Empty:
                pass
            try:
                self._q.put_nowait(text)
            except Exception:
                pass

    def stop(self):
        self._stop = True

    def _loop(self):
        while not self._stop:
            try:
                text = self._q.get(timeout=0.2)
            except Empty:
                continue
            try:
                self._speak_once(text)
            except Exception as e:
                print(f"[tts][ERROR] {e}")
            finally:
                self._q.task_done()

    def _speak_once(self, text: str):
        if self._mode == "piper":
            self._speak_piper(text)
            return
        if self._mode == "pyttsx3":
            self._pyttsx3.say(text)
            self._pyttsx3.runAndWait()
            return
        print(f"[tts] {text}")  # final fallback

    def _speak_piper(self, text: str):
        # piper: stdin=text -> stdout=wav -> pipe to output command (aplay or paplay)
        # Spawn piper
        piper = subprocess.Popen(
            [self._piper_path, "-m", self._piper_voice, "-f", "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )
        # Spawn output
        sink = subprocess.Popen(
            self._out_cmd,
            stdin=piper.stdout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        try:
            # Feed text to piper
            piper.stdin.write((text + "\n").encode("utf-8"))
            piper.stdin.flush()
            piper.stdin.close()
        except Exception:
            pass
        # Wait briefly; don't hang forever if audio sink is missing
        try:
            sink.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                sink.kill()
            except Exception:
                pass
        try:
            piper.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                piper.kill()
            except Exception:
                pass
# =========================================================
# Pick task & shared state
# =========================================================
@dataclass
class PickTask:
    label: str
    x_pick: float
    y_pick: float
    z_pick: float
    x_drop: float
    y_drop: float
    z_drop: float

class Shared:
    def __init__(self, bot: Dbt, tts: TTSSpeaker | None = None):
        self.object_ready = threading.Event()   # set by sensor thread when belt is stopped and object present
        self.pick_done    = threading.Event()   # set by worker when pick/place finished
        self.last_pick_ts = 0.0
        self.bot = bot
        self.pick_q: Queue = Queue(maxsize=4) 
        self.tts = tts or TTSSpeaker()          # non-blocking enqueue in callback
        print(self.tts)
# =========================================================
# Detection app & callback (runs permanently; NON-BLOCKING)
# =========================================================
class UserState(app_callback_class):
    def __init__(self, shared: Shared, use_frame=False):
        super().__init__()
        self.shared = shared
        self.use_frame = use_frame
        self.det_helper = DetectionHelper()

def app_callback(pad, info, state: UserState):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    state.increment()
    raw_dets = state.det_helper.parse(pad, buffer)

    # filter by allowlist, confidence, area
    dets = raw_dets
    if ALLOW_LABELS is not None:
        dets = [d for d in dets if d.get("label") in ALLOW_LABELS]
    dets = [d for d in dets if float(d.get("score", 0.0)) >= float(CONF_THRESH)]
    if MIN_BOX_AREA is not None:
        dets = [d for d in dets if (d.get("area") is None or float(d["area"]) >= float(MIN_BOX_AREA))]
    dets.sort(key=lambda d: int(d.get("u", 0)), reverse=True)

    if state.shared.object_ready.is_set():
        if dets:
            now = time.time()
            if now - state.shared.last_pick_ts >= COOLDOWN_S:
                d0 = dets[0]
                label, u, v = d0["label"], int(d0["u"]), int(d0["v"])
                print(f"[det] {label} ({d0.get('score',0):.2f}) at (u={u}, v={v}) while object_ready")

                # NEW: announce detection intent (non-blocking)
                try:
                    state.shared.tts.speak(f"{label} Detected, Picking the {label}")
                except Exception:
                    pass

                x_pick, y_pick, z_pick = uv_to_robot_xyz(label, u, v)

                if label in DROP_ZONES:
                    x_drop, y_drop, z_drop = DROP_ZONES[label]
                    task = PickTask(label, x_pick, y_pick, z_pick, x_drop, y_drop, z_drop)
                    try:
                        state.shared.pick_q.put_nowait(task)
                        state.shared.last_pick_ts = now
                        state.shared.object_ready.clear()
                    except Exception as e:
                        print(f"[queue][WARN] couldn’t enqueue pick task: {e}")
                        state.shared.object_ready.clear()
                        state.shared.pick_done.set()
                else:
                    print(f"[drop][WARN] No drop zone for '{label}'")
                    state.shared.object_ready.clear()
                    state.shared.pick_done.set()
        else:
            seen_labels = {d.get("label", "?") for d in raw_dets} if raw_dets else set()
            if seen_labels:
                msg = (f"Object not detected. Please remove it. "
                       f"(Saw: {', '.join(sorted(seen_labels))}; allowed: {', '.join(sorted(ALLOW_LABELS))})")
            else:
                msg = "Object not detected. Please remove it."
            print(msg)
            state.shared.object_ready.clear()
            state.shared.pick_done.set()

    return Gst.PadProbeReturn.OK
# =========================================================
# Worker: executes robot motions OFF the streaming thread
# =========================================================
def  pick_worker(shared: Shared):
    while not SHUTDOWN.is_set():
        try:
            task = shared.pick_q.get(timeout=0.1)
        except Empty:
            continue

        try:
            if SHUTDOWN.is_set():
                print("[pick] shutdown set before executing task; skipping")
                continue

            with ACTION_LOCK:
                if SHUTDOWN.is_set():
                    print("[pick] shutdown set after acquiring lock; skipping")
                    continue

                print(f"[pick] {task.label} -> "
                      f"pick({task.x_pick:.1f},{task.y_pick:.1f},{task.z_pick:.1f}) "
                      f"drop({task.x_drop:.1f},{task.y_drop:.1f},{task.z_drop:.1f})")

                shared.bot.pick_and_place_tray_loc(
                    task.x_pick, task.y_pick, task.z_pick,
                    task.x_drop, task.y_drop, task.z_drop,
                    staging_pos,
                    tray_approach_pos,
                )

                shared.bot.moveHome()
                _wait_until_pose(shared.bot, HOME[:3], tol=(1.5,1.5,1.5), timeout_s=10.0)

                try:
                    shared.tts.speak(f"Okay, Done with {task.label}")
                except Exception:
                    pass

        except Exception as e:
            print(f"[robot][ERROR] {e}")
            try:
                shared.tts.speak("Pick and place failed")
            except Exception:
                pass
        finally:
            shared.pick_done.set()
            shared.pick_q.task_done()

# =========================================================
# Thread A: Conveyor + Sensor loop
# =========================================================
def belt_and_sensor_loop(shared: Shared, *, speed_mm_s: float, sensor_pin: int):
    api = shared.bot.api
    SetIOMultiplexing(api, sensor_pin, 3, 0)

    print(f"[conv] start {speed_mm_s} mm/s")
    shared.bot.set_conveyor_speed(speed_mm_s, stepper_index)

    try:
        while not SHUTDOWN.is_set():
            # Check shutdown first before any blocking operations
            if SHUTDOWN.is_set():
                break

            try:
                di = GetIODI(api, sensor_pin)
                val = int(di[0]) if isinstance(di, (tuple, list)) else int(di)
            except Exception as e:
                print(f"[sensor][WARN] GetIODI failed: {e}")
                if SHUTDOWN.wait(0.05):
                    break
                continue

            if val == 0 and not shared.object_ready.is_set():
                print("[sensor] OBJECT DETECTED -> stopping belt")
                shared.bot.set_conveyor_speed(0.0,stepper_index)
                if SHUTDOWN.wait(0.7):
                    break

                shared.pick_done.clear()
                shared.object_ready.set()

                # Wait with timeout and check SHUTDOWN periodically
                timeout = 30.0  # max wait time
                start = time.time()
                while not shared.pick_done.is_set():
                    if SHUTDOWN.is_set():
                        print("[sensor] Shutdown during pick wait")
                        break
                    if time.time() - start > timeout:
                        print("[sensor] Pick timeout")
                        break
                    if SHUTDOWN.wait(0.1):
                        break

                if SHUTDOWN.is_set():
                    break

                print("[sensor] restarting belt")
                shared.bot.set_conveyor_speed(speed_mm_s, stepper_index)

            if SHUTDOWN.wait(0.1):
                break
    finally:
        print("[conv] shutdown → stopping conveyor")
        try:
            shared.bot.set_conveyor_speed(0.0, stepper_index)
            time.sleep(0.2)  # ensure command is processed
            print("[conv] conveyor stopped successfully")
        except Exception as e:
            print(f"[conv] Error stopping conveyor: {e}")



# =========================================================
# Main: start threads (no preview window)
# =========================================================
def main():
    env_file = Path(__file__).with_name(".env")
    if env_file.exists():
        os.environ["HAILO_ENV_FILE"] = str(env_file)

    print("[robot] connecting …")
    bot = Dbt(HOME[0], HOME[1], HOME[2])
    bot.moveHome()

    tts = TTSSpeaker()  # <-- NEW
    tts.speak("Hello, system is starting up")
    shutdown_count = {"n": 0}

    def _shutdown_request(reason: str):
        if not SHUTDOWN.is_set():
            print(f"[shutdown] requested by {reason}")
            SHUTDOWN.set()

    def _safe_shutdown_local():
        _best_effort_stop(bot)
        try:
            tts.stop()
        except Exception:
            pass

    def _sig_handler(signum, frame):
        shutdown_count["n"] += 1
        if shutdown_count["n"] == 1:
            print(f"[signal] received {signum}; EMERGENCY STOP")

            try:
                bot.set_conveyor_speed(0.0, stepper_index)
            except Exception as e:
                print(f"[signal] conveyor stop failed: {e}")

            SHUTDOWN.set()
        else:
            print("[signal] second Ctrl-C -> forcing process exit")
            os._exit(130)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _sig_handler)
        except Exception:
            pass

    shared = Shared(bot, tts=tts) 

    # Detection app (runs continuously; no preview)
    state = UserState(shared=shared, use_frame=False)
    app = GStreamerDetectionApp(app_callback, state)

    app = GStreamerDetectionApp(app_callback, state)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _sig_handler)

    t_detect = threading.Thread(target=app.run, daemon=True)
    t_detect.start()

    # Picker worker (executes robot motions off the streaming thread)
    t_picker = threading.Thread(target=pick_worker, args=(shared,), daemon=True)
    t_picker.start()

    # Belt + sensor loop
    t_belt = threading.Thread(
        target=belt_and_sensor_loop,
        args=(shared,),
        kwargs=dict(speed_mm_s=CONVEYOR_MM_S, sensor_pin=SENSOR_PIN),
        daemon=False  # <-- CHANGED: non-daemon so it blocks exit
    )
    t_belt.start()

    print("[INFO] Running. Ctrl-C to quit.")
    try:
        while not SHUTDOWN.wait(0.1):
            pass
    except KeyboardInterrupt:
        print("[main] KeyboardInterrupt - stopping system...")
        SHUTDOWN.set()
    finally:
        print("[main] Shutting down threads...")
        SHUTDOWN.set()

        # Give threads time to see SHUTDOWN and clean up
        print("[main] Waiting for belt thread...")
        t_belt.join(timeout=3.0)

        # Force stop conveyor if thread didn't finish
        print("[main] Ensuring conveyor is stopped...")
        try:
            bot.set_conveyor_speed(0.0, stepper_index)
            time.sleep(0.2)
        except Exception as e:
            print(f"[main] Error stopping conveyor: {e}")

        print("[main] Stopping GStreamer pipeline...")

        try:
            if app.pipeline:
                app.pipeline.set_state(Gst.State.NULL)
            if app.loop:
                app.loop.quit()
        except Exception as e:
            print(f"[main] GStreamer stop error: {e}")
        _safe_shutdown_local()
        print("[main] Shutdown complete.")


if __name__ == "__main__":
    main()

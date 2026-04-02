"""
Voice-Controlled Fruit Sorting with Hailo + USB Camera + DoBot Arm

Purpose
-------
This script runs a real-time fruit detection pipeline on a live camera feed using a Hailo
accelerator (HEF model). It optionally enables voice commands (Vosk) and text-to-speech
feedback (pyttsx3), and can control a DoBot robotic arm to pick detected fruit and place
it into class-specific drop locations (e.g., tray bins).

At a high level, the program:
1) Opens the camera stream (e.g., /dev/video0).
2) Runs the Hailo GStreamer detection app using the provided .hef and label map (.json).
3) Receives detection callbacks (bounding boxes + labels + confidence).
4) Chooses a target fruit (based on your logic in the callback/app classes).
5) (Optional) Listens for voice commands to start/stop/confirm actions.
6) (Optional) Commands the DoBot arm to pick and place fruit into the correct bin.
7) (Optional) Draws an on-screen overlay showing detections/targets/drop hints.

How to Run
----------
1) Activate the venv that contains Hailo + OpenCV + Vosk dependencies:
   source /home/dobot/Documents/Dobot/Ishan/better/venv_hailo_rpi_examples/bin/activate

2) Run the script with:
   python working_vcv.py \
     --hef-path resources/fruit_dobot_v1.hef \
     --input /dev/video0 \
     --labels-json resources/fruit_dobot_v1.json

Key CLI Inputs
--------------
--hef-path     Path to the compiled Hailo HEF model file.
--input        Camera device path (/dev/video0) or a video file path depending on your app.
--labels-json  JSON label map used to translate model class IDs -> human-readable names.

Notes
-----
- Robot control only works if `dobot_lib.DoBotArm` is installed and accessible.
- The Hailo detection loop is driven by GStreamer; detections are typically delivered via
  callbacks implemented in `hailo_updated_classes.py`.
"""

from pathlib import Path
import os
import sys
import time
import threading
import queue
import json

# --- Audio / Voice
import pyaudio #microphone input stream
from vosk import Model, KaldiRecognizer #offline speech recognition
import pyttsx3 #python text to speech
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
# --- Vision / Math / GStreamer
import numpy as np #numeric operations (coords, scaling, filtering)
import cv2   #drawing overlays, frame conversions, optional visualization
import gi    #GStreamer pipeline control + main loop integration

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# --- Hailo app imports (provided by your environment)
import hailo
from hailo_apps.hailo_app_python.core.common.buffer_utils import (
    get_caps_from_pad, get_numpy_from_buffer
)
# Keep your updated classes from the base file
from hailo_updated_classes import app_callback_class, GStreamerDetectionApp

# --- Robot
try:
    import dobot_lib.DoBotArm as Dbt
    ROBOT_AVAILABLE = True
except ImportError:
    print("Warning: DoBotArm module not found. Robot functionality disabled.")
    ROBOT_AVAILABLE = False

# ======================================================================
# Feature toggles
# ======================================================================
USE_TRAY_PATHS_IF_AVAILABLE = True   # prefer tray path when label has a custom drop
DRAW_OVERLAY = True                  # draw target + drop hints on frames
ENABLE_GST_MON = False               # optional mic level monitor via GStreamer

# ======================================================================
# TTS helper (thread-safe queue)
# ======================================================================

class TTSWorker:
    """Human-like TTS with pluggable backends.
    Prefers Piper (local neural TTS) if installed; falls back to pyttsx3.
    """
    def __init__(self):
        import shutil, subprocess, os, queue, threading
        self._q = queue.Queue()
        self._stop = threading.Event()
        self._backend = None

        # Try Piper first (best naturalness offline)
        try:
            self._backend = self._try_init_piper()
        except Exception as e:
            print("[TTS] Piper unavailable:", e)
            self._backend = None

        # Fallback to pyttsx3
        if self._backend is None:
            try:
                self._backend = self._init_pyttsx3()
            except Exception as e:
                print("[TTS] pyttsx3 unavailable:", e)
                self._backend = None

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # ---------------- Backend selection ----------------
    def _try_init_piper(self):
        import shutil, os
        piper_bin = shutil.which("piper")
        aplay_bin = shutil.which("aplay")
        if not piper_bin or not aplay_bin:
            raise RuntimeError("piper/aplay not in PATH")

        # Pick model: env var preferred, else try common locations
        env_model = os.getenv("PIPER_MODEL", "").strip()
        candidates = [env_model] if env_model else []
        candidates += [
            "/home/dobot/voices/en_US-amy-medium.onnx",
            "/home/dobot/voices/en_US/amy/medium/en_US-amy-medium.onnx",
            "/home/pi/voices/en_US-amy-medium.on"
            "nx",
            "/usr/share/piper/voices/en_US-amy-medium.onnx",
        ]
        model_path = next((p for p in candidates if p and os.path.isfile(p)), None)
        if not model_path:
            raise RuntimeError("no Piper voice model (.onnx) found; set PIPER_MODEL env var")

        # Optional tunables
        self._piper = {
            "bin": piper_bin,
            "aplay": aplay_bin,
            "model": model_path,
            "device": os.getenv("PIPER_ALSA_DEVICE", "default"),
            "speaker": os.getenv("PIPER_SPEAKER", ""),  # numeric id or empty
            "length_scale": float(os.getenv("PIPER_LENGTH", "0.75")),  # <1 faster, >1 slower
            "noise_scale": float(os.getenv("PIPER_NOISE", "0.6")),    # 0..1
            "noise_w": float(os.getenv("PIPER_NOISE_W", "0.8")),
            "lead_ms": int(float(os.getenv("PIPER_LEAD_MS", "240"))),  # prepend this much silence to avoid clipped first word
        }
        print(f"[TTS] Piper ready with model: {model_path}")
        return "piper"

    def _init_pyttsx3(self):
        import pyttsx3
        eng = pyttsx3.init()
        # Smoother prosody: slightly slower and softer
        try:
            eng.setProperty("rate", 100)
            eng.setProperty("volume", 0.95)
        except Exception:
            pass
        # Try to pick a more natural English voice if available
        try:
            preferred = ("ryan", "en-us", "english", "male")
            for v in eng.getProperty("voices") or []:
                label = (getattr(v, "name", "") + " " + "".join(getattr(v, "languages", []) or [])).lower()
                if any(p in label for p in preferred):
                    eng.setProperty("voice", v.id)
                    break
        except Exception:
            pass
        self._pyttsx3 = eng
        print("[TTS] Using pyttsx3 fallback")
        return "pyttsx3"

    # ---------------- Speaking loop ----------------
    def _loop(self):
        while not self._stop.is_set():
            try:
                text = self._q.get(timeout=0.2)
            except Exception:
                continue
            if not text:
                continue
            try:
                self._speak_once(self._naturalize(text))
            except Exception as e:
                print("[TTS] speak error:", e)

    # Lightweight prosody tweaks (works for both backends)
    def _naturalize(self, text: str) -> str:
        # Insert micro-pauses and contract some phrases for more natural rhythm
        t = text.strip()
        t = t.replace("  ", " ")
        for token in ("however", "therefore", "meanwhile", "okay", "alright"):
            t = t.replace(f" {token} ", f", {token}, ")
        return t

    def _speak_once(self, text: str):
        if self._backend == "piper":
            try:
                self._piper_say(text)
            except Exception as e:
                print("[TTS] Piper failed, falling back to pyttsx3:", e)
                try:
                    self._backend = self._init_pyttsx3()
                    self._pyttsx3_say(text)
                except Exception:
                    print("[TTS] All TTS backends failed.")
        elif self._backend == "pyttsx3":
            self._pyttsx3_say(text)
        else:
            print("[TTS:FALLBACK]", text)

    # Piper: pipe TTS audio directly to aplay for low-latency playback
    def _piper_say(self, text: str):
        import subprocess, tempfile, os, wave, audioop

        # Normalize text to a single line so we synthesize once per utterance
        normalized = " ".join(text.replace("\n", " ").split()).strip()
        if not normalized:
            return

        p = self._piper
        base_cmd = [
            p["bin"], "-m", p["model"], "-f", "-",            # wav on stdout
            "--length_scale", str(p["length_scale"]),
            "--noise_scale", str(p["noise_scale"]),
            "--noise_w", str(p["noise_w"]),
            "--sentence_silence", "0.0"                      # keep snappy between sentences
        ]
        if str(p.get("speaker", "")).strip():
            base_cmd += ["-s", str(p["speaker"]).strip()]

        # 1) Synthesize to a temp WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as raw_wav:
            raw_path = raw_wav.name
        try:
            # Run Piper and write stdout → temp wav
            piper = subprocess.Popen(base_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            # Important: ensure trailing newline so Piper actually renders
            piper.stdin.write((normalized + "\n").encode("utf-8", errors="ignore"))
            piper.stdin.close()

            with open(raw_path, "wb") as out_f:
                # stream copy Piper's stdout to file
                while True:
                    chunk = piper.stdout.read(65536)
                    if not chunk:
                        break
                    out_f.write(chunk)
            piper.wait(timeout=15)

            # 2) Prepend lead-in silence to avoid clipping the first phoneme
            lead_ms = max(0, int(p.get("lead_ms", 240)))
            with wave.open(raw_path, "rb") as w:
                params = w.getparams()
                n_channels = w.getnchannels()
                sampwidth  = w.getsampwidth()
                framerate  = w.getframerate()
                frames     = w.readframes(w.getnframes())

            if lead_ms > 0:
                # create silence of desired duration with the same audio format
                silent_len_frames = int(framerate * lead_ms / 1000.0)
                silence = audioop.mul(b"\x00" * silent_len_frames * n_channels * sampwidth, sampwidth, 0)
                merged = silence + frames
            else:
                merged = frames

            # 3) Write merged audio to a second temp file and play it
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as merged_wav:
                merged_path = merged_wav.name
            try:
                with wave.open(merged_path, "wb") as wout:
                    wout.setnchannels(n_channels)
                    wout.setsampwidth(sampwidth)
                    wout.setframerate(framerate)
                    wout.writeframes(merged)

                aplay_cmd = [p["aplay"], "-q", "-D", p["device"], merged_path]
                subprocess.run(aplay_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
            finally:
                try:
                    os.unlink(merged_path)
                except Exception:
                    pass
        finally:
            try:
                os.unlink(raw_path)
            except Exception:
                pass


    # pyttsx3 backend
    def _pyttsx3_say(self, text: str):
        try:
            self._pyttsx3.say(text)
            self._pyttsx3.runAndWait()
        except Exception:
            pass

    def say(self, text: str):
        try:
            self._q.put_nowait(str(text))
        except Exception:
            pass

    def stop(self):
        self._stop.set()
        try:
            self._q.put_nowait("")
        except Exception:
            pass
        if getattr(self, "_backend", None) == "pyttsx3":
            try:
                self._pyttsx3.stop()
            except Exception:
                pass

# single global TTS worker
tts = TTSWorker()

# ======================================================================
# Configuration (camera, calibration, robot)
# ======================================================================

H_FILE                  = "pixel_to_table_H.npy"
CALIB_FRAME_W           = 640
CALIB_FRAME_H           = 480
MIRROR_X                = False
UNDISTORT_BEFORE_H      = True

MIC_ALSA_DEVICE  = "hw:2,0"

# Camera intrinsics (from your setup)
K_MATRIX = np.array([[9.0518e3, 0.0,      353.7677],
                     [0.0,      3.6651e4, 257.2904],
                     [0.0,      0.0,      1.0     ]], dtype=np.float64)
DIST_COEFFS = np.array([9.7265, -1.0076e4, 0.0, 0.0], dtype=np.float64)

# Robot parameters
HOME_POSE        = [231.9, -8.7, 153.8, 0]
camera_offset_X  = 0.0
camera_offset_Y  = 0.0

# Fallback single-drop (used if a label has no custom drop position)
FALLBACK_DROP_X  = 198.3
FALLBACK_DROP_Y  = 216.6

Z_PICK_DEFAULT   = -22.5
Z_DROP_DEFAULT   = 35.0    # tray surface Z for dropping
DROP_FILTER_MM   = 20.0    # ignore detections too close to the drop zone

# Per-label custom drop positions (x, y, z_drop)
drop_positions = {
    "red apple":   (123.0,  -237.6,  Z_DROP_DEFAULT),
    "green apple": ( 63.0,  -234.1,  Z_DROP_DEFAULT),
    "pumpkin":     (  5.6,  -228.9,  Z_DROP_DEFAULT),
    "red grape":   ( -0.2,  -288.1,  Z_DROP_DEFAULT),
    "black grape": ( 56.5,  -296.8,  Z_DROP_DEFAULT),
    "olive":       (114.8,  -301.3,  Z_DROP_DEFAULT),
    "strawberry":  (112.9,  -347.4,  Z_DROP_DEFAULT),
    "pomegranate": ( 53.2,  -352.8,  Z_DROP_DEFAULT),
    "carrot":      (  4.2,  -351.1,  Z_DROP_DEFAULT),
}

# Allowed detection labels (kept broad to match model classes)
ALLOWED_LABELS = [
    "black grape", "carrot", "corn", "eggplant", "green apple",
    "olive", "pomegranate", "pumpkin", "red apple", "red grape",
    "strawberry", "white radish",
]

# Per-label Z offsets (mm) for pick height tuning
Z_PICK_DICT = {
    "black grape":   -22.5,
    "carrot":        Z_PICK_DEFAULT,
    "corn":          Z_PICK_DEFAULT,
    "eggplant":      Z_PICK_DEFAULT,
    "green apple":   -11.0,
    "olive":         -16.5,
    "pomegranate":   -18.5,
    "pumpkin":       -19.0,
    "red apple":     -11.0,
    "red grape":     -22.5,
    "strawberry":    -19,
    "white radish":  Z_PICK_DEFAULT,
}

# Vosk ASR
VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"
VOSK_SAMPLE_RATE = 16000
VOSK_BLOCKSIZE   = 8000
VOSK_MAX_WORDS   = 3

# Talkback helpers

def _plural(label: str) -> str:
    if label.endswith("s"):
        return label
    if label.endswith("y"):
        return label[:-1] + "ies"
    if label in ("red apple", "green apple", "red grape", "black grape"):
        return label + "s"
    return label + "s"

# ======================================================================
# Helpers / VoiceListener
# ======================================================================

def undistort_point_like_undistort_image(u, v, K, dist):
    pts = np.array([[[u, v]]], dtype=np.float64)
    uvn = cv2.undistortPoints(pts, K, dist, P=K)
    return float(uvn[0, 0, 0]), float(uvn[0, 0, 1])

# Only fruit phrases (no stop, no pick-everything)
FRUIT_LABEL_MAP = {
    "black grape": ["black grape"],
    "black grapes": ["black grape"],
    "carrot": ["carrot"],
    "corn": ["corn"],
    "eggplant": ["eggplant"],
    "green apple": ["green apple"],
    "green apples": ["green apple"],
    "olive": ["olive"],
    "pomegranate": ["pomegranate"],
    "pumpkin": ["pumpkin"],
    "red apple": ["red apple"],
    "red apples": ["red apple"],
    "red grape": ["red grape"],
    "red grapes": ["red grape"],
    "strawberry": ["strawberry"],
    "strawberries": ["strawberry"],
    "white radish": ["white radish"],
}

def _build_grammar_phrases():
    phrases = set(FRUIT_LABEL_MAP.keys())
    return sorted(phrases)

VOSK_PHRASES = _build_grammar_phrases()

def pick_working_mic_index(prefer=("usb", "mic", "microphone")) -> int:
    pa = pyaudio.PyAudio()
    candidates = []
    try:
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            name = (info.get("name") or "").lower()
            max_in = int(info.get("maxInputChannels") or 0)
            if max_in > 0:
                score = 1
                if any(k in name for k in prefer):
                    score = 0
                candidates.append((score, i, info))
        if not candidates:
            raise RuntimeError("No input-capable audio devices found.")
        for _, idx, info in sorted(candidates):
            try:
                stream = pa.open(format=pyaudio.paInt16,
                                 channels=1, rate=VOSK_SAMPLE_RATE,
                                 input=True, input_device_index=idx,
                                 frames_per_buffer=1024)
                stream.close()
                print(f"[AUDIO] Selected input {idx}: {info['name']}")
                return idx
            except Exception as e:
                print(f"[AUDIO] Skipping {idx} ({info['name']}): {e}")
        idx = sorted(candidates)[0][1]
        print(f"[AUDIO] Falling back to {idx}")
        return idx
    finally:
        pa.terminate()

class VoiceListener:
    def __init__(self, device_index=None, on_label=None):
        self.device_index = device_index
        self.on_label = on_label or (lambda label, all_targets=None: None)
        self._stop = False
        self._thread = None

        try:
            if not os.path.isdir(VOSK_MODEL_PATH):
                raise RuntimeError(
                    f"Vosk model folder not found at '{VOSK_MODEL_PATH}'. "
                    "Download & unzip a model, then update VOSK_MODEL_PATH."
                )
            self._vosk_model = Model(VOSK_MODEL_PATH)
        except Exception as e:
            print(f"[VOSK] Failed to load model: {e}")
            self._vosk_model = None

    @staticmethod
    def find_input_device_index(prefer_keywords=("usb", "mic", "microphone")):
        pa = pyaudio.PyAudio()
        chosen = None
        try:
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if int(info.get("maxInputChannels", 0)) > 0:
                    name = (info.get("name") or "").lower()
                    if any(k in name for k in prefer_keywords):
                        chosen = i
                        break
            if chosen is None:
                chosen = pa.get_default_input_device_info()["index"]
        finally:
            pa.terminate()
        return chosen

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        if self.device_index is None:
            self.device_index = self.find_input_device_index()
        if self._vosk_model is None:
            print("[VOSK] No model loaded; voice control disabled.")
            return
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print("[Voice] Say a fruit name, like 'red apple' or 'strawberries'.")
        tts.say(" Hello Dobot Here. Say a fruit name.")

    def stop(self):
        self._stop = True
        if self._thread:
            self._thread.join()
        print("\n[Voice] Listener stopped.")

    def _run(self):
        pa = pyaudio.PyAudio()
        stream = None
        try:
            stream = pa.open(format=pyaudio.paInt16,
                             channels=1,
                             rate=VOSK_SAMPLE_RATE,
                             input=True,
                             input_device_index=self.device_index,
                             frames_per_buffer=VOSK_BLOCKSIZE)

            grammar_json = json.dumps(VOSK_PHRASES)
            recognizer = KaldiRecognizer(self._vosk_model, VOSK_SAMPLE_RATE, grammar_json)

            print("Listening (Vosk, offline, constrained grammar)...")
            while not self._stop:
                data = stream.read(VOSK_BLOCKSIZE, exception_on_overflow=False)
                if not data:
                    continue

                if recognizer.AcceptWaveform(data):
                    result = recognizer.Result()
                    try:
                        parsed = json.loads(result)
                    except Exception:
                        parsed = {}
                    self._handle_final(parsed)
                else:
                    pass

        except Exception as e:
            print(f"[Fatal Voice Error] {e}")
        finally:
            try:
                if stream is not None:
                    stream.stop_stream()
                    stream.close()
            except Exception:
                pass
            pa.terminate()

    def _handle_final(self, parsed):
        text = (parsed.get("text") or "").strip().lower()
        if not text:
            return

        # Keep short phrases only
        words = text.split()
        if VOSK_MAX_WORDS and len(words) > VOSK_MAX_WORDS:
            return

        # Fruits (via map) only
        if text in FRUIT_LABEL_MAP:
            canon = FRUIT_LABEL_MAP[text][0]
            print(f"[YOU SAID] {text} → {canon}")
            self.on_label(canon)
            return

        # Otherwise ignore silently
        return

class MicGst:
    def __init__(self, alsa_device=MIC_ALSA_DEVICE, log_levels=True, level_interval_ms=200):
        self.alsa_device = alsa_device
        self.log_levels = log_levels
        self.level_interval_ns = int(level_interval_ms * 1e6)
        self.pipeline = None
        self.bus = None

        Gst.init(None)
        if self.log_levels:
            desc = (
                f'alsasrc device={self.alsa_device} ! tee name=t '
                't. ! queue ! audioconvert ! audioresample ! volume name=micvol volume=1.0 ! autoaudiosink '
                f't. ! queue ! level name=miclevel interval={self.level_interval_ns} message=true ! fakesink'
            )
        else:
            desc = (
                f'alsasrc device={self.alsa_device} ! '
                'queue ! audioconvert ! audioresample ! volume name=micvol volume=1.0 ! autoaudiosink '
            )
        self.pipeline = Gst.parse_launch(desc)
        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect("message", self._on_bus_message)

    def _on_bus_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.ELEMENT and self.log_levels:
            s = message.get_structure()
            if s and s.get_name() == "level":
                peaks = s.get_value("peak")
                rms = s.get_value("rms")
                if peaks and len(peaks) > 0 and rms and len(rms) > 0:
                    print(f"[MIC] peak: {peaks[0]:.1f} dB | rms: {rms[0]:.1f} dB", end="\r")
        elif t == Gst.MessageType.ERROR:
            err, dbg = message.parse_error()
            print(f"\n[MicGst] ERROR: {err} | debug: {dbg}")
        elif t == Gst.MessageType.WARNING:
            wrn, dbg = message.parse_warning()
            print(f"\n[MicGst] WARNING: {wrn} | debug: {dbg}")

    def set_volume(self, gain_linear: float):
        vol = self.pipeline.get_by_name("micvol")
        if vol:
            vol.set_property("volume", float(gain_linear))

    def start(self):
        if not self.pipeline:
            return False
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("[MicGst] Failed to start microphone pipeline.")
            return False
        print(f"[MicGst] Mic monitor PLAYING on '{self.alsa_device}'.")
        return True

    def stop(self):
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
            print("\n[MicGst] Mic monitor stopped.")

# ======================================================================
# User-defined app callback (robot + detections + voice bridge)
# ======================================================================

class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()

        # Robot
        self.robot = None
        self.robot_lock = threading.Lock()
        self.robot_thread = None
        self.is_robot_moving = False

        # Detections cache: list of (u_cal, v_cal, label, track_id)
        self.last_detections = []

        # Homography / camera
        self.H = None
        self.H_INV = None
        self.K = K_MATRIX
        self.dist = DIST_COEFFS

        # Params
        self.home_pose = HOME_POSE
        self.camera_offset_x = camera_offset_X
        self.camera_offset_y = camera_offset_Y

        # Fallback single-drop
        self.fallback_drop_x = FALLBACK_DROP_X
        self.fallback_drop_y = FALLBACK_DROP_Y

        # Per-label drops
        self.drop_positions = drop_positions

        self.z_pick_default = Z_PICK_DEFAULT
        self.z_drop_default = Z_DROP_DEFAULT
        self.z_pick_dict = Z_PICK_DICT
        self.drop_filter_mm = DROP_FILTER_MM

        # Conversational state
        self.active_label = None        # currently selected fruit label
        self.pick_mode_active = False   # True if robot should actively look for targets
        self.active_done_miss = 0       # "no targets seen" streak counter
        self.active_done_patience = 0  # frames with no target before declaring done
        self._picked_ids = set()        # track_ids already attempted
        self._quick_empty_frames = 0  # fast-finish when no targets for a couple frames

        # Voice UX flags for the current run (generic across labels)
        self._announced_start_for_active = False   # said "picking {label}"
        self._saw_any_for_active = False           # saw at least one detection for {label}

        # Init robot + homography if available
        if ROBOT_AVAILABLE:
            self.init_robot()
            self.load_homography()
            self.start_prompt()

    # --- Robot / Homography ---

    def load_homography(self):
        try:
            self.H = np.load(H_FILE)
            self.H_INV = np.linalg.inv(self.H)
            print(f"Loaded homography from '{H_FILE}'")
            print("H =\n", self.H)
            print(f"Calib: {CALIB_FRAME_W}x{CALIB_FRAME_H} | MirrorX={MIRROR_X} | Undistort={UNDISTORT_BEFORE_H}")
            return True
        except Exception as e:
            print(f"Failed to load homography file: {e}")
            return False

    def init_robot(self):
        try:
            with self.robot_lock:
                self.robot = Dbt.DoBotArm(*self.home_pose[:3])
                self.robot.moveHome()
                print("Dobot connected and homed.")
            return True
        except Exception as e:
            print(f"Failed to connect to Dobot: {e}")
            self.robot = None
            return False

    def start_prompt(self):
        print("Voice control:")
        print("  - Say a fruit (e.g., 'red apples', 'strawberries', 'white radish').")

    # --- Geometry ---

    def pixel_to_xy(self, u_pix: float, v_pix: float):
        if self.H is None:
            return u_pix, v_pix
        uv1 = np.array([u_pix, v_pix, 1.0], dtype=np.float64)
        xyw = self.H @ uv1
        return xyw[0] / xyw[2], xyw[1] / xyw[2]

    # --- Drop helpers ---

    def get_drop_xyz(self, label: str):
        """
        Returns (drop_x, drop_y, drop_z) for a given fruit label,
        falling back to the single-drop if not found.
        """
        pos = self.drop_positions.get(label)
        if pos is not None:
            return float(pos[0]), float(pos[1]), float(pos[2])
        # Fallback to v1 single drop with default Z for tray
        return float(self.fallback_drop_x), float(self.fallback_drop_y), float(self.z_drop_default)

    def filter_detection_by_drop_zone(self, X_mm, Y_mm, label: str):
        """
        Ignore detections that are too close to THAT LABEL'S drop zone if available,
        else apply the fallback single-drop region.
        """
        pos = self.drop_positions.get(label)
        if pos:
            drop_x, drop_y = pos[0], pos[1]
        else:
            drop_x, drop_y = self.fallback_drop_x, self.fallback_drop_y
        return (abs(X_mm - drop_x) < self.drop_filter_mm and
                abs(Y_mm - drop_y) < self.drop_filter_mm)

    # --- Conversational flow ---

    def start_category_run(self, label: str):
        """Begin picking only this category until none remain."""
        self.active_label = label
        self._picked_ids.clear()
        self.active_done_miss = 0
        self._quick_empty_frames = 0
        self.pick_mode_active = True

        # reset voice flags each run
        self._announced_start_for_active = False
        self._saw_any_for_active = False

    def finish_category_run(self):
        """Called when that category appears cleared or patience exceeded."""
        lbl = self.active_label or ""
        self.pick_mode_active = False

        # Decide end-of-run message BEFORE wiping label
        if self._saw_any_for_active:
            tts.say(f" Done with {lbl}. Say another fruit.")
        else:
            tts.say(f" {lbl} not present say another fruit")

        # reset state
        self.active_label = None
        self.active_done_miss = 0
        self._picked_ids.clear()
        self._announced_start_for_active = False
        self._saw_any_for_active = False
        self._quick_empty_frames = 0

    # --- Detection handling / robot execution ---

    def process_detections_for_robot(self, detections, frame_width, frame_height):
        """Fill self.last_detections with calibrated pixels/labels/IDs."""
        self.last_detections.clear()

        for det in detections:
            label = det.get_label().strip().lower() if isinstance(det.get_label(), str) else None
            if label not in Z_PICK_DICT:
                continue

            bbox = det.get_bbox()
            track_id = 0
            track = det.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if len(track) == 1:
                track_id = track[0].get_id()

            cx = bbox.xmin() + (bbox.width() / 2.0)
            cy = bbox.ymin() + (bbox.height() / 2.0)
            u = cx * frame_width if cx <= 1.0 else cx
            v = cy * frame_height if cy <= 1.0 else cy

            if MIRROR_X and frame_width is not None:
                u = (frame_width - u)
            if UNDISTORT_BEFORE_H:
                u, v = undistort_point_like_undistort_image(u, v, self.K, self.dist)

            if self.H is not None:
                X_mm, Y_mm = self.pixel_to_xy(u, v)
                if self.filter_detection_by_drop_zone(X_mm, Y_mm, label):
                    continue

            self.last_detections.append((u, v, label, track_id))

        return len(self.last_detections)

    def pick_all_step(self):
        # Only run when voice mode has armed the picker
        if not self.pick_mode_active or not self.active_label:
            return

        # If the robot is already moving, do nothing this frame
        if self.is_robot_moving:
            return

        # Only consider detections matching the active label
        candidates = []
        for (u_cal, v_cal, label, track_id) in self.last_detections:
            if label != self.active_label:
                continue
            if track_id and track_id in self._picked_ids:
                continue
            candidates.append((u_cal, v_cal, label, track_id))

        # Voice UX: first time we actually see candidates → announce "picking {label}"
        if self.active_label and candidates:
            self._saw_any_for_active = True
            if not self._announced_start_for_active:
                tts.say(f"Okay,Picking {self.active_label}")
                self._announced_start_for_active = True

        if not candidates:
            # Fast-finish: if robot is idle and we don't see any more of this label
            if not self.is_robot_moving:
                self._quick_empty_frames += 1
                # 2 frames ≈ 70–130ms depending on FPS → very snappy “done”
                if self._quick_empty_frames >= 2:
                    self.finish_category_run()
                    return
            else:
                # robot still moving → don't fast-finish yet
                self._quick_empty_frames = 0

            # Fallback to slower patience (safety net)
            self.active_done_miss += 1
            if self.active_done_miss >= self.active_done_patience:
                self.finish_category_run()
            return


        self.active_done_miss = 0
        cx0, cy0 = CALIB_FRAME_W / 2, CALIB_FRAME_H / 2
        target = min(candidates, key=lambda d: (d[0] - cx0) ** 2 + (d[1] - cy0) ** 2)

        if ROBOT_AVAILABLE and self.robot is not None:
            self.is_robot_moving = True
            self.robot_thread = threading.Thread(
                target=self.robot_worker, args=target, daemon=True
            )
            self.robot_thread.start()
            if target[3] != 0:
                self._picked_ids.add(target[3])

    def robot_worker(self, u_cal, v_cal, label, track_id):
        """Convert cal-pixels → XY and perform pick-and-place with smart drop selection."""
        print(f"⬇ Pick → {label} (ID:{track_id}) pix_cal=({u_cal:.1f},{v_cal:.1f})")

        try:
            if not ROBOT_AVAILABLE or self.robot is None:
                print("Robot not available.")
                tts.say("Robot is not available.")
                return

            X, Y = self.pixel_to_xy(u_cal, v_cal)
            Z_pick = self.z_pick_dict.get(label, self.z_pick_default)

            drop_x, drop_y, drop_z = self.get_drop_xyz(label)

            print(f"  → XY(mm)=({X:.1f},{Y:.1f},{Z_pick})  | drop=({drop_x:.1f},{drop_y:.1f},{drop_z:.1f})")

            with self.robot_lock:
                if not self.pick_mode_active or label != (self.active_label or ""):
                    print("Pick cancelled (state changed).")
                    return

                # Prefer tray paths if available and a per-label drop exists
                use_tray = (
                    USE_TRAY_PATHS_IF_AVAILABLE and
                    label in self.drop_positions and
                    hasattr(self.robot, "pick_and_place_tray_loc")
                )

                if use_tray:
                    # reasonable defaults; adjust for your rig
                    staging       = (257.7,   6.5, 66.3)
                    tray_approach = ( 91.2, -239.5, 77.9)
                    self.robot.pick_and_place_tray_loc(
                        X + self.camera_offset_x,
                        Y + self.camera_offset_y,
                        Z_pick,
                        drop_x,
                        drop_y,
                        drop_z,
                        staging=staging,
                        tray_approach=tray_approach
                    )
                else:
                    # fall back to classic single-segment drop
                    if hasattr(self.robot, "pick_and_place"):
                        self.robot.pick_and_place(
                            X + self.camera_offset_x,
                            Y + self.camera_offset_y,
                            Z_pick,
                            drop_x,
                            drop_y,
                            self.z_drop_default
                        )
                    else:
                        # If classic method is missing, use tray_loc with defaults anyway
                        staging       = (257.7,   6.5, 66.3)
                        tray_approach = ( 91.2, -239.5, 77.9)
                        self.robot.pick_and_place_tray_loc(
                            X + self.camera_offset_x,
                            Y + self.camera_offset_y,
                            Z_pick,
                            drop_x,
                            drop_y,
                            drop_z,
                            staging=staging,
                            tray_approach=tray_approach
                        )

                # self.robot.moveHome()
                print("Pick complete.")
                self.robot.moveHome()
                # intentionally no per-pick TTS to keep UX concise


        except Exception as e:
            print(f"Robot move failed: {e}")
            tts.say(f"Sorry, I could not complete the pick for {label}.")
        finally:
            with self.robot_lock:
                self.is_robot_moving = False
        with self.robot_lock:
            self.is_robot_moving = False
            # Encourage fast-finish check to trigger on the next frame if table is clear
            if self.pick_mode_active and self.active_label:
                self._quick_empty_frames = max(self._quick_empty_frames, 1)


    # --- UI overlay / per-frame processing ---

    def annotate_frame_with_robot_info(self, frame, detection_count, width, height):
        if frame is None:
            return

        is_moving_status = "ACTIVE" if self.is_robot_moving else "IDLE"

        cv2.putText(frame, f"Detections: {detection_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        status = (f"Robot:{'ON' if self.robot is not None else 'DISABLED'} "
                  f"| Mode:{'RUN' if self.pick_mode_active else 'IDLE'} "
                  f"| Active:{self.active_label or '-'}")
        cv2.putText(frame, status, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        cv2.putText(frame, f"Robot Motion: {is_moving_status}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        if DRAW_OVERLAY and self.active_label:
            dx, dy, _ = self.get_drop_xyz(self.active_label)
            cv2.putText(frame, f"Drop: {self.active_label} → ({dx:.1f}, {dy:.1f})",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    def process_callback_detections(self, detections, format_info, frame=None):
        width, height = format_info[1], format_info[2]

        detection_count = self.process_detections_for_robot(detections, width, height)

        self.pick_all_step()

        if self.use_frame and frame is not None and DRAW_OVERLAY:
            try:
                if self.last_detections and not self.is_robot_moving and self.active_label:
                    cx0, cy0 = CALIB_FRAME_W / 2, CALIB_FRAME_H / 2
                    filtered = [d for d in self.last_detections if d[2] == self.active_label]
                    if filtered:
                        target = min(filtered, key=lambda d: (d[0] - cx0) ** 2 + (d[1] - cy0) ** 2)
                        u_cal, v_cal, _, _ = target
                        sx = (width / CALIB_FRAME_W) if CALIB_FRAME_W else 1.0
                        sy = (height / CALIB_FRAME_H) if CALIB_FRAME_H else 1.0
                        u_show = width - (u_cal * sx) if MIRROR_X else (u_cal * sx)
                        v_show = v_cal * sy
                        cv2.circle(frame, (int(u_show), int(v_show)), 6, (255, 0, 0), -1)
            except Exception:
                pass

            self.annotate_frame_with_robot_info(frame, len(self.last_detections), width, height)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.set_frame(frame)

        return ""

# ======================================================================
# Pad probe callback for Hailo GStreamer app
# ======================================================================

def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    format, width, height = get_caps_from_pad(pad)
    format_info = (format, width, height)

    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        frame = get_numpy_from_buffer(buffer, format, width, height)

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    msg = user_data.process_callback_detections(detections, format_info, frame)
    if msg:
        print(msg, end="")
    return Gst.PadProbeReturn.OK

# ======================================================================
# Voice callback (only fruit names)
# ======================================================================

def on_voice_label(label, all_targets=None):
    cmd = (label or "").strip().lower()
    user_data.start_category_run(cmd)

# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    env_file     = project_root / ".env"
    os.environ["HAILO_ENV_FILE"] = str(env_file)

    user_data = user_app_callback_class()

    mic_mon = None
    # if ENABLE_GST_MON:
    #     mic_mon = MicGst(alsa_device=MIC_ALSA_DEVICE, log_levels=True, level_interval_ms=200)
    #     mic_mon.start()

    MIC_DEVICE_INDEX = pick_working_mic_index()
    voice = VoiceListener(device_index=MIC_DEVICE_INDEX, on_label=on_voice_label)
    voice.start()

    app = GStreamerDetectionApp(app_callback, user_data)

    print("Robot Integration Active!")
    if ROBOT_AVAILABLE and user_data.robot is not None:
        print("  - Voice: say a fruit (e.g., 'red apples', 'strawberry', 'white radish').")

    try:
        app.run()
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        try:
            tts.stop()
        except Exception:
            pass
        if mic_mon:
            mic_mon.stop()
        if ROBOT_AVAILABLE and user_data.robot:
            print("Attempting to home robot before exit.")
            with user_data.robot_lock:
                try:
                    user_data.robot.moveHome()
                except Exception:
                    pass

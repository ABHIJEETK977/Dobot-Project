#!/usr/bin/env python3
# Calibrate pixel→(X,Y) using the SAME Hailo-like camera pipeline (mirror + 640x480 RGB).
# Controls:
#   c : lock pixel under mouse (green dot)
#   o : pair locked pixel with current robot (X,Y)
#   z : undo last pair
#   h : robot home
#   q : solve & save homography (needs ≥4 pairs)
# Esc : quit without saving

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import numpy as np
import cv2
import threading
import time
import json
import sys

# ---- robot (adjust import if needed) ----
import dobot_lib.DoBotArm as Dbt

# ---- config ----
SAVE_FILE   = "resources/pixel_to_table_H.npy"
PAIRS_LOG   = "resources/homography_pairs.json"
RANSAC_THRESH = 2.0
FRAME_W, FRAME_H = 640, 480
FPS = 30
CAM_DEVICE = "/dev/video0"   # change if needed

# ---------- GStreamer appsink wrapper (USB cam → decode → H-FLIP → RGB 640x480) ----------
class GStreamerCamera:
    def __init__(self, device=CAM_DEVICE, width=FRAME_W, height=FRAME_H, fps=FPS):
        Gst.init(None)
        self.width = width
        self.height = height
        self.frame = None
        self.lock = threading.Lock()
        self.loop = None

        # JPEG v4l2src → decodebin → videoflip horiz → videoscale → videoconvert → RGB 640x480 → appsink
        pipeline_str = (
            f"v4l2src device={device} name=source ! "
            f"image/jpeg, framerate={fps}/1, width=640, height=480 ! "
            f"queue leaky=no max-size-buffers=3 ! "
            f"decodebin name=dec ! "
            f"videoflip video-direction=horiz ! "
            f"queue leaky=no max-size-buffers=3 ! "
            f"videoscale ! videoconvert ! "
            f"video/x-raw,format=RGB,width={width},height={height},framerate={fps}/1 ! "
            f"queue leaky=no max-size-buffers=3 ! "
            f"appsink name=sink emit-signals=true max-buffers=1 drop=true"
        )

        self.pipeline = Gst.parse_launch(pipeline_str)
        self.sink = self.pipeline.get_by_name("sink")
        self.sink.connect("new-sample", self.on_sample)

    def on_sample(self, sink):
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR
        buf = sample.get_buffer()
        caps = sample.get_caps()
        st = caps.get_structure(0)
        w = st.get_value("width")
        h = st.get_value("height")
        success, mapinfo = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR
        try:
            arr = np.frombuffer(mapinfo.data, dtype=np.uint8)
            frame = arr.reshape(h, w, 3)  # RGB
            with self.lock:
                self.frame = frame.copy()
        finally:
            buf.unmap(mapinfo)
        return Gst.FlowReturn.OK

    def start(self):
        self.loop = GLib.MainLoop()
        self.pipeline.set_state(Gst.State.PLAYING)
        th = threading.Thread(target=self.loop.run, daemon=True)
        th.start()

    def stop(self):
        try:
            self.pipeline.set_state(Gst.State.NULL)
            if self.loop is not None:
                self.loop.quit()
        except Exception:
            pass

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

# ---------- Calibrator logic ----------
def main():
    # Connect robot & home
    HOME_POSE = (231.9, -8.7, 153.8) # X,Y,Z, rHead (adjust as needed)
    bot = Dbt.DoBotArm(HOME_POSE[0], HOME_POSE[1], HOME_POSE[2])
    print("Connected to DoBot. Homing...")
    bot.moveHome()

    cam = GStreamerCamera()
    cam.start()
    time.sleep(0.8)

    pix_pts, world_pts = [], []
    captured_pixel = None   # frozen (u,v) after 'c'
    have_unpaired_pixel = False
    last_mouse = (0, 0)

    def on_mouse(event, x, y, flags, param):
        nonlocal last_mouse
        if event == cv2.EVENT_MOUSEMOVE:
            last_mouse = (x, y)

    cv2.namedWindow("calib", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("calib", FRAME_W, FRAME_H)
    cv2.setMouseCallback("calib", on_mouse)

    print("\nControls:")
    print("  Move mouse over any checker/feature corner")
    print("  c : capture pixel")
    print("  o : pair with current robot pose (X,Y)")
    print("  z : undo last pair")
    print("  q : solve & save homography (needs ≥4 pairs)")
    print("  h : robot home")
    print("Esc: quit without saving\n")

    while True:
        ok, frame = cam.read()
        if not ok:
            time.sleep(0.01)
            continue

        # Calibrate EXACTLY on the mirrored+scaled RGB frame
        disp = frame.copy()  # RGB

        # live cursor (white)
        cv2.circle(disp, last_mouse, 4, (255, 255, 255), -1)

        # locked pixel (green)
        if captured_pixel is not None:
            cv2.circle(disp, captured_pixel, 6, (0, 255, 0), -1)
            cv2.putText(disp, f"locked: {captured_pixel}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(disp, f"pairs: {len(pix_pts)}   c:pixel  o:pair  z:undo  q:solve  h:home  Esc:quit",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 28), 2)

        # show (convert RGB→BGR for display)
        cv2.imshow("calib", cv2.cvtColor(disp, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            captured_pixel = tuple(last_mouse)  # freeze now
            have_unpaired_pixel = True
            print(f"pixel captured (u,v)= {captured_pixel}")

        elif key == ord('o'):
            if not have_unpaired_pixel or captured_pixel is None:
                print("Press 'c' first.")
                continue

            time.sleep(0.8)  # wait for robot to settle after move/home

            pose = np.asarray(bot.cur_location(), float)
            X, Y = pose[:2]

            if world_pts:
                lastX, lastY = world_pts[-1]
                if ((X - lastX) ** 2 + (Y - lastY) ** 2) ** 0.5 < 8.0:
                    print(f"Rejected: XY too close to previous point ({X:.1f},{Y:.1f})")
                    continue

            pix_pts.append(captured_pixel)
            world_pts.append((float(X), float(Y)))
            print(f"paired (u,v)= {captured_pixel}  ↔  (X,Y)= ({X:.1f},{Y:.1f})")
            captured_pixel = None
            have_unpaired_pixel = False

        elif key == ord('z'):
            if pix_pts:
                print("Undo last pair:", pix_pts[-1], "<->", world_pts[-1])
                pix_pts.pop(); world_pts.pop()
            else:
                print("Nothing to undo.")

        elif key == ord('h'):
            try:
                bot.moveHome()
            except Exception as e:
                print("Home move error:", e)

        elif key == ord('q'):
            if len(pix_pts) < 4:
                print("Need at least 4 pairs."); continue
            if len(pix_pts) != len(world_pts):
                print("Pair list lengths mismatch—undo or complete the pending pair."); continue

            pix = np.array(pix_pts,   np.float32)
            wxy = np.array(world_pts, np.float32)

            H, inliers = cv2.findHomography(
                pix, wxy, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_THRESH
            )
            if H is None:
                print("Homography failed. Add more diverse points and try again."); continue

            # Save H
            np.save(SAVE_FILE, H)

            # Reprojection error (mm)
            pix_h = np.hstack([pix, np.ones((pix.shape[0], 1), np.float32)])  # [u v 1]
            proj = (H @ pix_h.T).T
            proj = proj[:, :2] / proj[:, 2:3]
            err = np.linalg.norm(proj - wxy, axis=1)
            inl = int(inliers.sum()) if inliers is not None else 0

            print("\nHomography H saved →", SAVE_FILE)
            print("H =\n", H)
            print(f"RANSAC inliers: {inl}/{len(pix)} | mean reproj err: {err.mean():.2f} mm")

            # Save pairs
            with open(PAIRS_LOG, "w") as fp:
                json.dump({"pixels": [tuple(p) for p in pix_pts],
                           "worldXY": [tuple(w) for w in world_pts]}, fp, indent=2)
            print("Pairs written to", PAIRS_LOG)
            break

        elif key == 27:
            print("Esc pressed – exiting without saving.")
            break

    cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")

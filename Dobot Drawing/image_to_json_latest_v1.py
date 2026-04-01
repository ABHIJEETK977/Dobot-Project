import cv2
import numpy as np
import json
import math
import os
from scipy.interpolate import splprep, splev


class ImageToSmoothPath:
    def __init__(
        self,
        image_path,
        output_width=100.0,
        output_height=100.0,
        # TUNING FOR V5
        stitch_threshold_mm=1.0,  # Strict stitching to prevent jumping across the face
        smoothing_factor=0.2,     # VERY LOW (0.2) to keep sharp details in eyes/mouth
        sample_density_mm=1.2,    # High resolution
        min_path_len_mm=2.0       # Keep small nose/eye details
    ):
        self.image_path = image_path
        self.output_width = output_width
        self.output_height = output_height
        self.stitch_threshold_mm = stitch_threshold_mm
        self.smoothing_factor = smoothing_factor
        self.sample_density_mm = sample_density_mm
        self.min_path_len_mm = min_path_len_mm

        self.img_height = 0
        self.img_width = 0

    # =========================================================
    # 1. HIGH-RES SKELETONIZATION
    # =========================================================

    def load_and_skeletonize(self):
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Image not found: {self.image_path}")

        img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Could not load image.")

        # CHANGE 1: Upscale 4x (Was 2x)
        # This makes the facial features "huge" so they don't merge when we thicken lines.
        scale_factor = 4
        img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

        # CHANGE 2: Conservative Dilation
        # We use a 3x3 kernel on a 4x image. This is effectively a "finer pen."
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.dilate(binary, kernel, iterations=1)

        # Close gaps (Morphological Closing) - helps with dashed lines
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        self.img_height, self.img_width = binary.shape

        # Standard Skeletonization
        skeleton = np.zeros(binary.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        while True:
            open_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, element)
            temp = cv2.subtract(binary, open_img)
            eroded = cv2.erode(binary, element)
            skeleton = cv2.bitwise_or(skeleton, temp)
            binary = eroded.copy()
            if cv2.countNonZero(binary) == 0:
                break

        return skeleton

    # =========================================================
    # 2. TRACE (Standard)
    # =========================================================
    def trace_skeleton(self, skeleton):
        paths = []
        skel_pad = np.pad(skeleton, 1, mode='constant', constant_values=0)

        H, W = skel_pad.shape

        while True:
            # Find one remaining non-zero pixel without allocating all coordinates
            flat_idx = int(np.argmax(skel_pad))
            if skel_pad.flat[flat_idx] == 0:
                break  # no pixels left

            cy, cx = np.unravel_index(flat_idx, (H, W))

            path = []
            skel_pad[cy, cx] = 0
            path.append((cx - 1, cy - 1))

            while True:
                neighbors = []
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        if skel_pad[cy + dy, cx + dx] > 0:
                            neighbors.append((cx + dx, cy + dy))

                if not neighbors:
                    break

                nx, ny = neighbors[0]
                skel_pad[ny, nx] = 0
                cx, cy = nx, ny
                path.append((cx - 1, cy - 1))

            if len(path) > 10:
                paths.append(path)

        return paths


    # =========================================================
    # 3. MAPPING (Standard)
    # =========================================================
    def pixels_to_mm(self, paths):
        mm_paths = []
        if self.img_width == 0: return []
        # Preserves aspect ratio while scaling image within image width and height and centers it
        scale = min(self.output_width / self.img_width, self.output_height / self.img_height)
        off_x = (self.output_width - (self.img_width * scale)) / 2
        off_y = (self.output_height - (self.img_height * scale)) / 2
        for path in paths:
            new_path = []
            for x, y in path:
                mx = (x * scale) + off_x
                my = (y * scale) + off_y 
                new_path.append((mx, my))
            mm_paths.append(new_path)
        return mm_paths

    # =========================================================
    # 4. STITCHING (Standard)
    # =========================================================
    def stitch_paths(self, paths):
        if not paths: return []
        pool = [list(p) for p in paths]  # Modifiable copy
        stitched = []
        
        while pool:
            current_path = pool.pop(0)
            changed = True
            while changed:
                changed = False
                best_idx = -1
                best_dist = self.stitch_threshold_mm
                should_reverse = False
                tail = current_path[-1]
                
                for i, candidate in enumerate(pool):
                    head = candidate[0]
                    tail_cand = candidate[-1]
                    d1 = math.hypot(tail[0]-head[0], tail[1]-head[1])
                    if d1 < best_dist:
                        best_dist = d1
                        best_idx = i
                        should_reverse = False
                    d2 = math.hypot(tail[0]-tail_cand[0], tail[1]-tail_cand[1])
                    if d2 < best_dist:
                        best_dist = d2
                        best_idx = i
                        should_reverse = True
                
                if best_idx != -1:
                    next_segment = pool.pop(best_idx)
                    if should_reverse: next_segment.reverse()
                    current_path.extend(next_segment)
                    changed = True
            
            total_len = 0
            for k in range(len(current_path)-1):
                total_len += math.hypot(current_path[k][0]-current_path[k+1][0], 
                                      current_path[k][1]-current_path[k+1][1])
            if total_len >= self.min_path_len_mm:
                stitched.append(current_path)
        return stitched

    # =========================================================
    # 5. SMOOTHING (Standard)
    # =========================================================
    def smooth_paths_spline(self, paths):
        smooth_paths = []
        for path in paths:
            if len(path) < 3:
                smooth_paths.append(path)
                continue
            points = np.array(path)
            x, y = points[:, 0], points[:, 1]
            valid_indices = [0]
            for i in range(1, len(points)):
                if np.linalg.norm(points[i] - points[valid_indices[-1]]) > 0.01:
                    valid_indices.append(i)
            if len(valid_indices) < 3:
                smooth_paths.append(path)
                continue
            x_clean, y_clean = x[valid_indices], y[valid_indices]
            try:
                tck, u = splprep([x_clean, y_clean], s=self.smoothing_factor)
                approx_len = np.sum(np.sqrt(np.diff(x_clean)**2 + np.diff(y_clean)**2))
                num_points = int(approx_len / self.sample_density_mm)
                if num_points < 2: num_points = 2
                u_new = np.linspace(0, 1, num_points)
                x_new, y_new = splev(u_new, tck)
                smooth_paths.append(list(zip(x_new, y_new)))
            except:
                smooth_paths.append(path)
        return smooth_paths

    # =========================================================
    # 6. JSON EXPORT
    # =========================================================
    def generate_dobot_json(self, paths):
        commands = []
        if not paths: return {"commands": []}
        ordered_paths = []
        remaining = paths[:]
        current_pos = (0, 0)
        while remaining:
            best_idx = 0
            best_dist = float('inf')
            for i, p in enumerate(remaining):
                start_point = p[0]
                d = math.hypot(start_point[0] - current_pos[0], start_point[1] - current_pos[1])
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            next_path = remaining.pop(best_idx)
            ordered_paths.append(next_path)
            current_pos = next_path[-1]

        for path in ordered_paths:
            commands.append({"action": "MOVE", "x": round(path[0][0], 2), "y": round(path[0][1], 2), "pen_state": "UP"})
            commands.append({"action": "PEN_DOWN", "x": round(path[0][0], 2), "y": round(path[0][1], 2), "pen_state": "DOWN"})
            for x, y in path[1:]:
                commands.append({"action": "DRAW", "x": round(x, 2), "y": round(y, 2), "pen_state": "DOWN"})
        return {"commands": commands}

    # =========================================================
    # 7. RUNNER
    # =========================================================
    def run(self, output_dir_name="output_v5"):
        print(f"\n=== HIGH-RES PROCESSING V5: {self.image_path} ===")
        if not os.path.exists(output_dir_name): os.makedirs(output_dir_name)
        
        print("1. 4x Upscaling & Skeletonizing...")
        skel = self.load_and_skeletonize()
        print("2. Tracing...")
        pixel_paths = self.trace_skeleton(skel)
        print("3. Converting...")
        mm_paths = self.pixels_to_mm(pixel_paths)
        print("4. Stitching...")
        stitched = self.stitch_paths(mm_paths)
        print(f"   Paths count: {len(stitched)}")
        print("5. Smoothing...")
        smooth = self.smooth_paths_spline(stitched)

        data = self.generate_dobot_json(smooth)

        json_path = os.path.join(output_dir_name, "robot_commands.json")
        with open(json_path, 'w') as f: json.dump(data, f, indent=2)

        svg_path = os.path.join(output_dir_name, "preview.svg")
        with open(svg_path, "w") as f:
            f.write(f'<svg width="{self.output_width}mm" height="{self.output_height}mm" viewBox="0 0 {self.output_width} {self.output_height}" xmlns="http://www.w3.org/2000/svg">')
            for path in smooth:
                pts = " ".join([f"{x:.2f},{y:.2f}" for x,y in path])
                f.write(f'<polyline points="{pts}" fill="none" stroke="black" stroke-width="0.5"/>')
            f.write('</svg>')

        print(f"✓ DONE! Output saved to {output_dir_name}/")

if __name__ == "__main__":
    converter = ImageToSmoothPath(
        image_path="images/AI/Generated/Promt2_AI_portrait.png",
        output_width=100.0,
        output_height=100.0,

        # TUNING FOR V5
        stitch_threshold_mm=1.0,  # Strict stitching
        smoothing_factor=0.2,     # LOW: Keep sharp details
        sample_density_mm=0.5,    # High res
        min_path_len_mm=2.0       # Keep small details
    )
    converter.run("output_latest_smooth")
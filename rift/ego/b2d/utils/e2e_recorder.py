import shutil
import cv2
import os
import numpy as np
import json
import imageio
from pathlib import Path
from tqdm import trange
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import splprep, splev

FONT_PATH = "/usr/share/fonts/truetype/msttcorefonts/times.ttf"
FONT_SMALL = ImageFont.truetype(FONT_PATH, 20)
FONT_MEDIUM = ImageFont.truetype(FONT_PATH, 26)
FONT_LARGE = ImageFont.truetype(FONT_PATH, 28)
FONT_PLUS = ImageFont.truetype(FONT_PATH, 32)


def resize_and_draw(raw_img, size, key, font=FONT_SMALL, position=(7, 7), color=(0, 0, 0)):
    # resize
    img = Image.fromarray(raw_img).resize(size)
    draw = ImageDraw.Draw(img)

    # Draw the text
    draw.text(position, key, font=font, fill=color[::-1])
    # Convert back to numpy array
    return np.array(img)


class E2ERecorder():
    def __init__(self, save_path, fps=10):
        self.save_path = save_path
        self.fps = fps
        self.cam_size = (400, 300)   # for individual camera views
        self.bev_size = (400, 600)   # for bird’s-eye view
        self.text_color=(0, 0, 0)
        self.command_list = [
            "VOID", "LEFT", "RIGHT", "STRAIGHT",
            "LANE FOLLOW", "CHANGE LANE LEFT", "CHANGE LANE RIGHT"
        ]
        self.coor2topdown = np.array([[5.48993772e+02,  0.00000000e+00, -2.56000000e+02,  1.28000000e+04],
                         [ 0.00000000e+00, -5.48993772e+02, -2.56000000e+02,  1.28000000e+04],
                         [ 0.00000000e+00,  0.00000000e+00, -1.00000000e+00,  5.00000000e+01],
                         [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        self._reset_recorder()

    def _reset_recorder(self):
        self.image_list = []

    def add_image(self, imgs, meta):
        # --- Load camera images ---
        cam_imgs = imgs['imgs']
        views = {
            key: resize_and_draw(img, self.cam_size, key, font=FONT_SMALL)
            for key, img in cam_imgs.items()
        }
        # --- Load BEV image ---
        bev = self.draw_arrowed_traj_bev(np.array(meta['plan']), imgs['bev'], is_ego=True)
        bev = cv2.resize(bev, self.bev_size)

        command = int(meta.get('command', 0))
        command_str = self.command_list[command] if 0 <= command < len(self.command_list) else "UNKNOWN"

        # --- Create all_view layout ---
        top_row = np.hstack([views['CAM_FRONT_LEFT'], views['CAM_FRONT'], views['CAM_FRONT_RIGHT']])
        bottom_row = np.hstack([views['CAM_BACK_LEFT'], views['CAM_BACK'], views['CAM_BACK_RIGHT']])
        all_view = np.vstack([top_row, bottom_row])

        # add all view text
        all_view = self.draw_cam_hud(all_view, meta, font_path=FONT_PATH)
        # add bev text
        bev = self.draw_bev_hud(bev, command_str, font_path=FONT_PATH)
        
        # --- Final layout: [all_view | bev] ---
        final_frame = np.hstack([all_view, bev])

        self.image_list.append(final_frame)

    def save_video(self, video_name):
        if not self.image_list:
            raise RuntimeError("No images to save. Please add images before saving the video.")
        
        # Create a directory for the video if it doesn't exist
        self.save_path.mkdir(parents=True, exist_ok=True)

        # Save the video
        video_path = self.save_path / video_name
        imageio.mimsave(video_path, self.image_list, fps=self.fps, codec='libx264', format='ffmpeg', macro_block_size=1)

        self._reset_recorder()  # Reset the recorder after saving

    def draw_arrowed_traj_bev(self, traj, img, canvas_size=(512, 512), thickness=2, is_ego=False, hue_start=120, hue_end=80, step=8):

        # Step 1: Add ego starting point if needed
        if is_ego:
            traj = np.concatenate([np.zeros((1, 2)), traj], axis=0)

        # Step 2: Homogeneous transformation to top-down (BEV) view
        num_pts = traj.shape[0]
        pts_4d = np.stack([traj[:, 0], traj[:, 1], np.zeros(num_pts), np.ones(num_pts)])
        projected = (self.coor2topdown @ pts_4d).T
        projected[:, :2] /= projected[:, 2:3]

        # Step 3: Filter valid points inside canvas
        x, y = projected[:, 0], projected[:, 1]
        valid_mask = (x > 0) & (x < canvas_size[1]) & (y > 0) & (y < canvas_size[0])
        if not np.any(valid_mask):
            return img

        pts_2d = projected[valid_mask, :2]

        # Step 4: Smooth trajectory using B-spline
        try:
            tck, _ = splprep([pts_2d[:, 0], pts_2d[:, 1]], s=0)
        except:
            return img

        smoothed = np.stack(splev(np.linspace(0, 1, 100), tck), axis=-1).astype(int)

        # Step 5: Draw arrows at regular intervals along the smoothed trajectory
        num_points = len(smoothed)
        for i in range(0, num_points - step, step):
            pt1 = tuple(smoothed[i])
            pt2 = tuple(smoothed[i + step])
            # Ensure points are in bounds
            if all(0 < v < canvas_size[1] for v in (pt1[0], pt2[0])) and all(0 < v < canvas_size[0] for v in (pt1[1], pt2[1])):
                hue = hue_start + (hue_end - hue_start) * (i / num_points)
                hsv_color = np.array([hue, 255, 255], dtype=np.uint8).reshape(1, 1, 3)
                rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR).reshape(-1)
                color = tuple(map(int, rgb_color))
                cv2.arrowedLine(img, pt1, pt2, color=color, thickness=thickness, tipLength=0.2)

        return img
    
    def draw_bev_hud(self, bev_img: np.ndarray, command_str: str,
                             font_path: str, hud_h: int = 35, alpha: float = 0.6) -> np.ndarray:
        """
        Draw a HUD panel on the bottom of the BEV image to show command info.
        Args:
            bev_img: np.ndarray, BEV image (HWC, RGB)
            command_str: string to display (e.g., 'LEFT', 'STRAIGHT')
            font_path: path to .ttf font (e.g., Times New Roman)
            hud_h: height of the bottom HUD strip
            alpha: transparency of the HUD strip
        Returns:
            Modified image with HUD strip at the bottom.
        """
        h, w = bev_img.shape[:2]
        hud = Image.new("RGBA", (w, hud_h), (50, 50, 50, int(alpha * 255)))
        draw = ImageDraw.Draw(hud)

        # Load font
        font = ImageFont.truetype(font_path, 22)
        text = f"Command: {command_str}"
        text_size = draw.textbbox((0, 0), text, font=font)
        text_w = text_size[2] - text_size[0]
        text_h = text_size[3] - text_size[1]

        # Center text
        x = (w - text_w) // 2
        y = (hud_h - text_h) // 2 - 5
        draw.text((x, y), text, font=font, fill=(255, 255, 255))

        # Convert and blend onto BEV image
        hud_np = np.array(hud)
        result = bev_img.copy()
        roi = result[h - hud_h:h, :, :]
        alpha_mask = hud_np[:, :, 3:4] / 255.0
        roi = (roi * (1 - alpha_mask) + hud_np[:, :, :3] * alpha_mask).astype(np.uint8)
        result[h - hud_h:h, :, :] = roi
        return result

    def draw_cam_hud(self, all_view_img: np.ndarray,
                    meta: dict,
                    font_path: str,
                    hud_h: int = 70,
                    alpha: float = 0.6) -> np.ndarray:
        """
        Draw a 2-row HUD inside all_view's bottom area.

        meta keys expected: speed, throttle, brake, steer.
        """
        # ---------- basic sizes ----------
        h, w = all_view_img.shape[:2]
        line_y1, line_y2 = 15, 15 + 30          # y inside the HUD strip
        margin_x = 200                      # left margin for first label
        bar_w, bar_h = 160, 18
        gap_after_bar = 8

        # ---------- font ----------
        font = ImageFont.truetype(font_path, 24)
        white = (255, 255, 255)

        # ---------- create HUD RGBA strip ----------
        hud = Image.new("RGBA", (w, hud_h), (50, 50, 50, int(alpha * 255)))
        draw = ImageDraw.Draw(hud)

        # ---------- helpers ----------
        def draw_bar(label, value, vmin, vmax, x, y, centered=False):
            # label
            draw.text((x, y - 4), f"{label}", font=font, fill=white)
            lbl_w = draw.textbbox((x, y - 4), f"{label}", font=font)[2] - x
            bar_x = x + lbl_w + 10
            # bar border
            draw.rectangle([bar_x, y + 2, bar_x + bar_w, y + bar_h], outline=white, width=2)
            # fill or slider
            if centered:   # steer bar
                cx = bar_x + bar_w // 2
                # center line (optional light grey)
                draw.line([cx, y + 2, cx, y + bar_h], fill=(200, 200, 200), width=1)
                # slider
                ratio = np.clip((value - vmin) / (vmax - vmin), 0.0, 1.0)
                slider_x = bar_x + int(ratio * bar_w)
                draw.rectangle([slider_x - 3, y + 2, slider_x + 3, y + bar_h],
                            fill=white, outline=None)
                # numeric value
                txt_x = bar_x + bar_w + gap_after_bar
                draw.text((txt_x, y - 4), f"{value:.2f}", font=font, fill=white)
            else:          # throttle / brake
                ratio = np.clip((value - vmin) / (vmax - vmin), 0.0, 1.0)
                fill_w = int(ratio * bar_w)
                draw.rectangle([bar_x, y + 2, bar_x + fill_w, y + bar_h], fill=white)
                # numeric value
                txt_x = bar_x + bar_w + gap_after_bar
                draw.text((txt_x, y - 4), f"{value:.2f}", font=font, fill=white)

        # ---------- first row ----------
        speed_txt = f"Speed: {meta.get('speed', 0):.1f} km/h"
        draw.text((margin_x, line_y1 - 10), speed_txt, font=font, fill=white)

        # throttle bar
        draw_bar("Throttle: ", meta.get("throttle", 0.0), 0, 1,
                x=700, y=line_y1 - 6, centered=False)

        # ---------- second row ----------
        draw_bar("Steer: ", meta.get("steer", 0.0), -1, 1,
                x=margin_x, y=line_y2 - 6, centered=True)

        draw_bar("Brake:    ", meta.get("brake", 0.0), 0, 1,
                x=700, y=line_y2 - 6, centered=False)

        # ---------- blend HUD strip onto all_view ----------
        hud_np = np.array(hud)
        bg = all_view_img.copy()
        hud_y0 = h - hud_h
        # alpha blend
        roi = bg[hud_y0:h, :, :]
        alpha_mask = hud_np[:, :, 3:4] / 255.0
        roi = (roi * (1 - alpha_mask) + hud_np[:, :, :3] * alpha_mask).astype(np.uint8)
        bg[hud_y0:h, :, :] = roi
        return bg


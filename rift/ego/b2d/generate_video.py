import cv2
import os
import numpy as np
import json
import imageio
from pathlib import Path
from tqdm import trange
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "/usr/share/fonts/truetype/msttcorefonts/times.ttf"
FONT_SMALL = ImageFont.truetype(FONT_PATH, 20)
FONT_MEDIUM = ImageFont.truetype(FONT_PATH, 26)
FONT_LARGE = ImageFont.truetype(FONT_PATH, 28)
FONT_PLUS = ImageFont.truetype(FONT_PATH, 32)


def load_and_resize(image_path, size):
    img = cv2.imread(image_path)
    if img is None:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), size)

def draw_for_each_camera_img(
    img, 
    text, 
    font,
    position=(7, 7), 
    color=(0, 0, 0)
):
    """
    Draw text at the specified position on the image using a custom TrueType font.

    Parameters:
        img        : OpenCV image (NumPy array, BGR)
        text       : Text string to draw
        position   : (x, y) tuple for the top-left position of the text
        font_path  : Path to the .ttf font file
        font_size  : Font size (default: 24)
        color      : Text color in BGR format (default: black)

    Returns:
        Modified image with text drawn.
    """
    # Convert OpenCV BGR image to PIL RGB image
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # Draw the text (convert BGR to RGB for Pillow)
    draw.text(position, text, font=font, fill=color[::-1])

    # Convert back to OpenCV BGR format
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img

def draw_text_with_custom_font(
    image, 
    text, 
    font,
    color=(255, 255, 255),
    bottom_margin=20
):
    """
    Draws text centered at the bottom of the image using a custom TrueType font.
    """
    # Convert OpenCV image (BGR) to PIL image (RGB)
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)

    # Get text bounding box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Compute position for bottom-centered text
    width, height = image_pil.size
    x = (width - text_width) // 2
    y = height - text_height - bottom_margin

    # Draw text (convert BGR to RGB for Pillow)
    draw.text((x, y), text, font=font, fill=color[::-1])

    # Convert back to OpenCV format (RGB to BGR)
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)


def create_multiview_video(images_folder, fps=10, text_color=(0, 0, 0)):
    camera_dirs = {
        'CAM_FRONT_LEFT': 'rgb_front_left',
        'CAM_FRONT': 'rgb_front',
        'CAM_FRONT_RIGHT': 'rgb_front_right',
        'CAM_BACK_LEFT': 'rgb_back_left',
        'CAM_BACK': 'rgb_back',
        'CAM_BACK_RIGHT': 'rgb_back_right',
        'BEV': 'bev'
    }

    command_list = [
        "VOID", "LEFT", "RIGHT", "STRAIGHT",
        "LANE FOLLOW", "CHANGE LANE LEFT", "CHANGE LANE RIGHT"
    ]

    # Use rgb_front as reference
    ref_dir = os.path.join(images_folder, camera_dirs['CAM_FRONT'])
    images = sorted([img for img in os.listdir(ref_dir) if img.endswith('.jpg') or img.endswith('.png')])
    if not images:
        raise RuntimeError(f"No images found in {ref_dir}")

    cam_size = (400, 300)   # for individual camera views
    bev_size = (400, 600)   # for bird’s-eye view
    frames = []

    for i in trange(len(images)):
        filename = images[i]
        meta_path = os.path.join(images_folder, f'meta/{i:04}.json')

        # --- Load camera images ---
        views = {key: load_and_resize(os.path.join(images_folder, path, filename), cam_size)
                 for key, path in camera_dirs.items() if key != 'BEV'}
        
        for key, img in views.items():
            # Draw text on each camera image
            views[key] = draw_for_each_camera_img(img, key, font=FONT_SMALL)
        
        bev = load_and_resize(os.path.join(images_folder, camera_dirs['BEV'], filename), bev_size)
        bev = draw_for_each_camera_img(bev, 'BEV', font=FONT_MEDIUM)

        # --- Load metadata and overlay text ---
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
        except FileNotFoundError:
            print(f"[WARN] Missing metadata: {meta_path}")
            continue

        steer = float(meta['steer'])
        throttle = float(meta['throttle'])
        brake = float(meta['brake'])
        speed = float(meta['speed'])
        command = int(meta.get('command', 0))
        command_str = command_list[command] if 0 <= command < len(command_list) else "UNKNOWN"

        # --- Create all_view layout ---
        top_row = np.hstack([views['CAM_FRONT_LEFT'], views['CAM_FRONT'], views['CAM_FRONT_RIGHT']])
        bottom_row = np.hstack([views['CAM_BACK_LEFT'], views['CAM_BACK'], views['CAM_BACK_RIGHT']])
        all_view = np.vstack([top_row, bottom_row])

        # add all view text
        all_view_text = f'Speed: {round(speed,2)}   Steer: {round(steer,2)}   Throttle: {round(throttle,2)}   Brake: {round(brake,2)}'
        all_view = draw_text_with_custom_font(all_view, all_view_text, color=text_color, font=FONT_PLUS, bottom_margin=20)
        # add bev text
        bev_text = f'Command: {command_str}'
        bev = draw_text_with_custom_font(bev, bev_text, color=text_color, font=FONT_LARGE, bottom_margin=30)
        
        # --- Final layout: [all_view | bev] ---
        final_frame = np.hstack([all_view, bev])

        frames.append(final_frame)

    # Save video
    save_path = os.path.join(images_folder, 'video.mp4')
    imageio.mimsave(save_path, frames, fps=fps, codec='libx264', format='ffmpeg', macro_block_size=1)
    print(f'[INFO] Video saved to {save_path}')


def main(args):
    root = Path(args.base_images_folder)
    target_paths = root.glob('Town*/route_*')
    for path in target_paths:
        if path.is_dir():
            print(f"Processing: {path}")
            create_multiview_video(path, args.fps)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='argparse')
    parser.add_argument('--base_images_folder','-i', type=str, default='log/eval/vad-standard-rule-seed5')
    parser.add_argument('--fps', type=int, default=5)

    args = parser.parse_args()

    main(args)

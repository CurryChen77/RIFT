import cv2
import os
import numpy as np
import json
import imageio
from pathlib import Path
from tqdm import trange


def load_and_resize(image_path, size):
    img = cv2.imread(image_path)
    if img is None:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), size)

def create_multiview_video(images_folder, fps=10, font_scale=0.8, text_color=(0, 0, 0), text_position=(20, 40)):
    camera_dirs = {
        'front_left': 'rgb_front_left',
        'front': 'rgb_front',
        'front_right': 'rgb_front_right',
        'back_left': 'rgb_back_left',
        'back': 'rgb_back',
        'back_right': 'rgb_back_right',
        'bev': 'bev'
    }

    command_list = [
        "VOID", "LEFT", "RIGHT", "STRAIGHT",
        "LANE FOLLOW", "CHANGE LANE LEFT", "CHANGE LANE RIGHT"
    ]

    # Use rgb_front as reference
    ref_dir = os.path.join(images_folder, camera_dirs['front'])
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
                 for key, path in camera_dirs.items() if key != 'bev'}
        bev = load_and_resize(os.path.join(images_folder, camera_dirs['bev'], filename), bev_size)

        # --- Create all_view layout ---
        top_row = np.hstack([views['front_left'], views['front'], views['front_right']])
        bottom_row = np.hstack([views['back_left'], views['back'], views['back_right']])
        all_view = np.vstack([top_row, bottom_row])

        # --- Final layout: [all_view | bev] ---
        final_frame = np.hstack([all_view, bev])

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

        text = f'speed: {round(speed,2)}, steer: {round(steer,2)}, throttle: {round(throttle,2)}, brake: {round(brake,2)}, command: {command_str}'

        final_frame = cv2.putText(
            final_frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, text_color, 2, cv2.LINE_AA
        )

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

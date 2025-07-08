#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : video_render.py
@Date    : 2024/11/23
'''
import os
from typing import Any, Dict
import imageio
from pathlib import Path

import numpy as np

from rift.gym_carla.visualization.nuplan_scenario_render import NuplanScenarioRender
from rift.scenario.tools.carla_data_provider import CarlaDataProvider

class VideoRender():
    def __init__(self, output_dir, env_params, logger=None):
        self.output_dir = output_dir
        self._frame_rate = CarlaDataProvider.get_frame_rate()
        self.num_scenario = env_params['num_scenario']
        self.image_size = env_params['img_size']
        self.logger = logger

        self._scene_render = NuplanScenarioRender(env_params)
        self.reset_image_dict()

    def reset_image_dict(self):
        self.BEV_image_dict = {}
        self.sensor_image_dict = {}

    def add_BEV_image(self, index, render_data: Dict[str, Any]):
        # render the BEV image
        if index not in self.BEV_image_dict:
            self.BEV_image_dict[index] = []
        BEV_image = self._scene_render.render(**render_data)
        self.BEV_image_dict[index].append(BEV_image)

    def add_sensor_image(self, index, sensor_image):
        if index not in self.sensor_image_dict:
            self.sensor_image_dict[index] = []
        self.sensor_image_dict[index].append(sensor_image)

    def get_BEV_image(self, index):
        return self.BEV_image_dict[index][-1]

    def save_video(self, map_name):
        video_dir = self.output_dir / map_name / "videos"

        Path(video_dir).mkdir(exist_ok=True, parents=True)

        for (index_1, BEV_imgs), (index_2, sensor_imgs) in zip(self.BEV_image_dict.items(), self.sensor_image_dict.items()):
            
            combined_imgs = self.combine_image(BEV_imgs, sensor_imgs)

            assert index_1 == index_2, 'the index of BEV images and sensor images should be the same'

            video_path = video_dir / f"route_{index_1}.mp4"
            # delete the existed file
            if os.path.exists(video_path):
                os.remove(video_path)
                self.logger.log(f">> Existing file deleted: {video_path}", color="red")
            
            imageio.mimsave(video_path, combined_imgs, codec='libx264', format='ffmpeg')
            self.logger.log(f">> video saved to {video_path}", color="yellow")
        
        self.reset_image_dict()
        self._scene_render.clean_up()

    def combine_image(self, BEV_imgs, sensor_imgs, separator_width=10):
        combined_imgs = []
        assert len(BEV_imgs) == len(sensor_imgs), 'the length of BEV images and sensor images should be the same'
        for BEV_img, sensor_img in zip(BEV_imgs, sensor_imgs):
            # combine the sensor image and BEV image
            combined_img = np.hstack((sensor_img, np.rot90(BEV_img, k=1)))
            combined_imgs.append(combined_img)
        return combined_imgs
    

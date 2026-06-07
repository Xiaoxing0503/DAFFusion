from PIL import Image
import torch
from torch.utils.data import Dataset
import os
import random

class PromptDataSet(Dataset):
    def __init__(self, train_vi_noise_path_list,
                 val_vi_noise_path_list,
                 train_over_exposure_path_list,
                 val_over_exposure_path_list,
                 train_ir_low_contrast_path_list,
                 val_ir_low_contrast_path_list,
                 train_ir_noise_path_list,
                 val_ir_noise_path_list,
                 train_ir_stripe_noise_path_list,
                 val_ir_stripe_noise_path_list,
                 train_vi_blur_path_list,
                 val_vi_blur_path_list,
                 train_vi_haze_path_list,
                 val_vi_haze_path_list,
                 train_vi_low_light_path_list,
                 val_vi_low_light_path_list,
                 train_vi_rain_path_list,
                 val_vi_rain_path_list,
                 train_vi_snow_path_list,
                 val_vi_snow_path_list,
                 train_VI_Blur_Haze_path_list,
                 val_VI_Blur_Haze_path_list,
                 train_VI_Blur_Low_path_list,
                 val_VI_Blur_Low_path_list,
                 train_VI_Blur_OverExposure_path_list,
                 val_VI_Blur_OverExposure_path_list,
                 train_VI_Blur_Rain_path_list,
                 val_VI_Blur_Rain_path_list,
                 train_VI_Blur_Snow_path_list,
                 val_VI_Blur_Snow_path_list,
                 train_VI_Haze_Low_path_list,
                 val_VI_Haze_Low_path_list,
                 train_VI_Haze_rain_path_list,
                 val_VI_Haze_rain_path_list,
                 train_VI_Haze_Snow_path_list,
                 val_VI_Haze_Snow_path_list,
                 train_VI_Noise_Haze_path_list,
                 val_VI_Noise_Haze_path_list,
                 train_VI_LowLight_Rain_path_list,
                 val_VI_LowLight_Rain_path_list,
                 train_VI_LowLight_Snow_path_list,
                 val_VI_LowLight_Snow_path_list,
                 train_VI_Noise_Low_path_list,
                 val_VI_Noise_Low_path_list,
                 train_VI_Noise_Rain_path_list,
                 val_VI_Noise_Rain_path_list,
                 train_VI_Noise_Snow_path_list,
                 val_VI_Noise_Snow_path_list,
                 train_VI_OverExposure_Noise_path_list,
                 val_VI_OverExposure_Noise_path_list,
                 train_VI_Haze_Low_Rain_path_list,
                 val_VI_Haze_Low_Rain_path_list,
                 train_VI_Haze_Low_Snow_path_list,
                 val_VI_Haze_Low_Snow_path_list,
                 train_IR_path_list,
                 val_IR_path_list,
                 train_VI_path_list,
                 val_VI_path_list,
                 extra_train_path_lists=None,
                 extra_val_path_lists=None,
                 phase="train", transform=None):
        self.phase = phase
        if phase == "train":
            self.paths = {
                'vi_noise_A': train_vi_noise_path_list[0],
                'vi_noise_B': train_vi_noise_path_list[1],
                'vi_over_exposure_A': train_over_exposure_path_list[0],
                'vi_over_exposure_B': train_over_exposure_path_list[1],
                'ir_low_contrast_A': train_ir_low_contrast_path_list[0],
                'ir_low_contrast_B': train_ir_low_contrast_path_list[1],
                'ir_noise_A': train_ir_noise_path_list[0],
                'ir_noise_B': train_ir_noise_path_list[1],
                'ir_stripe_noise_A': train_ir_stripe_noise_path_list[0],
                'ir_stripe_noise_B': train_ir_stripe_noise_path_list[1],
                'vi_blur_A': train_vi_blur_path_list[0],
                'vi_blur_B': train_vi_blur_path_list[1],
                'vi_haze_A': train_vi_haze_path_list[0],
                'vi_haze_B': train_vi_haze_path_list[1],
                'vi_low_light_A': train_vi_low_light_path_list[0],
                'vi_low_light_B': train_vi_low_light_path_list[1],
                'vi_rain_A': train_vi_rain_path_list[0],
                'vi_rain_B': train_vi_rain_path_list[1],
                'vi_snow_A': train_vi_snow_path_list[0],
                'vi_snow_B': train_vi_snow_path_list[1],
                'vi_Blur_Haze_A': train_VI_Blur_Haze_path_list[0],
                'vi_Blur_Haze_B': train_VI_Blur_Haze_path_list[1],
                'vi_Blur_Low_A': train_VI_Blur_Low_path_list[0],
                'vi_Blur_Low_B': train_VI_Blur_Low_path_list[1],
                'vi_Blur_OverExposure_A': train_VI_Blur_OverExposure_path_list[0],
                'vi_Blur_OverExposure_B': train_VI_Blur_OverExposure_path_list[1],
                'vi_Blur_Rain_A': train_VI_Blur_Rain_path_list[0],
                'vi_Blur_Rain_B': train_VI_Blur_Rain_path_list[1],
                'vi_Blur_Snow_A': train_VI_Blur_Snow_path_list[0],
                'vi_Blur_Snow_B': train_VI_Blur_Snow_path_list[1],
                'vi_Haze_Low_A': train_VI_Haze_Low_path_list[0],
                'vi_Haze_Low_B': train_VI_Haze_Low_path_list[1],
                'vi_Haze_rain_A': train_VI_Haze_rain_path_list[0],
                'vi_Haze_rain_B': train_VI_Haze_rain_path_list[1],
                'vi_Haze_Snow_A': train_VI_Haze_Snow_path_list[0],
                'vi_Haze_Snow_B': train_VI_Haze_Snow_path_list[1],
                'vi_Noise_Haze_A': train_VI_Noise_Haze_path_list[0],
                'vi_Noise_Haze_B': train_VI_Noise_Haze_path_list[1],
                'vi_LowLight_Rain_A': train_VI_LowLight_Rain_path_list[0],
                'vi_LowLight_Rain_B': train_VI_LowLight_Rain_path_list[1],
                'vi_LowLight_Snow_A': train_VI_LowLight_Snow_path_list[0],
                'vi_LowLight_Snow_B': train_VI_LowLight_Snow_path_list[1],
                'vi_Noise_Low_A': train_VI_Noise_Low_path_list[0],
                'vi_Noise_Low_B': train_VI_Noise_Low_path_list[1],
                'vi_Noise_Rain_A': train_VI_Noise_Rain_path_list[0],
                'vi_Noise_Rain_B': train_VI_Noise_Rain_path_list[1],
                'vi_Noise_Snow_A': train_VI_Noise_Snow_path_list[0],
                'vi_Noise_Snow_B': train_VI_Noise_Snow_path_list[1],
                'vi_OverExposure_Noise_A': train_VI_OverExposure_Noise_path_list[0],
                'vi_OverExposure_Noise_B': train_VI_OverExposure_Noise_path_list[1],
                'vi_Haze_Low_Rain_A': train_VI_Haze_Low_Rain_path_list[0],
                'vi_Haze_Low_Rain_B': train_VI_Haze_Low_Rain_path_list[1],
                'vi_Haze_Low_Snow_A': train_VI_Haze_Low_Snow_path_list[0],
                'vi_Haze_Low_Snow_B': train_VI_Haze_Low_Snow_path_list[1],
                'ir_A': train_IR_path_list[0],
                'ir_B': train_IR_path_list[1],
                'vi_A': train_VI_path_list[0],
                'vi_B': train_VI_path_list[1],
            }
            self.paths_gt = {
                'vi_noise_A_gt': train_vi_noise_path_list[2],
                'vi_noise_B_gt': train_vi_noise_path_list[3],
                'vi_over_exposure_A_gt': train_over_exposure_path_list[2],
                'vi_over_exposure_B_gt': train_over_exposure_path_list[3],
                'ir_low_contrast_A_gt': train_ir_low_contrast_path_list[2],
                'ir_low_contrast_B_gt': train_ir_low_contrast_path_list[3],
                'ir_noise_A_gt': train_ir_noise_path_list[2],
                'ir_noise_B_gt': train_ir_noise_path_list[3],
                'ir_stripe_noise_A_gt': train_ir_stripe_noise_path_list[2],
                'ir_stripe_noise_B_gt': train_ir_stripe_noise_path_list[3],
                'vi_blur_A_gt': train_vi_blur_path_list[2],
                'vi_blur_B_gt': train_vi_blur_path_list[3],
                'vi_haze_A_gt': train_vi_haze_path_list[2],
                'vi_haze_B_gt': train_vi_haze_path_list[3],
                'vi_low_light_A_gt': train_vi_low_light_path_list[2],
                'vi_low_light_B_gt': train_vi_low_light_path_list[3],
                'vi_rain_A_gt': train_vi_rain_path_list[2],
                'vi_rain_B_gt': train_vi_rain_path_list[3],
                'vi_snow_A_gt': train_vi_snow_path_list[2],
                'vi_snow_B_gt': train_vi_snow_path_list[3],
                'vi_Blur_Haze_A_gt': train_VI_Blur_Haze_path_list[2],
                'vi_Blur_Haze_B_gt': train_VI_Blur_Haze_path_list[3],
                'vi_Blur_Low_A_gt': train_VI_Blur_Low_path_list[2],
                'vi_Blur_Low_B_gt': train_VI_Blur_Low_path_list[3],
                'vi_Blur_OverExposure_A_gt': train_VI_Blur_OverExposure_path_list[2],
                'vi_Blur_OverExposure_B_gt': train_VI_Blur_OverExposure_path_list[3],
                'vi_Blur_Rain_A_gt': train_VI_Blur_Rain_path_list[2],
                'vi_Blur_Rain_B_gt': train_VI_Blur_Rain_path_list[3],
                'vi_Blur_Snow_A_gt': train_VI_Blur_Snow_path_list[2],
                'vi_Blur_Snow_B_gt': train_VI_Blur_Snow_path_list[3],
                'vi_Haze_Low_A_gt': train_VI_Haze_Low_path_list[2],
                'vi_Haze_Low_B_gt': train_VI_Haze_Low_path_list[3],
                'vi_Haze_rain_A_gt': train_VI_Haze_rain_path_list[2],
                'vi_Haze_rain_B_gt': train_VI_Haze_rain_path_list[3],
                'vi_Haze_Snow_A_gt': train_VI_Haze_Snow_path_list[2],
                'vi_Haze_Snow_B_gt': train_VI_Haze_Snow_path_list[3],
                'vi_Noise_Haze_A_gt': train_VI_Noise_Haze_path_list[2],
                'vi_Noise_Haze_B_gt': train_VI_Noise_Haze_path_list[3],
                'vi_LowLight_Rain_A_gt': train_VI_LowLight_Rain_path_list[2],
                'vi_LowLight_Rain_B_gt': train_VI_LowLight_Rain_path_list[3],
                'vi_LowLight_Snow_A_gt': train_VI_LowLight_Snow_path_list[2],
                'vi_LowLight_Snow_B_gt': train_VI_LowLight_Snow_path_list[3],
                'vi_Noise_Low_A_gt': train_VI_Noise_Low_path_list[2],
                'vi_Noise_Low_B_gt': train_VI_Noise_Low_path_list[3],
                'vi_Noise_Rain_A_gt': train_VI_Noise_Rain_path_list[2],
                'vi_Noise_Rain_B_gt': train_VI_Noise_Rain_path_list[3],
                'vi_Noise_Snow_A_gt': train_VI_Noise_Snow_path_list[2],
                'vi_Noise_Snow_B_gt': train_VI_Noise_Snow_path_list[3],
                'vi_OverExposure_Noise_A_gt': train_VI_OverExposure_Noise_path_list[2],
                'vi_OverExposure_Noise_B_gt': train_VI_OverExposure_Noise_path_list[3],
                'vi_Haze_Low_Rain_A_gt': train_VI_Haze_Low_Rain_path_list[2],
                'vi_Haze_Low_Rain_B_gt': train_VI_Haze_Low_Rain_path_list[3],
                'vi_Haze_Low_Snow_A_gt': train_VI_Haze_Low_Snow_path_list[2],
                'vi_Haze_Low_Snow_B_gt': train_VI_Haze_Low_Snow_path_list[3],
                'ir_A_gt': train_IR_path_list[2],
                'ir_B_gt': train_IR_path_list[3],
                'vi_A_gt': train_VI_path_list[2],
                'vi_B_gt': train_VI_path_list[3],
            }
            if extra_train_path_lists is not None:
                for task_name, path_list in extra_train_path_lists.items():
                    self.paths.update({
                        task_name + '_A': path_list[0],
                        task_name + '_B': path_list[1],
                    })
                    self.paths_gt.update({
                        task_name + '_A_gt': path_list[2],
                        task_name + '_B_gt': path_list[3],
                    })
        else:
            self.paths = {
                'vi_noise_A': val_vi_noise_path_list[0],
                'vi_noise_B': val_vi_noise_path_list[1],
                'vi_over_exposure_A': val_over_exposure_path_list[0],
                'vi_over_exposure_B': val_over_exposure_path_list[1],
                'ir_low_contrast_A': val_ir_low_contrast_path_list[0],
                'ir_low_contrast_B': val_ir_low_contrast_path_list[1],
                'ir_noise_A': val_ir_noise_path_list[0],
                'ir_noise_B': val_ir_noise_path_list[1],
                'ir_stripe_noise_A': val_ir_stripe_noise_path_list[0],
                'ir_stripe_noise_B': val_ir_stripe_noise_path_list[1],
                'vi_blur_A': val_vi_blur_path_list[0],
                'vi_blur_B': val_vi_blur_path_list[1],
                'vi_haze_A': val_vi_haze_path_list[0],
                'vi_haze_B': val_vi_haze_path_list[1],
                'vi_low_light_A': val_vi_low_light_path_list[0],
                'vi_low_light_B': val_vi_low_light_path_list[1],
                'vi_rain_A': val_vi_rain_path_list[0],
                'vi_rain_B': val_vi_rain_path_list[1],
                'vi_snow_A': val_vi_snow_path_list[0],
                'vi_snow_B': val_vi_snow_path_list[1],
                'vi_Blur_Haze_A': val_VI_Blur_Haze_path_list[0],
                'vi_Blur_Haze_B': val_VI_Blur_Haze_path_list[1],
                'vi_Blur_Low_A': val_VI_Blur_Low_path_list[0],
                'vi_Blur_Low_B': val_VI_Blur_Low_path_list[1],
                'vi_Blur_OverExposure_A': val_VI_Blur_OverExposure_path_list[0],
                'vi_Blur_OverExposure_B': val_VI_Blur_OverExposure_path_list[1],
                'vi_Blur_Rain_A': val_VI_Blur_Rain_path_list[0],
                'vi_Blur_Rain_B': val_VI_Blur_Rain_path_list[1],
                'vi_Blur_Snow_A': val_VI_Blur_Snow_path_list[0],
                'vi_Blur_Snow_B': val_VI_Blur_Snow_path_list[1],
                'vi_Haze_Low_A': val_VI_Haze_Low_path_list[0],
                'vi_Haze_Low_B': val_VI_Haze_Low_path_list[1],
                'vi_Haze_rain_A': val_VI_Haze_rain_path_list[0],
                'vi_Haze_rain_B': val_VI_Haze_rain_path_list[1],
                'vi_Haze_Snow_A': val_VI_Haze_Snow_path_list[0],
                'vi_Haze_Snow_B': val_VI_Haze_Snow_path_list[1],
                'vi_Noise_Haze_A': val_VI_Noise_Haze_path_list[0],
                'vi_Noise_Haze_B': val_VI_Noise_Haze_path_list[1],
                'vi_LowLight_Rain_A': val_VI_LowLight_Rain_path_list[0],
                'vi_LowLight_Rain_B': val_VI_LowLight_Rain_path_list[1],
                'vi_LowLight_Snow_A': val_VI_LowLight_Snow_path_list[0],
                'vi_LowLight_Snow_B': val_VI_LowLight_Snow_path_list[1],
                'vi_Noise_Low_A': val_VI_Noise_Low_path_list[0],
                'vi_Noise_Low_B': val_VI_Noise_Low_path_list[1],
                'vi_Noise_Rain_A': val_VI_Noise_Rain_path_list[0],
                'vi_Noise_Rain_B': val_VI_Noise_Rain_path_list[1],
                'vi_Noise_Snow_A': val_VI_Noise_Snow_path_list[0],
                'vi_Noise_Snow_B': val_VI_Noise_Snow_path_list[1],
                'vi_OverExposure_Noise_A': val_VI_OverExposure_Noise_path_list[0],
                'vi_OverExposure_Noise_B': val_VI_OverExposure_Noise_path_list[1],
                'vi_Haze_Low_Rain_A': val_VI_Haze_Low_Rain_path_list[0],
                'vi_Haze_Low_Rain_B': val_VI_Haze_Low_Rain_path_list[1],
                'vi_Haze_Low_Snow_A': val_VI_Haze_Low_Snow_path_list[0],
                'vi_Haze_Low_Snow_B': val_VI_Haze_Low_Snow_path_list[1],
                'ir_A': val_IR_path_list[0],
                'ir_B': val_IR_path_list[1],
                'vi_A': val_VI_path_list[0],
                'vi_B': val_VI_path_list[1],
            }
            self.paths_gt = {
                'vi_noise_A_gt': val_vi_noise_path_list[0],
                'vi_noise_B_gt': val_vi_noise_path_list[1],
                'vi_over_exposure_A_gt': val_over_exposure_path_list[0],
                'vi_over_exposure_B_gt': val_over_exposure_path_list[1],
                'ir_low_contrast_A_gt': val_ir_low_contrast_path_list[0],
                'ir_low_contrast_B_gt': val_ir_low_contrast_path_list[1],
                'ir_noise_A_gt': val_ir_noise_path_list[0],
                'ir_noise_B_gt': val_ir_noise_path_list[1],
                'ir_stripe_noise_A_gt': val_ir_stripe_noise_path_list[0],
                'ir_stripe_noise_B_gt': val_ir_stripe_noise_path_list[1],
                'vi_blur_A_gt': val_vi_blur_path_list[0],
                'vi_blur_B_gt': val_vi_blur_path_list[1],
                'vi_haze_A_gt': val_vi_haze_path_list[0],
                'vi_haze_B_gt': val_vi_haze_path_list[1],
                'vi_low_light_A_gt': val_vi_low_light_path_list[0],
                'vi_low_light_B_gt': val_vi_low_light_path_list[1],
                'vi_rain_A_gt': val_vi_rain_path_list[0],
                'vi_rain_B_gt': val_vi_rain_path_list[1],
                'vi_snow_A_gt': val_vi_snow_path_list[0],
                'vi_snow_B_gt': val_vi_snow_path_list[1],
                'vi_Blur_Haze_A_gt': val_VI_Blur_Haze_path_list[0],
                'vi_Blur_Haze_B_gt': val_VI_Blur_Haze_path_list[1],
                'vi_Blur_Low_A_gt': val_VI_Blur_Low_path_list[0],
                'vi_Blur_Low_B_gt': val_VI_Blur_Low_path_list[1],
                'vi_Blur_OverExposure_A_gt': val_VI_Blur_OverExposure_path_list[0],
                'vi_Blur_OverExposure_B_gt': val_VI_Blur_OverExposure_path_list[1],
                'vi_Blur_Rain_A_gt': val_VI_Blur_Rain_path_list[0],
                'vi_Blur_Rain_B_gt': val_VI_Blur_Rain_path_list[1],
                'vi_Blur_Snow_A_gt': val_VI_Blur_Snow_path_list[0],
                'vi_Blur_Snow_B_gt': val_VI_Blur_Snow_path_list[1],
                'vi_Haze_Low_A_gt': val_VI_Haze_Low_path_list[0],
                'vi_Haze_Low_B_gt': val_VI_Haze_Low_path_list[1],
                'vi_Haze_rain_A_gt': val_VI_Haze_rain_path_list[0],
                'vi_Haze_rain_B_gt': val_VI_Haze_rain_path_list[1],
                'vi_Haze_Snow_A_gt': val_VI_Haze_Snow_path_list[0],
                'vi_Haze_Snow_B_gt': val_VI_Haze_Snow_path_list[1],
                'vi_Noise_Haze_A_gt': val_VI_Noise_Haze_path_list[0],
                'vi_Noise_Haze_B_gt': val_VI_Noise_Haze_path_list[1],
                'vi_LowLight_Rain_A_gt': val_VI_LowLight_Rain_path_list[0],
                'vi_LowLight_Rain_B_gt': val_VI_LowLight_Rain_path_list[1],
                'vi_LowLight_Snow_A_gt': val_VI_LowLight_Snow_path_list[0],
                'vi_LowLight_Snow_B_gt': val_VI_LowLight_Snow_path_list[1],
                'vi_Noise_Low_A_gt': val_VI_Noise_Low_path_list[0],
                'vi_Noise_Low_B_gt': val_VI_Noise_Low_path_list[1],
                'vi_Noise_Rain_A_gt': val_VI_Noise_Rain_path_list[0],
                'vi_Noise_Rain_B_gt': val_VI_Noise_Rain_path_list[1],
                'vi_Noise_Snow_A_gt': val_VI_Noise_Snow_path_list[0],
                'vi_Noise_Snow_B_gt': val_VI_Noise_Snow_path_list[1],
                'vi_OverExposure_Noise_A_gt': val_VI_OverExposure_Noise_path_list[0],
                'vi_OverExposure_Noise_B_gt': val_VI_OverExposure_Noise_path_list[1],
                'vi_Haze_Low_Rain_A_gt': val_VI_Haze_Low_Rain_path_list[0],
                'vi_Haze_Low_Rain_B_gt': val_VI_Haze_Low_Rain_path_list[1],
                'vi_Haze_Low_Snow_A_gt': val_VI_Haze_Low_Snow_path_list[0],
                'vi_Haze_Low_Snow_B_gt': val_VI_Haze_Low_Snow_path_list[1],
                'ir_A_gt': val_IR_path_list[0],
                'ir_B_gt': val_IR_path_list[1],
                'vi_A_gt': val_VI_path_list[0],
                'vi_B_gt': val_VI_path_list[1],
            }
            if extra_val_path_lists is not None:
                for task_name, path_list in extra_val_path_lists.items():
                    self.paths.update({
                        task_name + '_A': path_list[0],
                        task_name + '_B': path_list[1],
                    })
                    self.paths_gt.update({
                        task_name + '_A_gt': path_list[0],
                        task_name + '_B_gt': path_list[1],
                    })
        self.transform = transform

        # Create a list to hold all sample indices grouped by class
        self.class_indices = {}
        for class_key, paths in self.paths.items():
            self.class_indices[class_key] = list(range(len(paths)))
        pass

    def __len__(self):
        if self.phase == "train":
            return sum(len(paths) for paths in self.paths.values())
        else:
            # Return the part number of images in val all classes and subsets
            #return sum(len(paths) for paths in self.paths.values()) // 4
            return 80

    def __getitem__(self, item):
        # Randomly select a class, use the random sampling (equal to sequential sampling when the number of sampling is large)
        class_key = random.choice(list(self.paths.keys()))

        # Randomly select an index for the chosen class
        class_indices = self.class_indices[class_key]
        item_index = random.randint(0, len(class_indices) - 1)
        image_index = class_indices[item_index]

        # Load the A and B images based on the class and index
        image_A_path = self.paths[class_key[:-2] + '_A'][image_index]
        image_B_path = self.paths[class_key[:-2] + '_B'][image_index]

        image_A_gt_path = self.paths_gt[class_key[:-2] + '_A_gt'][image_index]
        image_B_gt_path = self.paths_gt[class_key[:-2] + '_B_gt'][image_index]

        image_A = Image.open(image_A_path).convert(mode='RGB')
        image_B = Image.open(image_B_path).convert(mode='RGB')
        image_A_gt = Image.open(image_A_gt_path).convert(mode='RGB')
        image_B_gt = Image.open(image_B_gt_path).convert(mode='RGB')

        image_full = image_A
        # Apply any specified transformations
        if self.transform is not None:
            image_A, image_B, image_A_gt, image_B_gt, image_full = self.transform(image_A, image_B, image_A_gt, image_B_gt, image_full)

        name = image_A_path.replace("\\", "/").split("/")[-1].split(".")[0]

        return image_A, image_B, image_A_gt, image_B_gt, image_full, class_key[:-2], name

    @staticmethod
    def collate_fn(batch):
        images_A, images_B, images_A_gt, images_B_gt, images_full, class_keys, name = zip(*batch)
        images_A = torch.stack(images_A, dim=0)
        images_B = torch.stack(images_B, dim=0)
        images_A_gt = torch.stack(images_A_gt, dim=0)
        images_B_gt = torch.stack(images_B_gt, dim=0)
        images_full = torch.stack(images_full, dim=0)
        return images_A, images_B, images_A_gt, images_B_gt, images_full, class_keys, name

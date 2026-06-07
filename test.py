from torchvision.transforms import functional as F
from model.DAFFusion import DAFFusion as create_model
from model.Adapter_both import Adapter as both
from model.Adapter_one import Adapter as one
from model.clip import DA_adapter
import torchvision.transforms as transforms

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
from PIL import Image
import cv2
import clip
import torch
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    # --- Configuration Section ---
    # All parameters are defined here instead of using command-line arguments.
    dataset_path = '/data1/xml/data/'
    weights_path = '/weights/checkpoint_lastest.pth'
    save_path = ''
    device_name = 'cuda'
    gpu_id = '0'

    


    # Set the visible GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    supported = [".jpg", ".JPG", ".png", ".PNG", ".bmp", 'tif', 'TIF']
    
    visible_root = os.path.join(dataset_path, "vi/")
    infrared_root = os.path.join(dataset_path, "ir/")

    visible_path = [os.path.join(visible_root, i) for i in os.listdir(visible_root)
                    if os.path.splitext(i)[-1] in supported]
    infrared_path = [os.path.join(infrared_root, i) for i in os.listdir(infrared_root)
                     if os.path.splitext(i)[-1] in supported]

    visible_path.sort()
    infrared_path.sort()

    print(f"Found {len(visible_path)} visible images and {len(infrared_path)} infrared images.")
    assert len(visible_path) == len(infrared_path), "The number of source images does not match!"

    print("Begin to run!")
    with torch.no_grad():
        # model_clip, _ = clip.load("ViT-L/14@336px", device=device)
        model_clip, _ = clip.load("ViT-B/32", device=device)
        for param in model_clip.parameters():
                param.requires_grad = False
        model_degrad = DA_adapter(model_clip).to(device)
        model_degrad.set_frozen()
        checkpoint_de = torch.load('/data1/xml/ControlFusion0127/experiments/ControlFusion_train_stage1_20260517-120951/weights/checkpointdegradprior_lastest.pth', map_location=device)
        model_degrad.load_state_dict(checkpoint_de['model'])
        model = create_model().to(device)

        model.load_state_dict(torch.load(weights_path, map_location=device)['model'])
        model.eval()

    # ================== Inference and Timing ==================
    total_time = 0.0
    num_samples = len(visible_path)
    if num_samples == 0:
        print("No images to process. Exiting.")
        return

    # Create CUDA events for accurate timing on GPU
    if device.type == 'cuda':
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    for i in range(num_samples):
        ir_path = infrared_path[i]
        vi_path = visible_path[i]
        img_name = os.path.basename(vi_path)

        ir = Image.open(ir_path).convert(mode="RGB")
        vi = Image.open(vi_path).convert(mode="RGB")

        # Using original size
        ir_tensor = F.to_tensor(ir).unsqueeze(0).to(device)
        vi_tensor = F.to_tensor(vi).unsqueeze(0).to(device)

        with torch.no_grad():
            # text = clip.tokenize(input_text).to(device)

            # --- Start Timer ---
            if device.type == 'cuda':
                starter.record()
            else:
                start_time = time.time()

            resize_transform = transforms.Resize(size=(224, 224), interpolation=transforms.InterpolationMode.BILINEAR)

            I_A_de_input = resize_transform(vi_tensor)
            I_B_de_input = resize_transform(ir_tensor)

            degrad_features_A = model_degrad.get_image_features(I_A_de_input)
            degrad_features_B = model_degrad.get_image_features(I_B_de_input)
                
            fused_tensor = model(vi_tensor, ir_tensor, degrad_features_A, degrad_features_B)

            # --- End Timer ---
            if device.type == 'cuda':
                ender.record()
                torch.cuda.synchronize()  # Wait for GPU to finish
                elapsed_time = starter.elapsed_time(ender) / 1000.0  # Convert to seconds
            else:
                end_time = time.time()
                elapsed_time = end_time - start_time

            total_time += elapsed_time

            # --- Process and Save Output ---
            fused_img_Y = tensor2numpy(fused_tensor)
            save_pic(fused_img_Y, save_path, img_name)

        print(f"Processed and saved {img_name} | Time: {elapsed_time:.4f}s")

    # --- Final Average Time Calculation ---
    if num_samples > 0:
        avg_time = total_time / num_samples
        print("\n-------------------------------------------")
        print(f"Average inference time: {avg_time:.4f} seconds")
        print("-------------------------------------------\n")

    print(f"Finish! The results are saved in {save_path}")


def tensor2numpy(img_tensor):
    img = img_tensor.squeeze(0).cpu().detach().numpy()
    img = np.transpose(img, [1, 2, 0])
    return img


def save_pic(outputpic, path, index: str):
    # Clip and convert to uint8 format
    outputpic = np.clip(outputpic * 255, 0, 255).astype(np.uint8)
    # Convert from RGB (used by PyTorch/PIL) to BGR (used by OpenCV)
    outputpic_bgr = cv2.cvtColor(outputpic, cv2.COLOR_RGB2BGR)
    save_path = os.path.join(path, os.path.splitext(index)[0] + ".png")
    cv2.imwrite(save_path, outputpic_bgr)


if __name__ == '__main__':
    main()

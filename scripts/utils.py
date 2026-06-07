import os
import sys
import random
import clip
import torch.nn.functional as F
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import cv2

from scripts.losses import fusion_prompt_loss
import torchvision.transforms as transforms
from scripts.losses import fusion_prompt_loss


def read_data(root: str):
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    train_root = os.path.join(root, "train")
    val_root = os.path.join(root, "test")
    assert os.path.exists(train_root), "train root: {} does not exist.".format(train_root)
    assert os.path.exists(val_root), "val root: {} does not exist.".format(val_root)

    train_images_visible_path = []
    train_images_infrared_path = []
    train_images_visible_gt_path = []
    train_images_infrared_gt_path = []
    val_images_visible_path = []
    val_images_infrared_path = []

    supported = [".jpg", ".JPG", ".png", ".PNG", ".bmp", 'tif', 'TIF']  # 支持的文件后缀类型

    train_visible_root = os.path.join(train_root, "Visible")
    train_infrared_root= os.path.join(train_root, "Infrared")

    train_visible_gt_root = os.path.join(train_root, "Visible_gt")
    train_infrared_gt_root= os.path.join(train_root, "Infrared_gt")

    val_visible_root = os.path.join(val_root, "Visible")
    val_infrared_root = os.path.join(val_root, "Infrared")

    train_visible_path = [os.path.join(train_visible_root, i) for i in os.listdir(train_visible_root)
                  if os.path.splitext(i)[-1] in supported]
    train_infrared_path = [os.path.join(train_infrared_root, i) for i in os.listdir(train_infrared_root)
                  if os.path.splitext(i)[-1] in supported]

    train_visible_gt_path = [os.path.join(train_visible_gt_root, i) for i in os.listdir(train_visible_gt_root)
                  if os.path.splitext(i)[-1] in supported]
    train_infrared_gt_path = [os.path.join(train_infrared_gt_root, i) for i in os.listdir(train_infrared_gt_root)
                  if os.path.splitext(i)[-1] in supported]

    val_visible_path = [os.path.join(val_visible_root, i) for i in os.listdir(val_visible_root)
                  if os.path.splitext(i)[-1] in supported]
    val_infrared_path = [os.path.join(val_infrared_root, i) for i in os.listdir(val_infrared_root)
                  if os.path.splitext(i)[-1] in supported]

    train_visible_path.sort()
    train_infrared_path.sort()
    train_visible_gt_path.sort()
    train_infrared_gt_path.sort()
    val_visible_path.sort()
    val_infrared_path.sort()

    assert len(train_visible_path) == len(train_infrared_path),' The length of train dataset does not match. low:{}, high:{}'.\
                                         format(len(train_visible_path),len(train_infrared_path))
    assert len(val_visible_path) == len(val_infrared_path),' The length of val dataset does not match. low:{}, high:{}'.\
                                          format(len(val_visible_path),len(val_infrared_path))
    print("Visible and Infrared images check finish")

    for index in range(len(train_visible_path)):
        img_visible_path=train_visible_path[index]
        img_infrared_path=train_infrared_path[index]
        train_images_visible_path.append(img_visible_path)
        train_images_infrared_path.append(img_infrared_path)

        img_visible_gt_path=train_visible_gt_path[index]
        img_infrared_gt_path=train_infrared_gt_path[index]
        train_images_visible_gt_path.append(img_visible_gt_path)
        train_images_infrared_gt_path.append(img_infrared_gt_path)

    for index in range(len(val_visible_path)):
        img_visible_path=val_visible_path[index]
        img_infrared_path=val_infrared_path[index]
        val_images_visible_path.append(img_visible_path)
        val_images_infrared_path.append(img_infrared_path)

    total_dataset_nums = len(train_visible_path) + len(train_infrared_path) + len(train_visible_gt_path) + len(train_infrared_gt_path) \
                         + len(val_visible_path) + len(val_infrared_path)
    print("{} images were found in the dataset.".format(total_dataset_nums))
    print("{} visible images for training.".format(len(train_visible_path)))
    print("{} infrared images for training.".format(len(train_infrared_path)))
    print("{} visible gt images for training.".format(len(train_visible_gt_path)))
    print("{} infrared gt images for training.".format(len(train_infrared_gt_path)))
    print("{} visible images for validation.".format(len(val_visible_path)))
    print("{} infrared images for validation.\n".format(len(val_infrared_path)))

    train_low_light_path_list = [train_visible_path, train_infrared_path, train_visible_gt_path, train_infrared_gt_path]
    val_low_light_path_list = [val_visible_path, val_infrared_path]
    return train_low_light_path_list, val_low_light_path_list


ATOMIC_DEGRADATIONS = [
    "Blur",
    "Haze",
    "Low light",
    "Rain",
    "Snow",
    "Noise",
    "Stripe noise",
    "Over exposure",
    "Low contrast",
]

ATOMIC2ID = {name: i for i, name in enumerate(ATOMIC_DEGRADATIONS)}
NUM_ATOMICS = len(ATOMIC_DEGRADATIONS)


def build_multi_hot_from_task(task_name, atomic2id):
    K = len(atomic2id)
    target = torch.zeros(K, dtype=torch.float32)

    def add(label):
        target[atomic2id[label]] = 1.0

    if task_name == "ir":
        use_A = False
    elif task_name == "vi":
        use_A = True
        
    elif task_name == "ir_low_contrast":
        add("Low contrast")
        use_A = False
    elif task_name == "ir_noise":
        add("Noise")
        use_A = False
    elif task_name == "ir_stripe_noise":
        add("Stripe noise")
        use_A = False

    elif task_name == "vi_noise":
        add("Noise")
        use_A = True
    elif task_name == "vi_over_exposure":
        add("Over exposure")
        use_A = True
    elif task_name == "vi_blur":
        add("Blur")
        use_A = True
    elif task_name == "vi_haze":
        add("Haze")
        use_A = True
    elif task_name == "vi_low_light":
        add("Low light")
        use_A = True
    elif task_name == "vi_rain":
        add("Rain")
        use_A = True
    elif task_name == "vi_snow":
        add("Snow")
        use_A = True

    elif task_name == "vi_Blur_Haze":
        add("Blur"); add("Haze")
        use_A = True
    elif task_name == "vi_Blur_Low":
        add("Blur"); add("Low light")
        use_A = True
    elif task_name == "vi_Blur_OverExposure":
        add("Blur"); add("Over exposure")
        use_A = True
    elif task_name == "vi_Blur_Rain":
        add("Blur"); add("Rain")
        use_A = True
    elif task_name == "vi_Blur_Snow":
        add("Blur"); add("Snow")
        use_A = True
    elif task_name == "vi_Haze_Low":
        add("Haze"); add("Low light")
        use_A = True
    elif task_name == "vi_Haze_rain":
        add("Haze"); add("Rain")
        use_A = True
    elif task_name == "vi_Haze_Snow":
        add("Haze"); add("Snow")
        use_A = True
    elif task_name == "vi_Noise_Haze":
        add("Noise"); add("Haze")
        use_A = True
    elif task_name == "vi_LowLight_Rain":
        add("Low light"); add("Rain")
        use_A = True
    elif task_name == "vi_LowLight_Snow":
        add("Low light"); add("Snow")
        use_A = True
    elif task_name == "vi_Noise_Low":
        add("Noise"); add("Low light")
        use_A = True
    elif task_name == "vi_Noise_Rain":
        add("Noise"); add("Rain")
        use_A = True
    elif task_name == "vi_Noise_Snow":
        add("Noise"); add("Snow")
        use_A = True
    elif task_name == "vi_OverExposure_Noise":
        add("Over exposure"); add("Noise")
        use_A = True

    elif task_name == "vi_Haze_Low_Rain":
        add("Haze"); add("Low light"); add("Rain")
        use_A = True
    elif task_name == "vi_Haze_Low_Snow":
        add("Haze"); add("Low light"); add("Snow")
        use_A = True

    else:
        raise ValueError(f"Unknown task: {task_name}")

    return target, use_A

def train_one_epoch_1(model, criterion, optimizer, lr_scheduler, data_loader, device, epoch):
    model.train()

    if torch.cuda.is_available():
        criterion = criterion.to(device)

    accu_total_loss = torch.zeros(1).to(device)


    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)

    atomic_text_tokens = clip.tokenize(ATOMIC_DEGRADATIONS).to(device)

    for step, data in enumerate(data_loader):
        I_A, I_B, _, _, _, task, _ = data
        if torch.cuda.is_available():
            I_A = I_A.to(device, non_blocking=True)
            I_B = I_B.to(device, non_blocking=True)

        B = len(task)
        image_list = []
        multi_hot_targets = []

        # 为每个样本构造：退化图像 + multi-hot标签
        for i in range(B):
            target_i, use_A = build_multi_hot_from_task(task[i], ATOMIC2ID)
            multi_hot_targets.append(target_i)

            if use_A:
                image_list.append(I_A[i])
            else:
                image_list.append(I_B[i])

        Image = torch.stack(image_list, dim=0).to(device)                 # [B, C, H, W]
        multi_hot_targets = torch.stack(multi_hot_targets, dim=0).to(device)

        image_features = model.get_image_features(Image)                 # [B, D]
        text_prototypes = model.get_text_features(atomic_text_tokens)    # [K, D]

        loss, logits = criterion(image_features, text_prototypes, multi_hot_targets)
        ##########################################################################
        # loss, loss_ssim, loss_max, loss_color, loss_text = loss_function_prompt(I_A_gt, I_B_gt, I_fused, task)

        loss.backward()

        accu_total_loss += loss.detach()
        # accu_ssim_loss += loss_ssim.detach()
        # accu_max_loss += loss_max.detach()
        # accu_color_loss += loss_color.detach()
        # accu_text_loss += loss_text


        lr = optimizer.param_groups[0]["lr"]

        data_loader.desc = "[train epoch {}] loss: {:.3f}  lr: {:.6f}".format(epoch, accu_total_loss.item() / (step + 1), lr)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    return accu_total_loss.item() / (step + 1), lr

def train_one_epoch(model,model_degrad, optimizer, lr_scheduler, data_loader, device, epoch):
    model.train()
    model_degrad.eval()
    loss_function_prompt = fusion_prompt_loss()

    if torch.cuda.is_available():
        loss_function_prompt = loss_function_prompt.to(device)

    accu_total_loss = torch.zeros(1).to(device)
    accu_ssim_loss = torch.zeros(1).to(device)
    accu_max_loss = torch.zeros(1).to(device)
    accu_color_loss = torch.zeros(1).to(device)
    accu_grad_loss = torch.zeros(1).to(device)


    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        I_A, I_B, I_A_gt, I_B_gt, _, task, _ = data
        text_line = []


        if torch.cuda.is_available():
            I_A = I_A.to(device)
            I_B = I_B.to(device)
            I_A_gt = I_A_gt.to(device)
            I_B_gt = I_B_gt.to(device)

        ##########################################################################
        resize_transform = transforms.Resize(size=(224, 224), interpolation=transforms.InterpolationMode.BILINEAR)

        I_A_de_input = resize_transform(I_A)
        I_B_de_input = resize_transform(I_B)

        degrad_features_A = model_degrad.get_image_features(I_A_de_input)
        degrad_features_B = model_degrad.get_image_features(I_B_de_input)
        ##########################################################################
        I_fused = model(I_A, I_B,degrad_features_A,degrad_features_B)

        ##########################################################################
        loss, loss_ssim, loss_max, loss_color, loss_grad = loss_function_prompt(I_A_gt, I_B_gt, I_fused, task)

        loss.backward()

        accu_total_loss += loss.detach()
        accu_ssim_loss += loss_ssim.detach()
        accu_max_loss += loss_max.detach()
        accu_color_loss += loss_color.detach()
        accu_grad_loss += loss_grad.detach()


        lr = optimizer.param_groups[0]["lr"]

        data_loader.desc = "[train epoch {}] loss: {:.3f}  ssim loss: {:.3f}  max loss: {:.3f}  color loss: {:.3f}  grad loss: {:.3f}  lr: {:.6f}".format(epoch, accu_total_loss.item() / (step + 1),
            accu_ssim_loss.item() / (step + 1), accu_max_loss.item() / (step + 1), accu_color_loss.item() / (step + 1), accu_grad_loss.item() / (step + 1), lr)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    return accu_total_loss.item() / (step + 1), accu_ssim_loss.item() / (step + 1), accu_max_loss.item() / (step + 1), accu_color_loss.item() / (step + 1), accu_grad_loss.item() / (step + 1), lr


@torch.no_grad()
def evaluate(model,model_degrad, data_loader, device, epoch, lr, filefold_path):
    loss_function_prompt = fusion_prompt_loss()

    model.eval()
    model_degrad.eval()
    accu_total_loss = torch.zeros(1).to(device)
    accu_ssim_loss = torch.zeros(1).to(device)
    accu_max_loss = torch.zeros(1).to(device)
    accu_color_loss = torch.zeros(1).to(device)
    accu_text_loss = torch.zeros(1).to(device)
    save_epoch = 1
    save_length = 60
    cnt = 0
    save_RGB_fuse = True

    if torch.cuda.is_available():
        loss_function_prompt = loss_function_prompt.to(device)
    
    if epoch % save_epoch == 0:
        evalfold_path = os.path.join(filefold_path, str(epoch))
        if os.path.exists(evalfold_path) is False:
            os.makedirs(evalfold_path)

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        I_A, I_B, I_A_gt, I_B_gt, I_full, task, name = data
       

        if torch.cuda.is_available():
            I_A = I_A.to(device)
            I_B = I_B.to(device)
            I_A_gt = I_A_gt.to(device)
            I_B_gt = I_B_gt.to(device)
            I_full = I_full.to(device)

        resize_transform = transforms.Resize(size=(224, 224), interpolation=transforms.InterpolationMode.BILINEAR)

        I_A_de_input = resize_transform(I_A)
        I_B_de_input = resize_transform(I_B)

        degrad_features_A = model_degrad.get_image_features(I_A_de_input)
        degrad_features_B = model_degrad.get_image_features(I_B_de_input)
        ##########################################################################
        I_fused = model(I_A, I_B,degrad_features_A,degrad_features_B)


        if epoch % save_epoch == 0:
            if cnt <= save_length:
                fused_img_Y = tensor2numpy(I_fused)
                img_full = tensor2numpy(I_full)
                img_ir = tensor2numpy(I_B_gt)
                save_pic(fused_img_Y, evalfold_path, str(name[0]))
                if save_RGB_fuse == True:
                    save_pic(img_full, evalfold_path, str(name[0]) + "vis")
                    save_pic(img_ir, evalfold_path, str(name[0]) + "ir")
                cnt += 1

        loss, loss_ssim, loss_max, loss_color, loss_text = loss_function_prompt(I_A_gt, I_B_gt, I_fused, task)

        accu_total_loss += loss
        accu_ssim_loss += loss_ssim.detach()
        accu_max_loss += loss_max.detach()
        accu_color_loss += loss_color.detach()
        accu_text_loss += loss_text

        data_loader.desc = "[val epoch {}] loss: {:.3f}  ssim loss: {:.3f}  max loss: {:.3f}  color loss: {:.3f}  text loss: {:.3f}  lr: {:.6f}".format(epoch, accu_total_loss.item() / (step + 1),
            accu_ssim_loss.item() / (step + 1), accu_max_loss.item() / (step + 1), accu_color_loss.item() / (step + 1), accu_text_loss.item() / (step + 1), lr)

    return accu_total_loss.item() / (step + 1), accu_ssim_loss.item() / (step + 1), accu_max_loss.item() / (step + 1), accu_color_loss.item() / (step + 1), accu_text_loss.item() / (step + 1)

def mergy_Y_RGB_to_YCbCr(img1, img2):
    Y_channel = img1.squeeze(0).cpu().numpy()
    Y_channel = np.transpose(Y_channel, [1, 2, 0])

    img2 = img2.squeeze(0).cpu().numpy()
    img2 = np.transpose(img2, [1, 2, 0])

    img2_YCbCr = cv2.cvtColor(img2, cv2.COLOR_RGB2YCrCb)
    CbCr_channels = img2_YCbCr[:, :, 1:]
    merged_img_YCbCr = np.concatenate((Y_channel, CbCr_channels), axis=2)
    merged_img = cv2.cvtColor(merged_img_YCbCr, cv2.COLOR_YCrCb2RGB)
    return merged_img

def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

def save_pic(outputpic, path, index : str):
    outputpic[outputpic > 1.] = 1
    outputpic[outputpic < 0.] = 0
    outputpic = cv2.UMat(outputpic).get()
    outputpic = cv2.normalize(outputpic, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
    outputpic=outputpic[:, :, ::-1]
    save_path = os.path.join(path, index + ".png")
    cv2.imwrite(save_path, outputpic)

def show_img(images,imagesl, B):
    for index in range(B):
        img = images[index, :]
        img_np = np.array(img.permute(1, 2, 0).detach().cpu())
        plt.figure(1)
        plt.imshow(img_np)
        img = imagesl[index, :]

        img_np = np.array(img.permute(1, 2, 0).detach().cpu())
        plt.figure(2)
        plt.imshow(img_np)
        plt.show(block=True)

def tensor2numpy(R_tensor):
    R = R_tensor.squeeze(0).cpu().detach().numpy()
    R = np.transpose(R, [1, 2, 0])
    return R

def tensor2numpy_single(L_tensor):
    L = L_tensor.squeeze(0)
    L_3 = torch.cat([L, L, L], dim=0)
    L_3 = L_3.cpu().detach().numpy()
    L_3 = np.transpose(L_3, [1, 2, 0])
    return L_3

# def merge_and_print_paths(
#     train_vi_noise_slight_path_list, val_vi_noise_slight_path_list,
#     train_vi_noise_moderate_path_list, val_vi_noise_moderate_path_list,
#     train_vi_noise_average_path_list, val_vi_noise_average_path_list,
#     train_vi_noise_serious_path_list, val_vi_noise_serious_path_list,
#     train_over_exposure_slight_path_list, val_over_exposure_slight_path_list,
#     train_over_exposure_moderate_path_list, val_over_exposure_moderate_path_list,
#     train_over_exposure_average_path_list, val_over_exposure_average_path_list,
#     train_over_exposure_serious_path_list, val_over_exposure_serious_path_list,
#     train_vi_blur_slight_path_list, val_vi_blur_slight_path_list,
#     train_vi_blur_moderate_path_list, val_vi_blur_moderate_path_list,
#     train_vi_blur_average_path_list, val_vi_blur_average_path_list,
#     train_vi_blur_serious_path_list, val_vi_blur_serious_path_list,
#     train_vi_haze_slight_path_list, val_vi_haze_slight_path_list,
#     train_vi_haze_moderate_path_list, val_vi_haze_moderate_path_list,
#     train_vi_haze_average_path_list, val_vi_haze_average_path_list,
#     train_vi_haze_serious_path_list, val_vi_haze_serious_path_list,
#     train_vi_low_light_slight_path_list, val_vi_low_light_slight_path_list,
#     train_vi_low_light_moderate_path_list, val_vi_low_light_moderate_path_list,
#     train_vi_low_light_average_path_list, val_vi_low_light_average_path_list,
#     train_vi_low_light_serious_path_list, val_vi_low_light_serious_path_list,
#     train_vi_rain_slight_path_list, val_vi_rain_slight_path_list,
#     train_vi_rain_moderate_path_list, val_vi_rain_moderate_path_list,
#     train_vi_rain_average_path_list, val_vi_rain_average_path_list,
#     train_vi_rain_serious_path_list, val_vi_rain_serious_path_list,
#     train_ir_low_contrast_slight_path_list, val_ir_low_contrast_slight_path_list,
#     train_ir_low_contrast_moderate_path_list, val_ir_low_contrast_moderate_path_list,
#     train_ir_low_contrast_average_path_list, val_ir_low_contrast_average_path_list,
#     train_ir_low_contrast_serious_path_list, val_ir_low_contrast_serious_path_list,
#     train_ir_noise_slight_path_list, val_ir_noise_slight_path_list,
#     train_ir_noise_moderate_path_list, val_ir_noise_moderate_path_list,
#     train_ir_noise_average_path_list, val_ir_noise_average_path_list,
#     train_ir_noise_serious_path_list, val_ir_noise_serious_path_list,
#     train_ir_stripe_noise_slight_path_list, val_ir_stripe_noise_slight_path_list,
#     train_ir_stripe_noise_moderate_path_list, val_ir_stripe_noise_moderate_path_list,
#     train_ir_stripe_noise_average_path_list, val_ir_stripe_noise_average_path_list,
#     train_ir_stripe_noise_serious_path_list, val_ir_stripe_noise_serious_path_list
# ):
#     # Combine paths for Visible and Infrared (VI) noise
#     vi_noise_train_path_list = [
#         train_vi_noise_slight_path_list,
#         train_vi_noise_moderate_path_list,
#         train_vi_noise_average_path_list,
#         train_vi_noise_serious_path_list
#     ]
#     vi_noise_val_path_list = [
#         val_vi_noise_slight_path_list,
#         val_vi_noise_moderate_path_list,
#         val_vi_noise_average_path_list,
#         val_vi_noise_serious_path_list
#     ]

#     # Combine paths for Over-Exposure
#     over_exposure_train_path_list = [
#         train_over_exposure_slight_path_list,
#         train_over_exposure_moderate_path_list,
#         train_over_exposure_average_path_list,
#         train_over_exposure_serious_path_list
#     ]
#     over_exposure_val_path_list = [
#         val_over_exposure_slight_path_list,
#         val_over_exposure_moderate_path_list,
#         val_over_exposure_average_path_list,
#         val_over_exposure_serious_path_list
#     ]

#     # Combine paths for blur
#     vi_blur_train_path_list = [
#         train_vi_blur_slight_path_list,
#         train_vi_blur_moderate_path_list,
#         train_vi_blur_average_path_list,
#         train_vi_blur_serious_path_list,
#     ]
#     vi_blur_val_path_list = [
#         val_vi_blur_slight_path_list,
#         val_vi_blur_moderate_path_list,
#         val_vi_blur_average_path_list,
#         val_vi_blur_serious_path_list,
#     ]

#     # Combine paths for haze
#     vi_haze_train_path_list = [
#         train_vi_haze_slight_path_list,
#         train_vi_haze_moderate_path_list,
#         train_vi_haze_average_path_list,
#         train_vi_haze_serious_path_list,
#     ]
#     vi_haze_val_path_list = [
#         val_vi_haze_slight_path_list,
#         val_vi_haze_moderate_path_list,
#         val_vi_haze_average_path_list,
#         val_vi_haze_serious_path_list,
#     ]

#     # Combine paths for low light
#     vi_low_light_train_path_list = [
#         train_vi_low_light_slight_path_list,
#         train_vi_low_light_moderate_path_list,
#         train_vi_low_light_average_path_list,
#         train_vi_low_light_serious_path_list,
#     ]
#     vi_low_light_val_path_list = [
#         val_vi_low_light_slight_path_list,
#         val_vi_low_light_moderate_path_list,
#         val_vi_low_light_average_path_list,
#         val_vi_low_light_serious_path_list,
#     ]

#     # Combine paths for rain
#     vi_rain_train_path_list = [
#         train_vi_rain_slight_path_list,
#         train_vi_rain_moderate_path_list,
#         train_vi_rain_average_path_list,
#         train_vi_rain_serious_path_list,
#     ]
#     vi_rain_val_path_list = [
#         val_vi_rain_slight_path_list,
#         val_vi_rain_moderate_path_list,
#         val_vi_rain_average_path_list,
#         val_vi_rain_serious_path_list,
#     ]

#     # Combine paths for IR Low Contrast
#     ir_low_contrast_train_path_list = [
#         train_ir_low_contrast_slight_path_list,
#         train_ir_low_contrast_moderate_path_list,
#         train_ir_low_contrast_average_path_list,
#         train_ir_low_contrast_serious_path_list
#     ]
#     ir_low_contrast_val_path_list = [
#         val_ir_low_contrast_slight_path_list,
#         val_ir_low_contrast_moderate_path_list,
#         val_ir_low_contrast_average_path_list,
#         val_ir_low_contrast_serious_path_list
#     ]

#     # Combine paths for IR Noise
#     ir_noise_train_path_list = [
#         train_ir_noise_slight_path_list,
#         train_ir_noise_moderate_path_list,
#         train_ir_noise_average_path_list,
#         train_ir_noise_serious_path_list
#     ]
#     ir_noise_val_path_list = [
#         val_ir_noise_slight_path_list,
#         val_ir_noise_moderate_path_list,
#         val_ir_noise_average_path_list,
#         val_ir_noise_serious_path_list
#     ]

#     # Combine paths for IR Stripe Noise
#     ir_stripe_noise_train_path_list = [
#         train_ir_stripe_noise_slight_path_list,
#         train_ir_stripe_noise_moderate_path_list,
#         train_ir_stripe_noise_average_path_list,
#         train_ir_stripe_noise_serious_path_list
#     ]
#     ir_stripe_noise_val_path_list = [
#         val_ir_stripe_noise_slight_path_list,
#         val_ir_stripe_noise_moderate_path_list,
#         val_ir_stripe_noise_average_path_list,
#         val_ir_stripe_noise_serious_path_list
#     ]

#     return (
#         vi_noise_train_path_list, vi_noise_val_path_list,
#         over_exposure_train_path_list, over_exposure_val_path_list,
#         vi_blur_train_path_list, vi_blur_val_path_list,
#         vi_haze_train_path_list, vi_haze_val_path_list,
#         vi_low_light_train_path_list, vi_low_light_val_path_list,
#         vi_rain_train_path_list, vi_rain_val_path_list,
#         ir_low_contrast_train_path_list, ir_low_contrast_val_path_list,
#         ir_noise_train_path_list, ir_noise_val_path_list,
#         ir_stripe_noise_train_path_list, ir_stripe_noise_val_path_list
#     )

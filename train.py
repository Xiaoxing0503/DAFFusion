import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import clip
from data.prompt_dataset import PromptDataSet

from model.clip import DA_adapter

from model.DAFFusion import DAFFusion as create_model
from scripts.utils import read_data, train_one_epoch, evaluate, create_lr_scheduler
import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import transforms as T

VI_DEGRADATION_FOLDERS = [
    "VI_Blur",
    "VI_Blur_Haze",
    "VI_Blur_Low",
    "VI_Blur_OverExposure",
    "VI_Blur_Rain",
    "VI_Blur_Snow",
    "VI_Haze",
    "VI_Haze_Low",
    "VI_Haze_Low_Rain",
    "VI_Haze_Low_Snow",
    "VI_Haze_rain",
    "VI_Haze_Snow",
    "VI_Low_light",
    "VI_LowLight_Rain",
    "VI_LowLight_Snow",
    "VI_Noise",
    "VI_Noise_Haze",
    "VI_Noise_Low",
    "VI_Noise_Rain",
    "VI_Noise_Snow",
    "VI_Over_exposure",
    "VI_OverExposure_Noise",
    "VI_Rain",
    "VI_Snow",
]

IR_DEGRADATION_FOLDERS = [
    "IR_Low_contrast",
    "IR_Noise",
    "IR_Stripe_noise",
]


def load_extra_datasets(root):
    extra_train_path_lists = {}
    extra_val_path_lists = {}

    for vi_folder in VI_DEGRADATION_FOLDERS:
        for ir_folder in IR_DEGRADATION_FOLDERS:
            folder_name = "{}_{}".format(vi_folder, ir_folder)
            dataset_path = os.path.join(root, folder_name)
            print("Loading IVF Fusion and {} Task!".format(folder_name))
            train_path_list, val_path_list = read_data(dataset_path)
            task_name = "vi" + folder_name[2:]
            extra_train_path_lists[task_name] = train_path_list
            extra_val_path_lists[task_name] = val_path_list

    return extra_train_path_lists, extra_val_path_lists


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./experiments") is False:
        os.makedirs("./experiments")

    file_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filefold_path = "./experiments/DAFFusion_train_{}".format(file_name)
    os.makedirs(filefold_path)
    file_img_path = os.path.join(filefold_path, "img")
    os.makedirs(file_img_path)
    file_weights_path = os.path.join(filefold_path, "weights")
    os.makedirs(file_weights_path)
    file_log_path = os.path.join(filefold_path, "log")
    os.makedirs(file_log_path)

    tb_writer = SummaryWriter(log_dir=file_log_path)

    best_val_loss = 1e5
    start_epoch = 0

    print("Loading IVF Fusion and ir_low_contrast Task!")
    if args.ir_low_contrast_average_path is not None:
        train_ir_low_contrast_path_list, val_ir_low_contrast_path_list = read_data(args.ir_low_contrast_average_path)
    else:
        train_ir_low_contrast_path_list = val_ir_low_contrast_path_list = None

    print("Loading IVF Fusion and ir_noise Task!")
    if args.ir_noise_average_path is not None:
        train_ir_noise_path_list, val_ir_noise_path_list = read_data(args.ir_noise_average_path)
    else:
        train_ir_noise_path_list = val_ir_noise_path_list = None

    print("Loading IVF Fusion and ir_stripe_noise Task!")
    if args.ir_stripe_noise_average_path is not None:
        train_ir_stripe_noise_path_list, val_ir_stripe_noise_path_list = read_data(args.ir_stripe_noise_average_path)
    else:
        train_ir_stripe_noise_path_list = val_ir_stripe_noise_path_list = None

    print("Loading IVF Fusion and noise Task!")
    if args.vi_noise_average_path is not None:
        train_vi_noise_path_list, val_vi_noise_path_list = read_data(args.vi_noise_average_path)
    else:
        train_vi_noise_path_list = val_vi_noise_path_list = None

    print("Loading IVF Fusion and Over-Exposure Task!")
    if args.over_exposure_average_path is not None:
        train_over_exposure_path_list, val_over_exposure_path_list = read_data(args.over_exposure_average_path)
    else:
        train_over_exposure_path_list = val_over_exposure_path_list = None

    print("Loading IVF Fusion and Blur Task!")
    if args.vi_blur_average_path is not None:
        train_vi_blur_path_list, val_vi_blur_path_list = read_data(args.vi_blur_average_path)
    else:
        train_vi_blur_path_list = val_vi_blur_path_list = None
    
    print("Loading IVF Fusion and Haze Task!")
    if args.vi_haze_average_path is not None:
        train_vi_haze_path_list, val_vi_haze_path_list = read_data(args.vi_haze_average_path)
    else:
        train_vi_haze_path_list = val_vi_haze_path_list = None
    
    print("Loading IVF Fusion and Rain Task!")
    if args.vi_rain_average_path is not None:
        train_vi_rain_path_list, val_vi_rain_path_list = read_data(args.vi_rain_average_path)
    else:
        train_vi_rain_path_list = val_vi_rain_path_list = None

    print("Loading IVF Fusion and Low Light Task!")
    if args.vi_low_light_average_path is not None:
        train_vi_low_light_path_list, val_vi_low_light_path_list = read_data(args.vi_low_light_average_path)
    else:
        train_vi_low_light_path_list = val_vi_low_light_path_list = None

    print("Loading IVF Fusion and Snow Task!")
    if args.VI_Snow_path is not None:
        train_vi_snow_path_list, val_vi_snow_path_list = read_data(args.VI_Snow_path)
    else:
        train_vi_snow_path_list = val_vi_snow_path_list = None

    print("Loading IVF Fusion and Blur Haze Task!")
    if args.VI_Blur_Haze_path is not None:
        train_VI_Blur_Haze_path_list, val_VI_Blur_Haze_path_list = read_data(args.VI_Blur_Haze_path)
    else:
        train_VI_Blur_Haze_path_list = val_VI_Blur_Haze_path_list = None

    print("Loading IVF Fusion and Blur Low Task!")
    if args.VI_Blur_Low_path is not None:
        train_VI_Blur_Low_path_list, val_VI_Blur_Low_path_list = read_data(args.VI_Blur_Low_path)
    else:
        train_VI_Blur_Low_path_list = val_VI_Blur_Low_path_list = None

    print("Loading IVF Fusion and Blur OverExposure Task!")
    if args.VI_Blur_OverExposure_path is not None:
        train_VI_Blur_OverExposure_path_list, val_VI_Blur_OverExposure_path_list = read_data(args.VI_Blur_OverExposure_path)
    else:
        train_VI_Blur_OverExposure_path_list = val_VI_Blur_OverExposure_path_list = None

    print("Loading IVF Fusion and Blur Rain Task!")
    if args.VI_Blur_Rain_path is not None:
        train_VI_Blur_Rain_path_list, val_VI_Blur_Rain_path_list = read_data(args.VI_Blur_Rain_path)
    else:
        train_VI_Blur_Rain_path_list = val_VI_Blur_Rain_path_list = None

    print("Loading IVF Fusion and Blur Snow Task!")
    if args.VI_Blur_Snow_path is not None:
        train_VI_Blur_Snow_path_list, val_VI_Blur_Snow_path_list = read_data(args.VI_Blur_Snow_path)
    else:
        train_VI_Blur_Snow_path_list = val_VI_Blur_Snow_path_list = None

    print("Loading IVF Fusion and Haze Low Task!")
    if args.VI_Haze_Low_path is not None:
        train_VI_Haze_Low_path_list, val_VI_Haze_Low_path_list = read_data(args.VI_Haze_Low_path)
    else:
        train_VI_Haze_Low_path_list = val_VI_Haze_Low_path_list = None

    print("Loading IVF Fusion and Haze rain Task!")
    if args.VI_Haze_rain_path is not None:
        train_VI_Haze_rain_path_list, val_VI_Haze_rain_path_list = read_data(args.VI_Haze_rain_path)
    else:
        train_VI_Haze_rain_path_list = val_VI_Haze_rain_path_list = None

    print("Loading IVF Fusion and Haze Snow Task!")
    if args.VI_Haze_Snow_path is not None:
        train_VI_Haze_Snow_path_list, val_VI_Haze_Snow_path_list = read_data(args.VI_Haze_Snow_path)
    else:
        train_VI_Haze_Snow_path_list = val_VI_Haze_Snow_path_list = None

    print("Loading IVF Fusion and Noise Haze Task!")
    if args.VI_Noise_Haze_path is not None:
        train_VI_Noise_Haze_path_list, val_VI_Noise_Haze_path_list = read_data(args.VI_Noise_Haze_path)
    else:
        train_VI_Noise_Haze_path_list = val_VI_Noise_Haze_path_list = None

    print("Loading IVF Fusion and LowLight Rain Task!")
    if args.VI_LowLight_Rain_path is not None:
        train_VI_LowLight_Rain_path_list, val_VI_LowLight_Rain_path_list = read_data(args.VI_LowLight_Rain_path)
    else:
        train_VI_LowLight_Rain_path_list = val_VI_LowLight_Rain_path_list = None

    print("Loading IVF Fusion and LowLight Snow Task!")
    if args.VI_LowLight_Snow_path is not None:
        train_VI_LowLight_Snow_path_list, val_VI_LowLight_Snow_path_list = read_data(args.VI_LowLight_Snow_path)
    else:
        train_VI_LowLight_Snow_path_list = val_VI_LowLight_Snow_path_list = None

    print("Loading IVF Fusion and Noise Low Task!")
    if args.VI_Noise_Low_path is not None:
        train_VI_Noise_Low_path_list, val_VI_Noise_Low_path_list = read_data(args.VI_Noise_Low_path)
    else:
        train_VI_Noise_Low_path_list = val_VI_Noise_Low_path_list = None

    print("Loading IVF Fusion and Noise Rain Task!")
    if args.VI_Noise_Rain_path is not None:
        train_VI_Noise_Rain_path_list, val_VI_Noise_Rain_path_list = read_data(args.VI_Noise_Rain_path)
    else:
        train_VI_Noise_Rain_path_list = val_VI_Noise_Rain_path_list = None

    print("Loading IVF Fusion and Noise Snow Task!")
    if args.VI_Noise_Snow_path is not None:
        train_VI_Noise_Snow_path_list, val_VI_Noise_Snow_path_list = read_data(args.VI_Noise_Snow_path)
    else:
        train_VI_Noise_Snow_path_list = val_VI_Noise_Snow_path_list = None

    print("Loading IVF Fusion and OverExposure Noise Task!")
    if args.VI_OverExposure_Noise_path is not None:
        train_VI_OverExposure_Noise_path_list, val_VI_OverExposure_Noise_path_list = read_data(args.VI_OverExposure_Noise_path)
    else:
        train_VI_OverExposure_Noise_path_list = val_VI_OverExposure_Noise_path_list = None

    print("Loading IVF Fusion and Haze Low Rain Task!")
    if args.VI_Haze_Low_Rain_path is not None:
        train_VI_Haze_Low_Rain_path_list, val_VI_Haze_Low_Rain_path_list = read_data(args.VI_Haze_Low_Rain_path)
    else:
        train_VI_Haze_Low_Rain_path_list = val_VI_Haze_Low_Rain_path_list = None

    print("Loading IVF Fusion and Haze Low Snow Task!")
    if args.VI_Haze_Low_Snow_path is not None:
        train_VI_Haze_Low_Snow_path_list, val_VI_Haze_Low_Snow_path_list = read_data(args.VI_Haze_Low_Snow_path)
    else:
        train_VI_Haze_Low_Snow_path_list = val_VI_Haze_Low_Snow_path_list = None

    print("Loading IVF Fusion and Clean IR Task!")
    if args.IR_path is not None:
        train_IR_path_list, val_IR_path_list = read_data(args.IR_path)
    else:
        train_IR_path_list = val_IR_path_list = None

    print("Loading IVF Fusion and Clean VI Task!")
    if args.VI_path is not None:
        train_VI_path_list, val_VI_path_list = read_data(args.VI_path)
    else:
        train_VI_path_list = val_VI_path_list = None

    if args.extra_dataset_root is not None:
        extra_train_path_lists, extra_val_path_lists = load_extra_datasets(args.extra_dataset_root)
    else:
        extra_train_path_lists = extra_val_path_lists = None

    data_transform = {
        "train": T.Compose([T.RandomCrop(128),
                            T.RandomHorizontalFlip(0.5),
                            T.RandomVerticalFlip(0.5),
                            T.ToTensor()]),

        "val": T.Compose([T.Resize_16(),
                          T.ToTensor()])}

    train_dataset = PromptDataSet(train_vi_noise_path_list=train_vi_noise_path_list,
                                  val_vi_noise_path_list=val_vi_noise_path_list,
                                  train_over_exposure_path_list=train_over_exposure_path_list,
                                  val_over_exposure_path_list=val_over_exposure_path_list,
                                  train_ir_low_contrast_path_list=train_ir_low_contrast_path_list,
                                  val_ir_low_contrast_path_list=val_ir_low_contrast_path_list,
                                  train_ir_noise_path_list=train_ir_noise_path_list,
                                  val_ir_noise_path_list=val_ir_noise_path_list,
                                  train_ir_stripe_noise_path_list = train_ir_stripe_noise_path_list,
                                  val_ir_stripe_noise_path_list = val_ir_stripe_noise_path_list,
                                  train_vi_blur_path_list=train_vi_blur_path_list,
                                  val_vi_blur_path_list=val_vi_blur_path_list,
                                  train_vi_haze_path_list=train_vi_haze_path_list,
                                  val_vi_haze_path_list=val_vi_haze_path_list,
                                  train_vi_low_light_path_list=train_vi_low_light_path_list,
                                  val_vi_low_light_path_list=val_vi_low_light_path_list,
                                  train_vi_rain_path_list=train_vi_rain_path_list,
                                  val_vi_rain_path_list=val_vi_rain_path_list,
                                  train_vi_snow_path_list=train_vi_snow_path_list,
                                  val_vi_snow_path_list=val_vi_snow_path_list,
                                  train_VI_Blur_Haze_path_list=train_VI_Blur_Haze_path_list,
                                  val_VI_Blur_Haze_path_list=val_VI_Blur_Haze_path_list,
                                  train_VI_Blur_Low_path_list=train_VI_Blur_Low_path_list,
                                  val_VI_Blur_Low_path_list=val_VI_Blur_Low_path_list,
                                  train_VI_Blur_OverExposure_path_list=train_VI_Blur_OverExposure_path_list,
                                  val_VI_Blur_OverExposure_path_list=val_VI_Blur_OverExposure_path_list,
                                  train_VI_Blur_Rain_path_list=train_VI_Blur_Rain_path_list,
                                  val_VI_Blur_Rain_path_list=val_VI_Blur_Rain_path_list,
                                  train_VI_Blur_Snow_path_list=train_VI_Blur_Snow_path_list,
                                  val_VI_Blur_Snow_path_list=val_VI_Blur_Snow_path_list,
                                  train_VI_Haze_Low_path_list=train_VI_Haze_Low_path_list,
                                  val_VI_Haze_Low_path_list=val_VI_Haze_Low_path_list,
                                  train_VI_Haze_rain_path_list=train_VI_Haze_rain_path_list,
                                  val_VI_Haze_rain_path_list=val_VI_Haze_rain_path_list,
                                  train_VI_Haze_Snow_path_list=train_VI_Haze_Snow_path_list,
                                  val_VI_Haze_Snow_path_list=val_VI_Haze_Snow_path_list,
                                  train_VI_Noise_Haze_path_list=train_VI_Noise_Haze_path_list,
                                  val_VI_Noise_Haze_path_list=val_VI_Noise_Haze_path_list,
                                  train_VI_LowLight_Rain_path_list=train_VI_LowLight_Rain_path_list,
                                  val_VI_LowLight_Rain_path_list=val_VI_LowLight_Rain_path_list,
                                  train_VI_LowLight_Snow_path_list=train_VI_LowLight_Snow_path_list,
                                  val_VI_LowLight_Snow_path_list=val_VI_LowLight_Snow_path_list,
                                  train_VI_Noise_Low_path_list=train_VI_Noise_Low_path_list,
                                  val_VI_Noise_Low_path_list=val_VI_Noise_Low_path_list,
                                  train_VI_Noise_Rain_path_list=train_VI_Noise_Rain_path_list,
                                  val_VI_Noise_Rain_path_list=val_VI_Noise_Rain_path_list,
                                  train_VI_Noise_Snow_path_list=train_VI_Noise_Snow_path_list,
                                  val_VI_Noise_Snow_path_list=val_VI_Noise_Snow_path_list,
                                  train_VI_OverExposure_Noise_path_list=train_VI_OverExposure_Noise_path_list,
                                  val_VI_OverExposure_Noise_path_list=val_VI_OverExposure_Noise_path_list,
                                  train_VI_Haze_Low_Rain_path_list=train_VI_Haze_Low_Rain_path_list,
                                  val_VI_Haze_Low_Rain_path_list=val_VI_Haze_Low_Rain_path_list,
                                  train_VI_Haze_Low_Snow_path_list=train_VI_Haze_Low_Snow_path_list,
                                  val_VI_Haze_Low_Snow_path_list=val_VI_Haze_Low_Snow_path_list,
                                  train_IR_path_list=train_IR_path_list,
                                  val_IR_path_list=val_IR_path_list,
                                  train_VI_path_list=train_VI_path_list,
                                  val_VI_path_list=val_VI_path_list,
                                  extra_train_path_lists=extra_train_path_lists,
                                  extra_val_path_lists=extra_val_path_lists,
                                  phase="train",
                              transform=data_transform["train"])

    val_dataset = PromptDataSet(train_vi_noise_path_list=train_vi_noise_path_list,
                                  val_vi_noise_path_list=val_vi_noise_path_list,
                                  train_over_exposure_path_list=train_over_exposure_path_list,
                                  val_over_exposure_path_list=val_over_exposure_path_list,
                                  train_ir_low_contrast_path_list=train_ir_low_contrast_path_list,
                                  val_ir_low_contrast_path_list=val_ir_low_contrast_path_list,
                                  train_ir_noise_path_list=train_ir_noise_path_list,
                                  val_ir_noise_path_list=val_ir_noise_path_list,
                                  train_ir_stripe_noise_path_list = train_ir_stripe_noise_path_list,
                                  val_ir_stripe_noise_path_list = val_ir_stripe_noise_path_list,
                                  train_vi_blur_path_list=train_vi_blur_path_list,
                                  val_vi_blur_path_list=val_vi_blur_path_list,
                                  train_vi_haze_path_list=train_vi_haze_path_list,
                                  val_vi_haze_path_list=val_vi_haze_path_list,
                                  train_vi_low_light_path_list=train_vi_low_light_path_list,
                                  val_vi_low_light_path_list=val_vi_low_light_path_list,
                                  train_vi_rain_path_list=train_vi_rain_path_list,
                                  val_vi_rain_path_list=val_vi_rain_path_list,
                                  train_vi_snow_path_list=train_vi_snow_path_list,
                                  val_vi_snow_path_list=val_vi_snow_path_list,
                                  train_VI_Blur_Haze_path_list=train_VI_Blur_Haze_path_list,
                                  val_VI_Blur_Haze_path_list=val_VI_Blur_Haze_path_list,
                                  train_VI_Blur_Low_path_list=train_VI_Blur_Low_path_list,
                                  val_VI_Blur_Low_path_list=val_VI_Blur_Low_path_list,
                                  train_VI_Blur_OverExposure_path_list=train_VI_Blur_OverExposure_path_list,
                                  val_VI_Blur_OverExposure_path_list=val_VI_Blur_OverExposure_path_list,
                                  train_VI_Blur_Rain_path_list=train_VI_Blur_Rain_path_list,
                                  val_VI_Blur_Rain_path_list=val_VI_Blur_Rain_path_list,
                                  train_VI_Blur_Snow_path_list=train_VI_Blur_Snow_path_list,
                                  val_VI_Blur_Snow_path_list=val_VI_Blur_Snow_path_list,
                                  train_VI_Haze_Low_path_list=train_VI_Haze_Low_path_list,
                                  val_VI_Haze_Low_path_list=val_VI_Haze_Low_path_list,
                                  train_VI_Haze_rain_path_list=train_VI_Haze_rain_path_list,
                                  val_VI_Haze_rain_path_list=val_VI_Haze_rain_path_list,
                                  train_VI_Haze_Snow_path_list=train_VI_Haze_Snow_path_list,
                                  val_VI_Haze_Snow_path_list=val_VI_Haze_Snow_path_list,
                                  train_VI_Noise_Haze_path_list=train_VI_Noise_Haze_path_list,
                                  val_VI_Noise_Haze_path_list=val_VI_Noise_Haze_path_list,
                                  train_VI_LowLight_Rain_path_list=train_VI_LowLight_Rain_path_list,
                                  val_VI_LowLight_Rain_path_list=val_VI_LowLight_Rain_path_list,
                                  train_VI_LowLight_Snow_path_list=train_VI_LowLight_Snow_path_list,
                                  val_VI_LowLight_Snow_path_list=val_VI_LowLight_Snow_path_list,
                                  train_VI_Noise_Low_path_list=train_VI_Noise_Low_path_list,
                                  val_VI_Noise_Low_path_list=val_VI_Noise_Low_path_list,
                                  train_VI_Noise_Rain_path_list=train_VI_Noise_Rain_path_list,
                                  val_VI_Noise_Rain_path_list=val_VI_Noise_Rain_path_list,
                                  train_VI_Noise_Snow_path_list=train_VI_Noise_Snow_path_list,
                                  val_VI_Noise_Snow_path_list=val_VI_Noise_Snow_path_list,
                                  train_VI_OverExposure_Noise_path_list=train_VI_OverExposure_Noise_path_list,
                                  val_VI_OverExposure_Noise_path_list=val_VI_OverExposure_Noise_path_list,
                                  train_VI_Haze_Low_Rain_path_list=train_VI_Haze_Low_Rain_path_list,
                                  val_VI_Haze_Low_Rain_path_list=val_VI_Haze_Low_Rain_path_list,
                                  train_VI_Haze_Low_Snow_path_list=train_VI_Haze_Low_Snow_path_list,
                                  val_VI_Haze_Low_Snow_path_list=val_VI_Haze_Low_Snow_path_list,
                                  train_IR_path_list=train_IR_path_list,
                                  val_IR_path_list=val_IR_path_list,
                                  train_VI_path_list=train_VI_path_list,
                                  val_VI_path_list=val_VI_path_list,
                                  extra_train_path_lists=extra_train_path_lists,
                                  extra_val_path_lists=extra_val_path_lists,
                                  phase="val",
                              transform=data_transform["val"])
    
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 16])
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)
    model_clip, _ = clip.load("ViT-B/32", device=device)
    for param in model_clip.parameters():
            param.requires_grad = False
    model_degrad = DA_adapter(model_clip).to(device)
    model_degrad.set_frozen()
    checkpoint_de = torch.load('/data1/xml/ControlFusion0127/experiments/ControlFusion_train_stage1_20260517-120951/weights/checkpointdegradprior_lastest.pth', map_location=device)
    model_degrad.load_state_dict(checkpoint_de['model'])

    model = create_model().to(device)

    # for param in model.model_clip.parameters():
        # param.requires_grad = False

    if args.use_dp == True:
        model = torch.nn.DataParallel(model).cuda()

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        print(model.load_state_dict(weights_dict, strict=False))


    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1

    for epoch in range(start_epoch, args.epochs):
        # train
        train_loss, train_ssim_loss, train_max_loss, train_color_loss, train_grad_loss, lr = train_one_epoch(model=model,
                                                model_degrad=model_degrad,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                lr_scheduler=lr_scheduler,
                                                device=device,
                                                epoch=epoch)

        tb_writer.add_scalar("train_total_loss", train_loss, epoch)
        tb_writer.add_scalar("train_ssim_loss", train_ssim_loss, epoch)
        tb_writer.add_scalar("train_max_loss", train_max_loss, epoch)
        tb_writer.add_scalar("train_color_loss", train_color_loss, epoch)
        tb_writer.add_scalar("train_grad_loss", train_grad_loss, epoch)

        if epoch % args.val_every_epcho == 0 and epoch != 0:
            val_loss, val_ssim_loss, val_max_loss, val_color_loss, val_grad_loss = evaluate(model=model,
                                        model_degrad=model_degrad,
                                        data_loader=val_loader,
                                        device=device,
                                        epoch=epoch, lr=lr, filefold_path=file_img_path)

            tb_writer.add_scalar("val_total_loss", val_loss, epoch)
            tb_writer.add_scalar("val_ssim_loss", val_ssim_loss, epoch)
            tb_writer.add_scalar("val_max_loss", val_max_loss, epoch)
            tb_writer.add_scalar("val_color_loss", val_color_loss, epoch)
            tb_writer.add_scalar("val_grad_loss", val_grad_loss, epoch)


        # if val_loss < best_val_loss:
        #     if args.use_dp == True:
        #         save_file = {"model": model.module.state_dict(),
        #                      "optimizer": optimizer.state_dict(),
        #                      "lr_scheduler": lr_scheduler.state_dict(),
        #                      "epoch": epoch,
        #                      "args": args}
        #     else:
        #         save_file = {"model": model.state_dict(),
        #                      "optimizer": optimizer.state_dict(),
        #                      "lr_scheduler": lr_scheduler.state_dict(),
        #                      "epoch": epoch,
        #                      "args": args}
        #     torch.save(save_file, file_weights_path + "/" + "checkpoint.pth")
        #     best_val_loss = val_loss
    
        if args.use_dp == True:
                save_file = {"model": model.module.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "lr_scheduler": lr_scheduler.state_dict(),
                                "epoch": epoch,
                                "args": args}
        else:
                save_file = {"model": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "lr_scheduler": lr_scheduler.state_dict(),
                                "epoch": epoch,
                                "args": args}
        torch.save(save_file, file_weights_path + "/" + "checkpoint_lastest.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--ir_low_contrast_average_path', type=str, default="/data1/xml/datasets/DDL-27/IR_Low_contrast")
    parser.add_argument('--ir_noise_average_path', type=str, default="/data1/xml/datasets/DDL-27/IR_Noise")
    parser.add_argument('--ir_stripe_noise_average_path', type=str, default="/data1/xml/datasets/DDL-27/IR_Stripe_noise")

    parser.add_argument('--vi_noise_average_path', type=str, default="/data1/xml/datasets/DDL-27/VI_Noise")
    parser.add_argument('--over_exposure_average_path', type=str, default="/data1/xml/datasets/DDL-27/VI_Over_exposure")
    parser.add_argument('--vi_blur_average_path', type=str, default="/data1/xml/datasets/DDL-27/VI_Blur")
    parser.add_argument('--vi_haze_average_path', type=str, default="/data1/xml/datasets/DDL-27/VI_Haze")
    parser.add_argument('--vi_rain_average_path', type=str, default="/data1/xml/datasets/DDL-27/VI_Rain")
    parser.add_argument('--vi_low_light_average_path', type=str, default="/data1/xml/datasets/DDL-27/VI_Low_light")
    parser.add_argument('--VI_Snow_path', type=str, default="/data1/xml/datasets/DDL-27/VI_Snow")

    parser.add_argument('--VI_Blur_Haze_path', type=str, default="/data1/xml/datasets/DDL-27/VI_Blur_Haze")
    parser.add_argument('--VI_Blur_Low_path', type=str, default="/data1/xml/datasets/DDL-27/VI_Blur_Low")
    parser.add_argument('--VI_Blur_OverExposure_path', type=str, default="/data1/xml/datasets/DDL-27/VI_Blur_OverExposure")
    parser.add_argument('--VI_Blur_Rain_path', type=str, default="/data1/xml/datasets/DDL-27/VI_Blur_Rain")
    parser.add_argument('--VI_Blur_Snow_path', type=str, default="/data1/xml/datasets/DDL-27/VI_Blur_Snow")
    parser.add_argument('--VI_Haze_Low_path', type=str, default="/data1/xml/datasets/DDL-27/VI_Haze_Low")
    parser.add_argument('--VI_Haze_rain_path', type=str, default="/data1/xml/datasets/DDL-27/VI_Haze_rain")
    parser.add_argument('--VI_Haze_Snow_path', type=str, default="/data1/xml/datasets/DDL-27/VI_Haze_Snow")
    parser.add_argument('--VI_Noise_Haze_path', type=str, default="/data1/xml/datasets/DDL-27/VI_Noise_Haze")
    parser.add_argument('--VI_LowLight_Rain_path', type=str, default="/data1/xml/datasets/DDL-27/VI_LowLight_Rain")
    parser.add_argument('--VI_LowLight_Snow_path', type=str, default="/data1/xml/datasets/DDL-27/VI_LowLight_Snow")
    parser.add_argument('--VI_Noise_Low_path', type=str, default="/data1/xml/datasets/DDL-27/VI_Noise_Low")
    parser.add_argument('--VI_Noise_Rain_path', type=str, default="/data1/xml/datasets/DDL-27/VI_Noise_Rain")
    parser.add_argument('--VI_Noise_Snow_path', type=str, default="/data1/xml/datasets/DDL-27/VI_Noise_Snow")
    parser.add_argument('--VI_OverExposure_Noise_path', type=str, default="/data1/xml/datasets/DDL-27/VI_OverExposure_Noise")
    
    parser.add_argument('--VI_Haze_Low_Rain_path', type=str, default="/data1/xml/datasets/DDL-27/VI_Haze_Low_Rain")
    parser.add_argument('--VI_Haze_Low_Snow_path', type=str, default="/data1/xml/datasets/DDL-27/VI_Haze_Low_Snow")
    parser.add_argument('--IR_path', type=str, default="/data1/xml/datasets/DDL-27/IR")
    parser.add_argument('--VI_path', type=str, default="/data1/xml/datasets/DDL-27/VI")
    parser.add_argument('--extra_dataset_root', type=str, default="/data1/xml/datasets/DDL-27")
    
# /data1/xml/ControlFusion0111/experiments/ControlFusion_train_20260114-140556/weights/checkpoint_lastest.pth
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--epochs', type=int, default=60)
    # set the appropriate batch-size value for your device
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--val_every_epcho', type=int, default=10, help='val every epcho')
    parser.add_argument('--resume', default= '/data1/xml/ControlFusion0127/experiments/ControlFusion_train_20260517-191507/weights/checkpoint_lastest.pth', help='resume from checkpoint')
    parser.add_argument('--use_dp', default = False, help='use dp-multigpus')
    parser.add_argument('--device', default='cuda', help='device (i.e. cuda or cpu)')
    parser.add_argument('--gpu_id', default='0', help='device id (i.e. 0, 1, 2 or 3)')

    opt = parser.parse_args()

    main(opt)

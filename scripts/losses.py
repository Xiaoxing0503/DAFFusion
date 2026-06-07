import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import exp
import math
try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False
try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features

class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, self.logit_scale.exp())

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss
    
class AtomicDegradationLoss(nn.Module):
    """
    图像特征 vs 原子退化文本原型
    多标签 BCE 损失

    image_features:   [B, D]
    text_prototypes:  [K, D]
    multi_hot_targets:[B, K]
    """
    def __init__(self, init_temperature=0.07, pos_weight=None):
        super().__init__()
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / init_temperature), dtype=torch.float32)
        )
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, image_features, text_prototypes, multi_hot_targets):
        # L2 normalize -> cosine similarity
        image_features = F.normalize(image_features, dim=-1) #16, 512
        text_prototypes = F.normalize(text_prototypes, dim=-1)#9,

        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = scale * image_features @ text_prototypes.t()   # [B, K]

        if self.pos_weight is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits, multi_hot_targets.float(), pos_weight=self.pos_weight
            )
        else:
            loss = F.binary_cross_entropy_with_logits(
                logits, multi_hot_targets.float()
            )

        return loss, logits
        
class fusion_loss(nn.Module):
    def __init__(self):
        super(fusion_loss, self).__init__()
        self.loss_func_ssim = L_SSIM(window_size=48)
        self.loss_func_Grad = L_Grad_position()
        self.loss_func_Max = L_Intensity()
        self.loss_func_color = L_color()

    def forward(self, image_visible, image_infrared, image_fused, max_ratio=4, ssim_vis_ratio=1, ssim_ir_ratio=1, ssim_ratio=1, color_ratio=12, text_ratio=10):
        image_visible_gray = self.rgb2gray(image_visible)
        image_infrared_gray = self.rgb2gray(image_infrared)
        image_fused_gray = self.rgb2gray(image_fused)
        loss_max = max_ratio * self.loss_func_Max(image_visible, image_infrared, image_fused)
        loss_ssim = ssim_ratio * (ssim_vis_ratio * self.loss_func_ssim(image_visible, image_fused) + ssim_ir_ratio * self.loss_func_ssim(image_infrared_gray, image_fused_gray))
        loss_color = color_ratio * self.loss_func_color(image_visible, image_fused)

        loss_text = text_ratio * self.loss_func_Grad(image_visible_gray, image_infrared_gray, image_fused_gray)
        total_loss = loss_max + loss_ssim + loss_color + loss_text
        # total_loss = loss_text + loss_max + loss_ssim
        return total_loss, loss_ssim, loss_max, loss_color, loss_text

    def rgb2gray(self, image):
        b, c, h, w = image.size()
        if c == 1:
            return image
        image_gray = 0.299 * image[:, 0, :, :] + 0.587 * image[:, 1, :, :] + 0.114 * image[:, 2, :, :]
        image_gray = image_gray.unsqueeze(dim=1)
        return image_gray
    
class fusion_loss2(nn.Module):
    def __init__(self):
        super(fusion_loss2, self).__init__()
        self.loss_fn  = nn.L1Loss()  
        self.loss_func_Grad = L_Grad_position()
    
    def forward(self, image_visible, image_infrared, image_fused, max_ratio=4, ssim_vis_ratio=1, ssim_ir_ratio=1, ssim_ratio=1, color_ratio=12, text_ratio=10):

        image_visible_gray = self.rgb2gray(image_visible)
        image_infrared_gray = self.rgb2gray(image_infrared)
        image_fused_gray = self.rgb2gray(image_fused)
        
        L1_V = self.loss_fn(image_fused,image_visible)
        L1_in = self.loss_fn(image_fused_gray,torch.max(image_visible_gray, image_infrared_gray))
        L1_I = self.loss_fn(image_fused, image_infrared)
       
        loss_grad = self.loss_func_Grad(image_visible_gray, image_infrared_gray, image_fused_gray)
        loss = L1_V.cuda('cuda:0') + L1_I.cuda('cuda:0') + L1_in.cuda('cuda:0') + 5 * loss_grad.cuda('cuda:0')
        return loss, L1_V, L1_in, L1_I, loss_grad
    
    def rgb2gray(self, image):
        b, c, h, w = image.size()
        if c == 1:
            return image
        image_gray = 0.299 * image[:, 0, :, :] + 0.587 * image[:, 1, :, :] + 0.114 * image[:, 2, :, :]
        image_gray = image_gray.unsqueeze(dim=1)
        return image_gray
    
class fusion_prompt_loss(nn.Module):
    def __init__(self):
        super(fusion_prompt_loss, self).__init__()
        self.fusion_loss = fusion_loss()

    def forward(self, image_A, image_B, image_fused, task):
        total_loss = 0
        total_ssim_loss = 0
        total_max_loss = 0
        total_color_loss = 0
        total_grad_loss = 0

        num_tasks = len(task)

        for idx, task_type in enumerate(task):
            img_A = self._get_image(image_A, idx)
            img_B = self._get_image(image_B, idx)
            img_fused = self._get_image(image_fused, idx)

            loss, ssim_loss, max_loss, color_loss, grad_loss = self.fusion_loss(img_A, img_B, img_fused, max_ratio=8, ssim_ratio=1, text_ratio=10)


            total_loss += loss
            total_ssim_loss += ssim_loss
            total_max_loss += max_loss
            total_color_loss += color_loss
            total_grad_loss += grad_loss


        # Calculate the average for each loss component
        return total_loss / num_tasks, total_ssim_loss / num_tasks, total_max_loss / num_tasks, total_color_loss / num_tasks, total_grad_loss / num_tasks,

    @staticmethod
    def _get_image(images, index):
        return images[index].unsqueeze(0)

class L_color(nn.Module):
    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, image_visible, image_fused):
        ycbcr_visible = self.rgb_to_ycbcr(image_visible)
        ycbcr_fused = self.rgb_to_ycbcr(image_fused)

        cb_visible = ycbcr_visible[:, 1, :, :]
        cr_visible = ycbcr_visible[:, 2, :, :]
        cb_fused = ycbcr_fused[:, 1, :, :]
        cr_fused = ycbcr_fused[:, 2, :, :]

        loss_cb = F.l1_loss(cb_visible, cb_fused)
        loss_cr = F.l1_loss(cr_visible, cr_fused)

        loss_color = loss_cb + loss_cr
        return loss_color

    def rgb_to_ycbcr(self, image):
        r = image[:, 0, :, :]
        g = image[:, 1, :, :]
        b = image[:, 2, :, :]

        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b

        ycbcr_image = torch.stack((y, cb, cr), dim=1)
        return ycbcr_image

class L_Intensity(nn.Module):
    def __init__(self):
        super(L_Intensity, self).__init__()

    def forward(self, image_visible, image_infrared, image_fused):
        gray_visible = torch.mean(image_visible, dim=1, keepdim=True)
        gray_infrared = torch.mean(image_infrared, dim=1, keepdim=True)

        mask = (gray_infrared > gray_visible).float()

        fused_image = mask * image_infrared + (1 - mask) * image_visible
        # Loss_intensity = F.l1_loss(fused_image, image_fused)
        Loss_intensity = F.mse_loss(gray_visible, image_fused)+F.mse_loss(gray_infrared, image_fused)
        return Loss_intensity

# Use it only if you have a consistent modal preference
class L_Intensity_Consist(nn.Module):
    def __init__(self):
        super(L_Intensity_Consist, self).__init__()

    def forward(self, image_visible, image_infrared, image_fused, vis_compose, ir_compose, consist_mode="l1"):
        if consist_mode == "l2":
            Loss_intensity = (vis_compose * F.mse_loss(image_visible, image_fused) + ir_compose * F.mse_loss(image_infrared, image_fused))/2
        else:
            Loss_intensity = (vis_compose * F.l1_loss(image_visible, image_fused) + ir_compose * F.l1_loss(image_infrared, image_fused))/2
        return Loss_intensity

# use the L_Grad_position or L_Grad
class L_Grad_position(nn.Module):
    def __init__(self):
        super(L_Grad_position, self).__init__()
        self.sobel_x = nn.Parameter(torch.FloatTensor([[-1, 0, 1],
                                                       [-2, 0, 2],
                                                       [-1, 0, 1]]).view(1, 1, 3, 3), requires_grad=False).cuda()
        self.sobel_y = nn.Parameter(torch.FloatTensor([[-1, -2, -1],
                                                       [0, 0, 0],
                                                       [1, 2, 1]]).view(1, 1, 3, 3), requires_grad=False).cuda()
        self.padding = (1, 1, 1, 1)

    def forward(self, image_A, image_B, image_fuse):
        gradient_A_x, gradient_A_y = self.gradient(image_A)
        gradient_B_x, gradient_B_y = self.gradient(image_B)
        gradient_fuse_x, gradient_fuse_y = self.gradient(image_fuse)
        loss = F.l1_loss(gradient_fuse_x, torch.max(gradient_A_x, gradient_B_x)) + F.l1_loss(gradient_fuse_y, torch.max(gradient_A_y, gradient_B_y))
        return loss

    def gradient(self, image):
        image = F.pad(image, self.padding, mode='replicate')
        gradient_x = F.conv2d(image, self.sobel_x, padding=0)
        gradient_y = F.conv2d(image, self.sobel_y, padding=0)
        return torch.abs(gradient_x), torch.abs(gradient_y)

class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobel_x = nn.Parameter(torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3), requires_grad=False).cuda()
        self.sobel_y = nn.Parameter(torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3), requires_grad=False).cuda()
        self.padding = (1, 1, 1, 1)

    def forward(self, image_visible, image_infrared, image_fused):
        gray_visible = self.tensor_RGB2GRAY(image_visible)
        gray_infrared = self.tensor_RGB2GRAY(image_infrared)
        gray_fused = self.tensor_RGB2GRAY(image_fused)

        d1 = self.gradient(gray_visible)
        d2 = self.gradient(gray_infrared)
        df = self.gradient(gray_fused)
        edge_loss = F.l1_loss(torch.max(d1, d2), df)
        return edge_loss

    def gradient(self, image):
        image = F.pad(image, self.padding, mode='replicate')
        gradient_x = F.conv2d(image, self.sobel_x, padding=0)
        gradient_y = F.conv2d(image, self.sobel_y, padding=0)
        return torch.abs(gradient_x) + torch.abs(gradient_y)
    
    def tensor_RGB2GRAY(self, image):
        b,c,h,w = image.size()
        if c == 1:
            return image
        image_gray = 0.299 * image[:, 0, :, :] + 0.587 * image[:, 1, :, :] + 0.114 * image[:, 2, :, :]
        image_gray = image_gray.unsqueeze(dim=1)
        return image_gray


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=24, window=None, size_average=True, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    return 1 - ret


class L_SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(L_SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        (_, channel_2, _, _) = img2.size()

        if channel != channel_2 and channel == 1:
            img1 = torch.concat([img1, img1, img1], dim=1)
            channel = 3

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window.cuda()
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window.cuda()
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


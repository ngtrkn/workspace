from __future__ import print_function, division

import logging
import os
import yaml
import gc
# import warnings
# import numpy as np
import sys
from tqdm import tqdm

from pathlib import Path
from collections import OrderedDict
sys.path.append('.')

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
from torchvision.utils import save_image

# from models.backbone import CENet
# from models import AUnet as AINet
# from models import Unet as AINet
from models import PAUnet as AINet
# from auxiliary.preprocessor import LayoutPreprocessor
from train_utils.optimizer import Ranger
from data.retourch_dataset import RetourchDataset, variable_mask_collate_fn



logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_CONFIG_PATH = Path(__file__).parent / "../config/retourch_config.yaml"


import functools

def control_loss(loss_func):
    """Print the function signature and return value"""
    @functools.wraps(loss_func)
    def wrapper_loss(*args, **kwargs):
        #TODO: control empty input
        num_target_dim = len(args[1].shape)
        target_check = torch.amax(args[1], dim=[i for i in range(num_target_dim) if i > 0])
        
        #TODO: remove item
        for v in args:
            filt_v = [
                item[ib:ib+1] for ib, item in enumerate(v)
                if target_check[ib] >= 0
            ]
            if len(filt_v) == 0:
                return 0
            v = torch.cat(filt_v, dim=0)
        # for ib, item in enumerate(target_check):
        #     if item < 0:
        #         print(f"{loss_func.__name__!r} ignore index {ib!r} due to {item!r} < 0")           # 1

        value = loss_func(*args, **kwargs)

        # print(f"{loss_func.__name__!r} return {value!r}")           # 2

        
        return value
    return wrapper_loss


BCE = nn.BCELoss(reduction="none")
EPS = 1e-10
# @control_loss
def bce_loss(input_, target, ignore_index=-100, reduction='mean'):
    out = BCE(input_, target)
    out = out[target != ignore_index]
    if reduction == "mean":
        return torch.mean(out)
    elif reduction == "sum":
        return torch.sum(out)
    else:
        raise ValueError(f"reduction type does not support {reduction}")

# @control_loss
def dice_loss(input_, target, ignore_index=-100):
    smooth = 1.0
    target[target == ignore_index] = 0
    iflat = input_.reshape(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

# from encoding.nn import SyncBatchNorm
# import sys
# sys.path.append("/mnt/sda1/code/github/PyTorch-Encoding/encoding/models/sseg")
# from encoding.models.sseg.deeplab import DeepLabV3

class BaseModel:
    def __init__(
        self,
        device="-1",
        AINet=AINet,
        input_channels=1,
        num_classes=3,
        weights_path=None,
        config=DEFAULT_CONFIG_PATH,
        mode="inference",
    ):
        """
        Base Model:
        """
        # if it's a config dict, use directly
        if isinstance(config, dict):
            self.cfg = config
        else:
            with open(config, "r") as f:
                self.cfg = yaml.load(f, Loader=yaml.FullLoader)

        #TODO: set up parameters
        self.mode = mode

        #TODO: setup device
        if (device != "-1") and (torch.cuda.is_available()):
            device_amount = len([int(x) for x in device.split(",")])
            cuda_amount = torch.cuda.device_count()
            if cuda_amount > device_amount:
                default_device = min([int(x) for x in device.split(",")])
                self.device = 'cuda:{}'.format(default_device)
            else:
                self.device = 'cuda:0'
        else:
            self.device = 'cpu'

        # model
        self.model = AINet(num_classes)
        # self.model = AINet(input_channels=input_channels, nclasses=num_classes)
        # self.model = DeepLabV3(num_classes, backbone='resnet50s', root='~/.encoding/models',
        #                 aux = False,
        #                 se_loss = False, 
        #                 norm_layer = SyncBatchNorm,
        #                 base_size=224, crop_size=192)

        # pretrain
        self.load(weights_path, mode=mode)
        # enable parallel
        if ('cuda' in self.device) and (mode == 'training') and (device_amount
                                                                 > 1):
            if cuda_amount > device_amount:
                self.model = nn.DataParallel(
                    self.model, device_ids=[int(x) for x in device.split(",")
                                            ]).to(device=self.device)
            else:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=[int(x) for x in range(len(device.split(",")))
                                ]).to(device=self.device)
            logger.info("DataParallel activated...")
        elif ('cuda' in self.device):
            self.model.to(device=self.device)
            logger.info("cuda activated...")
        else:
            logger.info("cpu mode....")

        if mode == "training":
            self.current_epoch = 0
            self.optimizer = None
            self.lr_scheduler = None
            assert (
                self.optimizer == None
            ), "optimizer only instantiates after applying first time running .fit() or load() from checkpoints"
            self.optimizer_state_dict = None
            self.is_optimizer_init = False
    

    def load(self, path, mode="inference", learning_rate=1e-3):

        if path != None and os.path.exists(path):
            checkpoint = torch.load(path,
                                    map_location=lambda storage, loc: storage)
            try:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            # If KeyError, consider the checkpoint itself is the weight (only) object
            except KeyError:
                try:
                    self.model.load_weights(path)
                except:
                    raise IOError(
                        f"your checkpoint at {path} do not contains \"model_state_dict\" and it is not the weights itself"
                    )

            if self.mode == "training":

                if 'model_config' in checkpoint:
                    self.cfg = checkpoint['config']

                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer = Ranger(self.model.parameters(),
                                            lr=learning_rate)

                    # Save optimizer state dict to init optimizer in .fit()
                    optimizer_state_dict = checkpoint["optimizer_state_dict"]
                    self.optimizer.load_state_dict(optimizer_state_dict)

                self.current_epoch = checkpoint.get("current_epoch", 0)

        else:
            logger.info("Weights initialized for training from scratch")

        # logger.info("Loading Successfully.......")

        return self

    @staticmethod
    def _prepare_data(x, y):
        sample_list = []
        for img, gts in zip(x, y):
            sample_point = [img] + gts
            sample_list.append(sample_point)

        return sample_list

    
    def _setup_optimizer_scheduler(
            self,
            learning_rate: float=1e-3,
            decayRate: int = 0.98):
        assert NotImplemented

    def _setup_dataloader(
            self,
            x, y,
            batch_size: int = 8,
            use_augmentation: bool = False,
            default_height: int = 128,
            num_workers: int = 0,
            ):
        assert NotImplemented


    def _setup_criteria(self):
        assert NotImplemented


    def fit(
        self,
        x: list,
        y: list,
        learning_rate: float = 1e-3,
        batch_size: int = 2,
        decayRate: int = 0.98,
        num_workers: int = 0,
        default_height: int = 128,
        disable_progress_bar: bool = False,
        use_augmentation=False,
        debug_path="debug",
        writer=None,
        **kwargs) -> dict:

        #TODO: activate train mode
        self.model.train()
        os.makedirs(debug_path, exist_ok=True)

        #TODO: setup optimiser: Initialize optimizer for first epoch
        if self.optimizer == None or self.lr_scheduler == None:
            self._setup_optimizer_scheduler()
        else:
            self.lr_scheduler.step()
        

        if batch_size == None:
            batch_size = self.cfg["train"]["batch_size"]

        #TODO: set up dataloader
        dataset_loader = self._setup_dataloader(
            x, y,
            batch_size=batch_size,
            use_augmentation=use_augmentation,
            default_height=default_height,
            num_workers=num_workers
        )

        #TODO: set up criterion
        self._setup_criteria()

        #TODO: set up progress bar
        n_batches = len(dataset_loader)
        progressbar = tqdm(range(n_batches),
                           total=n_batches,
                           desc=f"[Epoch {self.current_epoch}][Training]::",
                           disable=disable_progress_bar,
                           ascii=True)

        #TODO: training
        train_data = iter(dataset_loader)
        for iteration in progressbar:
            image, masks_list = next(train_data)
            num_masks = len(masks_list)

            #TODO: loading data into device
            input_ = image.to(self.device).detach().float()
            targets = [mask.to(self.device).detach().float() for mask in masks_list]
            target_ = torch.cat([mask.unsqueeze(1) for mask in masks_list], dim=1)

            #TODO: inferring
            output_ = self.model(input_)
            outputs = [output_[:, i, :, :] for i in range(num_masks)]

            #TODO: calculating losses
            postfix_progress = {}
            loss = 0
            with torch.autograd.set_detect_anomaly(True):
                for _name, _func in self.criterion.items():
                    loss_values = [
                        _func(output, target)
                        for output, target in zip(outputs, targets)
                    ]
                    loss_value = sum(loss_values) / num_masks

                    postfix_progress[_name] = loss_value.detach().item()
                
                    #TODO: updating model weights
                    loss_value.backward(retain_graph=True)
                    if self.cfg["train"]["clip_gradient"]:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3)

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    

            #TODO: updating progress bar
            progressbar.set_postfix(ordered_dict=postfix_progress)
            progressbar.update()

            #TODO: update in tensorboard logs
            if writer is not None:
                for name, value in postfix_progress.items():
                    writer.add_scalar(f"train_iteration/{name}", value, self.current_epoch*n_batches + iteration)

            #TODO: store debug
            if (iteration + 1) % 500 == 0:
                save_image(torch.cat(
                        [(1 + image) / 2.] + \
                        [output_.detach().cpu()] + \
                        [target_], dim=0),
                    f"{debug_path}/{self.current_epoch}_{iteration}.jpg",
                    nrow=output_.shape[0], 
                    normalize=False)
            
        self.current_epoch += 1

        return postfix_progress

    def save(self, path):

        if self.mode == "inference":
            raise RuntimeError(
                "Model in inference mode, consider using export() method")

        _, ext = os.path.splitext(path)

        # Check if exist parent folder else make dir
        father_dir = os.path.dirname(path)

        if father_dir != "" and not os.path.exists(father_dir):
            os.makedirs(father_dir, exist_ok=True)

        if ext != ".pth":
            warnings.warn(
                "As convention, you should save checkpoints with .pth extension",
                UserWarning,
            )

        if self.optimizer:
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "config": self.cfg,
                    "current_epoch": self.current_epoch,
                },
                path,
            )
        else:
            raise RuntimeError(
                "optimizer isn't initialized, please .fit() at least one epochs before save"
            )

        logger.info(f'Model saved at "{path}"')

        return path


    @torch.no_grad()
    def debug(self, x, y=[], batch_size=2, use_augmentation=False, default_height=1024, num_workers=0, debug_path="debug"):
        os.makedirs(debug_path, exist_ok=True)
        if not y:
            y = [[] for i in range(len(x))]
        #TODO: set up dataloader
        dataset_loader = self._setup_dataloader(
            x, y,
            batch_size=batch_size,
            use_augmentation=use_augmentation,
            default_height=default_height,
            num_workers=num_workers
        )

        for i, (image, labels) in enumerate(dataset_loader):
            #TODO: loading data into device
            input_ = image.to(self.device).detach().float()

            #TODO: inferring
            output_ = self.model(input_)
            outputs = [output_[:, i, :, :] for i in range(output_.shape[1])]
            

            if not labels:
                targets = []
            else:
                targets = [mask.to(self.device).detach().float() for mask in masks_list]

            #TODO: store debug
            debug_img = torch.cat([(1 + image) / 2.] + \
                            [output.unsqueeze(1).detach().cpu() for output in outputs] + \
                            [target.unsqueeze(1).detach().cpu() for target in targets], dim=0)
            
            save_image(
                    debug_img,
                    f"{debug_path}/{i}.jpg",
                    nrow=output_.shape[0],
                    padding=5,
                    pad_value=128,
                    normalize=False)
        

    @torch.no_grad()
    def process(self, image_path):
        raise NotImplemented

        
        


if __name__=='__main__':
    model = RetouchModel(
        device='0',
        weights_path=None, #r'D:\Workspace\cinnamon\code\prj\kawasakikisen\train\layout\lib-layout\scripts\checkpoints\JeffLayout-v0\JeffLayout-v0_epoch00010.pth',
        mode="training"
    )

    #TODO: test inference
    # output = model.process(r'D:\Workspace\cinnamon\data\KLine\version_0\train\images\20201224_追加学習データ_NOLA_Scarlet Eagle_5.png')
    # exit(0)
    #TODO: test fitting
    x = [
        r'D:\Workspace\data\FFHQ\thumbnails128x128\00000\00725.png',
        r'D:\Workspace\data\FFHQ\thumbnails128x128\00000\00726.png',
        r'D:\Workspace\data\FFHQ\thumbnails128x128\00000\00727.png',
    ]

    y = [
        [
            r'D:\Workspace\data\FFHQ\thumbs\00000\00725.png',
        ],
        [
            r'D:\Workspace\data\FFHQ\thumbs\00000\00726.png',
        ],
        [
            r'D:\Workspace\data\FFHQ\thumbs\00000\00727.png',
        ],
    ]

    print(model.fit(x, y))

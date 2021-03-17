from __future__ import print_function, division

import logging
import os
import yaml
import gc
import sys
from tqdm import tqdm

from pathlib import Path
from collections import OrderedDict
sys.path.append('.')

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
from torchvision.utils import save_image

# from models import AUnet as AINet
# from models import Unet as AINet
# from models import PAUnet as AINet
sys.path.append('/mnt/sda1/code/github/lib-layout/layout/jeff')
from backbone import CENet as AINet
from train_utils.optimizer import Ranger
from data.layout_dataset import LayoutDataset, variable_mask_collate_fn


from .base import BaseModel
from train_utils.losses import bce_loss, dice_loss, contour_loss


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_CONFIG_PATH = Path(__file__).parent / "../config/segment_config.yaml"


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
    if len(out) == 0:
        return 0
    if reduction == "mean":
        return torch.mean(out)
    elif reduction == "sum":
        return torch.sum(out)
    else:
        raise ValueError(f"reduction type does not support {reduction}")

# @control_loss
def dice_loss(input_, target, ignore_index=-100):
    smooth = 1.0
    # target[target == ignore_index] = 0

    # iflat = input_.reshape(-1)
    # tflat = target.view(-1)
    iflat = input_[target != ignore_index]
    tflat = target[target != ignore_index]
    if len(iflat) == 0: return 0

    intersection = (iflat * tflat).sum()

    return 1 - ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


class LayoutModel(BaseModel):
    def __init__(self,
        AINet=AINet,
        *args,
        **kwargs):
        super(LayoutModel, self).__init__(*args, AINet=AINet, **kwargs)


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
        self.optimizer = Ranger(self.model.parameters(), learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, gamma=decayRate)
        # Update learning rate
        for g in self.optimizer.param_groups:
            g["lr"] = learning_rate
        logger.info(f"Loading OPTIMISER {self.optimizer}")
        logger.info(f"Loading SCHEDULER {self.lr_scheduler}")

    def _setup_dataloader(
            self,
            x, y,
            batch_size: int = 8,
            use_augmentation: bool = False,
            default_height: int = 128,
            num_workers: int = 0,
            ):
        sample_list = self._prepare_data(x, y)
        dataset = LayoutDataset(sample_list, default_height=default_height, in_channels=1)
        logger.info(f"Loading dataset {dataset}")
        dataset_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=variable_mask_collate_fn,
        )
        return dataset_loader


    def _setup_criteria(self):
        self.criterion= {
            'BCE': bce_loss,
            'Dice': dice_loss,
            'Contour': contour_loss,
        }
        logger.info(f"Loading criteria {self.criterion}")


    def fit(
        self,
        x: list,
        y: list,
        learning_rate: float = 1e-3,
        batch_size: int = 2,
        decayRate: int = 0.98,
        num_workers: int = 0,
        default_height: int = 1024,
        disable_progress_bar: bool = False,
        use_augmentation=False,
        debug_path="layout_debug",
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
                    loss_values = [v for v in loss_values if v > 0]
                    loss_value = sum(loss_values) / len(loss_values)

                    postfix_progress[_name] = loss_value.detach().item()


                    if 'Dice' in _name: loss_value = 1.5 * loss_value


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
            if (iteration + 1) % 200 == 0:
                save_image(torch.cat(
                        [(1 + image) / 2.] + \
                        [output.unsqueeze(1).detach().cpu() for output in outputs] + \
                        [target.unsqueeze(1).detach().cpu() for target in targets], dim=0),
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


        
        


if __name__=='__main__':
    model = LayoutModel(
        device='0',
        weights_path=None, #r'D:\Workspace\cinnamon\code\prj\kawasakikisen\train\layout\lib-layout\scripts\checkpoints\JeffLayout-v0\JeffLayout-v0_epoch00010.pth',
        mode="training",
        num_classes=4,
    )

    #TODO: test inference
    # output = model.process(r'D:\Workspace\cinnamon\data\KLine\version_0\train\images\20201224_追加学習データ_NOLA_Scarlet Eagle_5.png')
    # exit(0)
    #TODO: test fitting
    x = [
        '/mnt/sda1/data/cinnamon/layout/Invoice_Train/AruNET_Invoice_Training/マ_カイロスマーケティング_0.jpg',
        '/mnt/sda1/data/cinnamon/layout/Invoice_Train/AruNET_Invoice_Training/マ_ネットプロテクションズ_0.jpg',
        '/mnt/sda1/data/cinnamon/layout/Invoice_Train/AruNET_Invoice_Training/アロマ_0.jpg',
    ]

    y = [
        [
            '/mnt/sda1/data/cinnamon/layout/Invoice_Train/AruNET_Invoice_Training/マ_カイロスマーケティング_0_GT0.jpg',
            '/mnt/sda1/data/cinnamon/layout/Invoice_Train/AruNET_Invoice_Training/マ_カイロスマーケティング_0_GT1.jpg',
            '/mnt/sda1/data/cinnamon/layout/Invoice_Train/AruNET_Invoice_Training/マ_カイロスマーケティング_0_GT2.jpg',
            '',
        ],
        [
            '/mnt/sda1/data/cinnamon/layout/Invoice_Train/AruNET_Invoice_Training/マ_ネットプロテクションズ_0_GT0.jpg',
            '/mnt/sda1/data/cinnamon/layout/Invoice_Train/AruNET_Invoice_Training/マ_ネットプロテクションズ_0_GT1.jpg',
            '/mnt/sda1/data/cinnamon/layout/Invoice_Train/AruNET_Invoice_Training/マ_ネットプロテクションズ_0_GT2.jpg',
            '',
        ],
        [
            '',
            '/mnt/sda1/data/cinnamon/layout/Invoice_Train/AruNET_Invoice_Training/アロマ_0_GT1.jpg',
            '/mnt/sda1/data/cinnamon/layout/Invoice_Train/AruNET_Invoice_Training/アロマ_0_GT2.jpg',
            '/mnt/sda1/data/cinnamon/layout/Invoice_Train/AruNET_Invoice_Training/アロマ_0_GT0.jpg'
        ],
    ]

    print(model.fit(x, y))

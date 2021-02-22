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

from models import AUnet as AINet
# from models import Unet as AINet
# from models import PAUnet as AINet
from train_utils.optimizer import Ranger
from data.segment_dataset import SegmentDataset, variable_mask_collate_fn


from .base import BaseModel


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_CONFIG_PATH = Path(__file__).parent / "../config/segment_config.yaml"


BCE = nn.BCELoss(reduction="none")
def bce_loss(input_, target, ignore_index=-100, reduction='mean'):
    out = BCE(input_, target)
    out = out[target != ignore_index]
    if reduction == "mean":
        return torch.mean(out)
    elif reduction == "sum":
        return torch.sum(out)
    else:
        raise ValueError(f"reduction type does not support {reduction}")


class SegmentModel(BaseModel):
    def __init__(self,
        *args,
        **kwargs):
        super(SegmentModel, self).__init__(*args, **kwargs)


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
        dataset = SegmentDataset(sample_list, default_height=default_height, in_channels=3)
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
        default_height: int = 128,
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
            # targets = [mask.to(self.device).detach().long() for mask in masks_list] 
            target_ = torch.cat([mask.unsqueeze(1) for mask in masks_list], 1).to(self.device).detach().float()

            #TODO: inferring
            output_ = self.model(input_)
            # outputs = [output_[:, i, :, :] for i in range(num_masks)]

            #TODO: calculating losses
            postfix_progress = {}
            loss = 0
            with torch.autograd.set_detect_anomaly(True):
                for _name, _func in self.criterion.items():
                    loss_value = _func(output_, target_)

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
            if (iteration + 1) % 100 == 0:
                outputs = torch.cat(
                    [
                        torch.argmax(output_, dim=1, keepdim=True).detach().cpu().float()/150,
                        torch.argmax(output_, dim=1, keepdim=True).detach().cpu().float()/150,
                        torch.argmax(output_, dim=1, keepdim=True).detach().cpu().float()/150
                    ], dim=1
                )
                targets = torch.cat(
                    [
                        torch.argmax(target_, dim=1, keepdim=True).detach().cpu().float()/150,
                        torch.argmax(target_, dim=1, keepdim=True).detach().cpu().float()/150,
                        torch.argmax(target_, dim=1, keepdim=True).detach().cpu().float()/150
                    ], dim=1
                )
                save_image(torch.cat(
                        [(1 + image) / 2.] + \
                        [outputs] + \
                        [targets], dim=0),
                    f"{debug_path}/{self.current_epoch}_{iteration}.jpg",
                    nrow=output_.shape[0], 
                    normalize=False)
            
        self.current_epoch += 1

        return postfix_progress

    def save(self, path):

        # if self.mode == "inference":
        #     raise RuntimeError(
        #         "Model in inference mode, consider using export() method")

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
    model = SegmentModel(
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

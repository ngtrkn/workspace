import torch
from torch.utils.data import Dataset

from PIL import Image
import numpy as np
import cv2


class BaseDataset(Dataset):
    def __init__(
        self,
        image_list: list,
        augmentation=None,
        default_height: int=1024,
    ):
        self.image_list = image_list
        self.augmentation = augmentation
        self.default_height = default_height
        self.width_limit = int(default_height*0.75) // 32 * 32

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        raise NotImplemented

    def read_image_path(self, image_path, image_channel=1):
        assert image_channel in [1, 3], "Accept only 1-channel (gray) / 3-channel (RGB) image"
        if image_channel == 1:
            return Image.open(image_path).convert("L")
        else:
            return Image.open(image_path).convert("RGB")

    def to_tensor(self, image_array_list):
        return [
            torch.FloatTensor(image_array) if image_array is not None else None
            for image_array in image_array_list
        ]

    def default_transform(self, image_array_list):
        origin_height, origin_width = image_array_list[0].shape[:2]
        width = int(origin_width * self.default_height / origin_height)

        #TODO: random crop to aspect width
        random_crop=np.random.randint(width - self.width_limit, size=None) if width > self.width_limit \
                else -1

        #TODO: resize to define height
        if random_crop >= 0:
            image_array_list = [
                cv2.resize(image, (width, self.default_height), interpolation=cv2.INTER_AREA)
                if image is not None
                else None
                for image in image_array_list
            ]
            image_array_list = [
                image[:, random_crop:random_crop + self.width_limit]
                if image is not None
                else None
                for image in image_array_list
            ]
        else:
            image_array_list = [
                cv2.resize(image, (self.width_limit, self.default_height), interpolation=cv2.INTER_AREA)
                if image is not None
                else None
                for image in image_array_list
            ]

        #TODO: export to 3 channel
        image_array_list = [
            None if image is None
            else np.expand_dims(image, axis=-1) if len(image.shape)==2 else image
            for image in image_array_list
        ]

        #TODO:
        # image_array_list = [
        #     np.pad(image, ((32, 32),(32,32),(0,0)), 'constant', constant_values=150)
        #     if image is not None else None 
        #     for i, image in enumerate(image_array_list)
        # ]
        
        #TODO: normalize
        image_array_list = [
            image / 127.5 - 1. 
            if i==0 
            else image / 255. if image is not None else None 
            for i, image in enumerate(image_array_list)
        ]

        #TODO: transpose to 3 channel
        image_array_list = [
            None if image is None
            else np.transpose(image, (2,0,1))
            for image in image_array_list
        ]
        
        return image_array_list
from .base import BaseDataset
import numpy as np
import cv2
import torch


def variable_mask_collate_fn(batch):
    batch_size = len(batch)
    max_n_masks = 151
    image_shape = list(batch[0][0].shape)

    images = torch.ones([batch_size] + image_shape) * (-100)
    masks_list = [
        torch.zeros([batch_size] + image_shape[1:])
        for _ in range(max_n_masks)
    ]
    
    
    

    for i, (image, label) in enumerate(batch):
        images[i] = batch[i][0]
        
        y_onehot = torch.FloatTensor(image_shape[1] * image_shape[2], max_n_masks)
        y_onehot.zero_()
        y_onehot.scatter_(1, (batch[i][1][0]*255).long().reshape(-1, 1), 1)
        y_onehot = y_onehot.reshape(image_shape[1], image_shape[2], -1)

        for j in range(max_n_masks):
            masks_list[j][i,:,:] = y_onehot[:,:,j]

    return images, masks_list


class SegmentDataset(BaseDataset):
    def __init__(
        self,
        *args,
        in_channels=1,
        **kwargs,
    ):
        super(SegmentDataset, self).__init__(*args, **kwargs)
        assert in_channels in [1, 3], "Allow only 1/3 channels input"
        self.in_channels = in_channels
        self.default_height = int(self.default_height) // 32 * 32
        self.width_limit = self.default_height

    def __getitem__(self, idx):
        images_path = self.image_list[idx]
        images_list = []

        #TODO: load image
        for _idx, _image_path in enumerate(images_path):
            #TODO: read image
            if _idx == 0:
                _image = self.read_image_path(_image_path, image_channel=self.in_channels)
            else:
                if _image_path in [None, '']:
                    _image = None
                else:
                    _image = self.read_image_path(_image_path, image_channel=1)

            #TODO: convert to numpy
            _image = np.array(_image) if _image is not None else None

            #TODO: 
            images_list.append(_image)

        #TODO: make transform
        if self.augmentation:
            images_list = self.augmentation(images_list)
            images_list = self.default_transform(images_list)
        else:
            images_list = self.default_transform(images_list)


        #TODO: to tensor
        images_list = self.to_tensor(image_array_list=images_list)

        return images_list[0], images_list[1:]


if __name__=='__main__':
    x = [
        r'D:\Workspace\data\FFHQ\thumbnails128x128\00000\00725.png',
        r'D:\Workspace\data\FFHQ\thumbnails128x128\00000\00726.png',
        r'D:\Workspace\data\FFHQ\thumbnails128x128\00000\00727.png',
    ]

    y = [
        [
            r'D:\Workspace\data\FFHQ\thumbnails128x128\00000\00725.png',
        ],
        [
            r'D:\Workspace\data\FFHQ\thumbnails128x128\00000\00726.png',
        ],
        [
            r'D:\Workspace\data\FFHQ\thumbnails128x128\00000\00727.png',
        ],
    ]
    image_list = []
    for x_, y_ in zip(x,y):
        element = [x_] + y_
        image_list.append(element)
    
    dataset = RetourchDataset(
        default_height=128,
        image_list=image_list,
        in_channels=3,
    )

    for data, label in dataset:
        print(data.shape, [item.shape if item is not None else None for item in label])

    from torch.utils.data import DataLoader

    dataset_loader = DataLoader(
            dataset=dataset,
            batch_size=2,
            shuffle=True,
            pin_memory=True,
            num_workers=0,
            collate_fn=variable_mask_collate_fn,
        )

    for data, label in dataset_loader:
        print(data.shape, [(item.shape, item.max(), item.min()) if item is not None else None for item in label])
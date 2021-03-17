import yaml
import os
from torch.utils.tensorboard import SummaryWriter

# from data_utils.create_data_retouch import create_train_data
# from data_utils.create_data_segment import create_train_data
from data_utils.create_data_layout import create_train_data

# from models import AUnet as AINet
import sys
sys.path.append('/mnt/sda1/code/github/lib-layout/layout/jeff')
from backbone import CENet as AINet

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# parameters
# train_config='config/train_retouch_config.yaml'
# train_config='config/train_segment_config.yaml'
train_config='config/train_cinlayout_config.yaml'
support_models = {
    'unet': 'BaseModel',
    'retouch': 'RetouchModel',
    'segment': 'SegmentModel',
    'layout': 'LayoutModel',
}

#TODO: load config
with open(train_config, "r", encoding="utf-8") as f:
    train_cfg = yaml.load(f, Loader=yaml.FullLoader)


            


    

#TODO: instancing train model
MODEL_NAME = train_cfg['model']
assert MODEL_NAME in support_models.keys(), f"{MODEL_NAME} is not in list supported {support_models.keys()}"
exec(f"from core import {support_models[MODEL_NAME]} ")
exec(f"MODEL_PT = {support_models[MODEL_NAME]}")

def train(model_pointer=MODEL_PT, train_cfg=train_cfg, model_cfg="config/retourch_config.yaml"):
    writer = SummaryWriter(f"logs/{train_cfg['session_code']}")
    checkpoint_dir = os.path.join(train_cfg["save_dir"], train_cfg["session_code"])

    #TODO: init model
    model = model_pointer(
        AINet=AINet,
        input_channels=1,
        weights_path=train_cfg.get("pretrained_path"),
        config=model_cfg,
        mode="training",
        device=train_cfg.get('gpu_ids', '-1'),
        num_classes=train_cfg.get('num_classes', 4),
    )

    #TODO: fetch data
    images_train, labels_train = create_train_data(train_cfg["train_dataset"])

    #TODO: Start training
    while model.current_epoch < train_cfg["n_epochs"]:
        loss = model.fit(
            x=images_train,
            y=labels_train,
            batch_size=train_cfg["batch_size"],
            num_workers=train_cfg["num_workers"],
            learning_rate=train_cfg["learning_rate"],
            use_augmentation=train_cfg["use_augmentation"],
            disable_progress_bar=train_cfg["disable_progress_bar"],
            writer=writer,
            debug_path=train_cfg['session_code'],
        )

        save_name = "{}_epoch{}".format(
            train_cfg["session_code"], str(model.current_epoch).zfill(5)
        )

        if model.current_epoch % train_cfg["n_epochs_validation"] == 0:
            # Retrieve saving extension
            model_save_extension = train_cfg["model_save_extension"]

            # Save model with save_name in checkpoint_dir with corresponding extension
            save_path = os.path.join(
                checkpoint_dir, f"{save_name}{model_save_extension}"
            )
            model.save(save_path)





if __name__=='__main__':
    train()
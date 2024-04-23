# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from torch.utils.data import Dataset

from logging import getLogger
from PIL import Image

import torch
import os
import json

ROOT_DATA_PATH = "/home/s/DATA"
PATHS_FILE = "/mnt/quanhd/endoscopy/pretrain.json"

_GLOBAL_SEED = 0
logger = getLogger()

def make_polyp(
    transform=None,
    batch_size=64,
    collator=None,
    pin_mem=True,
    num_workers=8,
    training=True,
    world_size=1,
    rank=0,
    root_path=PATHS_FILE,
    image_folder=ROOT_DATA_PATH,
    copy_data=False,
    drop_last=False,
):
    dataset = Polyp(root=root_path, image_folder=image_folder, transform=transform)
    logger.info("Polyp dataset created")

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        shuffle=True,
        # batch_sampler=trainSampler,
        num_workers=num_workers,
        persistent_workers=False,
    )
    logger.info("Polyp unsupervised data loader created")

    return dataset, data_loader


class Polyp(Dataset):
    def __init__(self, image_folder=ROOT_DATA_PATH, root=PATHS_FILE, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.root = root
        self.img_paths = self._get_img_paths()

    def __len__(self):
        return len(self.img_paths)

    def _get_img_paths(self):
        f = open(self.root)
        data = json.load(f)
        img_paths = [elem for elem in data["train"]]
        return img_paths

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        full_img_path = os.path.join(self.image_folder, img_path)
        image = Image.open(full_img_path)

        if self.transform:
            image = self.transform(image)

        return image


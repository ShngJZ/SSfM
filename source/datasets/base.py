import torch
from easydict import EasyDict as edict
from typing import Any, Dict

from source.utils.config_utils import override_options

default_conf = {'copy_data': False,
                # specific to cluster, if needs to copy from zip

                'resize': None,
                'resize_factor': None,
                'resize_by': 'max',
                'crop_ratio': None,
                'crop': None,
                'apply_augmentation': False,
                'train_sub': None,
                'val_sub': None,
                'mask_img': False,

                'increase_depth_range_by_x_percent': 0.,

                # llff
                'llffhold': 8,

                # dtu stuff
                'dtu_split_type': 'pixelnerf',
                'dtuhold': 8,
                'dtu_light_cond': 3,   # Light condition. Used only by DTU.
                'dtu_max_images': 49,   # Whether to restrict the max number of images.'
                }

class Dataset(torch.utils.data.Dataset):
    """Base for all datasets. """
    def __init__(self, args: Dict[str, Any], split: str):
        super().__init__()
        self.args = edict(override_options(default_conf, args))
        self.split = split

    def prefetch_all_data(self):
        # pre-iterate through all samples and group together
        self.all = torch.utils.data._utils.collate.default_collate([s for s in self])

    def setup_loader(self, shuffle: bool = False, drop_last: bool = False):
        loader = torch.utils.data.DataLoader(self,
            batch_size=1,
            num_workers=self.args.num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=False, # spews warnings in PyTorch 1.9 but should be True in general
        )
        print("number of samples: {}".format(len(self)))
        return loader
from source.datasets import dataset_dict
from torch.utils.data import Dataset
from torch.utils.data import DistributedSampler
from typing import Any, Dict


def create_dataset(args: Dict[str, Any], mode: str='train') -> Dataset:
    """
    Args:
        args (dict): settings. Must contain keys 'dataset' and 'scene'
        mode (str, optional): Defaults to 'train'.

    Returns:
        Dataset and optional sampler.
    """
    if mode == 'train':
        print('training dataset: {}'.format(args.dataset))

        # a single dataset
        train_dataset = dataset_dict[args.dataset](args, mode, scenes=args.scene)
        print('Train dataset {} has {} samples'.format(args.dataset, len(train_dataset)))
        train_sampler = DistributedSampler(train_dataset) if args.distributed else None

        return train_dataset, train_sampler

    elif mode in ['val', 'test']:

        print('eval dataset: {}'.format(args.dataset))
        print('eval scenes: {}'.format(args.scene))

        val_dataset = dataset_dict[args.dataset](
            args, mode,
            scenes=args.scene)
        print('Val dataset {} has {} samples'.format(args.dataset, len(val_dataset)))

        return val_dataset
    else:
        raise ValueError

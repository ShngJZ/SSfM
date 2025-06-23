import h5py, time
import torch.nn.functional as F
def acquire_depth_range(settings, data_dict):
    if settings.nerf.depth.param == 'inverse':
        depth_range = settings.nerf.depth.range
    elif settings.nerf.depth.param == 'monoguided':
        depth_range = data_dict['depth_est']
    elif settings.nerf.depth.param == 'datainverse':
        # use the one from the dataset and inverse
        depth_range = [1 / data_dict.depth_range[0, 0].item(), 1 / data_dict.depth_range[0, 1].item()]
    else:
        # use the one from the dataset
        depth_range = data_dict.depth_range[0]
        raise NotImplementedError()

    return depth_range


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, padding=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // padding) + 1) * padding - self.ht) % padding
        pad_wd = (((self.wd // padding) + 1) * padding - self.wd) % padding
        self._pad = [0, pad_wd, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def h5_open_wait(h5file):
    wait = 60 # Sleep one minute
    max_wait = 60 * 60 # Wait max one hour
    waited = 0

    while True:
        try:
            h5f = h5py.File(h5file, 'r')
            return h5f

        except FileNotFoundError:
            print('Error: HDF5 File not found')
            return None

        except OSError:
            if waited < max_wait:
                print(f'Error: HDF5 File locked, sleeping {wait} seconds...')
                time.sleep(wait)
                waited += wait
            else:
                print(f'waited too long= {waited} secs')
                return None

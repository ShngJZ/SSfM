import torch
import lpips
from easydict import EasyDict as edict
from typing import Callable, List, Tuple, Dict, Any


class Loss:
    """Overall loss module. Will compute each loss from a predefined list, 
        as well as combine their values. 
        
        Args:
            loss_modules (List): list of loss functions
    """
    def __init__(self, loss_modules: List[Callable[[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]], Any]]):

        self.loss_modules = loss_modules
    
    def compute_loss(self, opt: Dict[str, Any], data_dict: Dict[str, Any], output_dict: Dict[str, Any], 
                     iteration: int, mode: str=None, plot: bool=False, renderer=Any, do_log=False, **kwargs
                     ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Loops through all losses and computes them. At the end, computes final loss, as combination of the 
        different loss components. """
        loss = edict()
        stats_dict, plotting_dict = {}, {}
        for loss_module in self.loss_modules:
            if opt.loss_weight[loss_module.loss_tag] < -5:
                # Skip Loss too small
                continue

            loss_dict, stats_dict_, plotting_dict_ = \
                loss_module.compute_loss(
                    opt, data_dict, output_dict, iteration=iteration, mode=mode, plot=plot, renderer=renderer, do_log=do_log, **kwargs
                )
            loss.update(loss_dict)
            stats_dict.update(stats_dict_)
            plotting_dict.update(plotting_dict_)


        if opt.loss_weight.equalize_losses:
            loss = self.summarize_loss_w_equal_weights(opt, loss)
        else:
            loss = self.summarize_loss_w_predefined_weights(opt, loss)
        return loss, stats_dict, plotting_dict
    
    def get_flow_metrics(self) -> Dict[str, Any]:
        """Compute flow/correspondences metrics, if correspondences are used in any of the sub-modules. """
        stats = {}
        for loss_module in self.loss_modules:
            if hasattr(loss_module, 'get_flow_metrics'):
                stats_ = loss_module.get_flow_metrics()
                stats.update(stats_)
        return stats


    def summarize_loss_w_predefined_weights(self,opt: Dict[str, Any],loss_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Computes overall loss by summing the different loss terms, with predefined weights. """
        loss_all = 0.
        updated_loss = {}
        assert("all" not in loss_dict)
        # weigh losses

        for key in loss_dict:
            assert(key in opt.loss_weight)
            assert(loss_dict[key].shape==())

            if opt.loss_weight[key] is not None:
                assert not torch.isinf(loss_dict[key]),"loss {} is Inf".format(key)
                assert not torch.isnan(loss_dict[key]),"loss {} is NaN".format(key)
                if opt.loss_weight.parametrization == 'exp':
                    # weights are 10^w
                    w = 10**float(opt.loss_weight[key]) 
                else:
                    w = float(opt.loss_weight[key])
                weighted_loss = w* loss_dict[key]
                loss_all += weighted_loss
                updated_loss[key + '_after_w'] = weighted_loss
        loss_dict.update(all=loss_all)
        loss_dict.update(updated_loss)
        return loss_dict



vgg = lpips.LPIPS(net="vgg")

class BaseLoss:
    def __init__(self, device: torch.device):
        """ Base class for all losses. """
        self.device = device
        self.vgg = vgg.to(self.device)

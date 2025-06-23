import torch
from typing import Any, Optional, Dict, Tuple
from source.utils.torch import get_log_string
from source.training.engine.base_trainer import BaseTrainer


class IterBasedTrainer(BaseTrainer):
    def __init__(
        self,
        settings: Dict[str, Any],
        max_iteration: int,
        snapshot_steps: int,
        cudnn_deterministic: bool=True,
        autograd_anomaly_detection: bool=False,
        run_grad_check: bool=False,
        grad_acc_steps: int=1,
    ):
        super().__init__(
            settings,
            cudnn_deterministic=cudnn_deterministic,
            autograd_anomaly_detection=autograd_anomaly_detection,
            run_grad_check=run_grad_check,
            grad_acc_steps=grad_acc_steps,
        )
        self.max_iteration = max_iteration
        self.snapshot_steps = snapshot_steps
        self.val_steps = self.settings.val_steps
        self.scheduler_per_iteration = True

    def before_train(self) -> None:
        pass

    def after_train(self) -> None:
        pass

    def before_val(self) -> None:
        pass

    def after_val(self) -> None:
        pass

    def before_train_step(self, iteration: int, data_dict: Dict[str, Any]) -> None:
        pass

    def before_val_step(self, iteration: int, data_dict: Dict[str, Any]) -> None:
        pass

    def after_train_step(self, iteration: int, data_dict: Dict[str, Any], output_dict: Dict[str, Any], result_dict: Dict[str, Any]) -> None:
        pass

    def after_val_step(self, iteration: int, data_dict: Dict[str, Any], output_dict: Dict[str, Any], result_dict: Dict[str, Any]) -> None:
        pass

    def train_step(self, iteration: int, data_dict: Dict[str, Any]) -> Tuple[Dict, Dict, Dict]:
        pass

    def val_step(self, iteration: int, data_dict: Dict[str, Any]) -> Tuple[Dict, Dict, Dict]:
        pass

    def after_backward(self, net: torch.nn.Module, iteration: int, 
                       data_dict: Dict[str, Any]=None, output_dict: Dict[str, Any]=None, 
                       result_dict: Dict[str, Any]=None, gradient_clipping: float=None) -> bool:
        """ Returns if back propagation should actually happen for this step.
        Also can do some processing of the gradients like gradient clipping

        Args:
            net (nn.model, (list, tuple)): list of networks, or network 
            iteration (int): 
            data_dict (edict, optional):  Defaults to None.
            output_dict (edict, optional): Defaults to None.
            result_dict (edict, optional): Defaults to None.
            gradient_clipping (float, optional): Defaults to None.

        Returns:
            bool: do back_prop?
        """

        # gather all the network parameters
        if isinstance(net, (list, tuple)):
            parameters = []
            for i in range(len(net)):
                parameters += list(net[i].parameters())
        else:
            parameters = net.parameters()

        # check the gradients, if there are Nan or Inf
        do_backprop = self.check_invalid_gradients(net)

        # skipping gradients
        if do_backprop and hasattr(self.settings, 'skip_large_gradients') and self.settings.skip_large_gradients:
            if isinstance(parameters, torch.Tensor):
                parameters = [parameters]
            parameters = [p for p in parameters if p.grad is not None]
            device = parameters[0].grad.device
            norms = [p.grad.detach().abs().max().to(device) for p in parameters]
            max_ = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
            if max_ > self.settings.skip_large_gradients:
                do_backprop = False  # will skip optimizer

        # gradient clipping
        if do_backprop and gradient_clipping is not None:
            if self.settings.clip_by_norm:
                torch.nn.utils.clip_grad_norm_(parameters, gradient_clipping)
            else:
                # clip by value
                torch.nn.utils.clip_grad_value_(parameters, gradient_clipping)
        
        # printing gradients
        if hasattr(self.settings, 'print_gradients') and self.settings.print_gradients:
            norm_type = 2.0
            if isinstance(parameters, torch.Tensor):
                parameters = [parameters]
            parameters = [p for p in parameters if p.grad is not None]
            device = parameters[0].grad.device
            norms = [p.grad.detach().abs().max().to(device) for p in parameters]
            max = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))

            total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device)
                                                 for p in parameters]), norm_type)
            print('Max gradient={}, norm gradient={}'.format(max, total_norm))
        
        return do_backprop
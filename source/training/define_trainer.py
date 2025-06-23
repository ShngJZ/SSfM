import os
from typing import Dict, Any
from source.training.joint_pose_nerf_trainer import PoseAndNerfTrainerPerScene
from source.utils.config_utils import save_options_file


def define_trainer(
        args: Dict[str, Any],
        settings_model: Dict[str, Any],
        debug: bool=False,
        save_option: bool=True
):
    """Defines the trainer (NeRF with fixed ground-truth poses, NeRF with fixed
    colmap poses, joint pose-NeRF training)

    Args:
        args (edict): arguments from the command line. Importantly, contains
                      args.env
        settings_model (edict): config of the model
        debug (bool, optional): Defaults to False.
    """
    settings_model.update(args.args_to_update)

    assert settings_model.model == 'joint_pose_nerf_training'
    if 'scannet' in settings_model.dataset:
        settings_model.max_iter = 80000

    if debug:
        settings_model.vis_steps = 2    # visualize results (every N iterations)
        settings_model.log_steps = 2    # log losses and scalar states (every N iterations)
        settings_model.snapshot_steps = 5 
        settings_model.val_steps = 5 

    if save_option:
        save_options_file(settings_model, os.path.join(args.env.workspace_dir,  args.project_path), override='y')
    
    args.debug = debug
    args.update(settings_model)

    trainer = PoseAndNerfTrainerPerScene(args)
    return trainer

from easydict import EasyDict as edict
from source.utils.config_utils import override_options
from train_settings.default_config import get_joint_pose_nerf_default_config_llff



def get_config():
    default_config = get_joint_pose_nerf_default_config_llff()

    settings_model = edict()

    # camera options
    settings_model.camera = edict()
    settings_model.camera.pose_parametrization = 'two_columns_scale_optdepth'
    settings_model.camera.initial_pose = 'twoview_gpu_estdepth'
    # [twoview; gpu / opencv; gtscale / gtdepth / estdepth]

    # Sampling options
    settings_model.nerf = edict({'depth': edict()})
    settings_model.nerf.depth.param = 'datainverse'
    settings_model.nerf.feature_volume = True
    settings_model.nerf.feature_volume_size = [1, 2, 2, 128] # FeatureDim, DownsampleH, DownsampleW, Depth
    settings_model.nerf.rand_rays = int(2048 * 5)

    # scheduling
    settings_model.first_joint_pose_nerf_then_nerf = True
    settings_model.ratio_end_joint_nerf_pose_refinement = 1.0
    settings_model.barf_c2f = [0.4, 0.7]
    settings_model.start_iter = edict()
    settings_model.start_iter.pose = 0

    # dataset
    settings_model.dataset = 'scannet'
    settings_model.resize = None
    settings_model.llff_img_factor = 8

    # flow stuff
    settings_model.use_flow = True
    settings_model.flow_backbone = 'PDCNet'
    settings_model.use_monodepth = True
    settings_model.mondepth_backbone = 'Metric3DDepth'
    settings_model.mondepth_init = 'monodepth'

    # loss type
    settings_model.loss_type = 'triangulation'
    settings_model.matching_pair_generation = 'all_to_all'

    # triangulation loss settings
    settings_model.loss_triangulation = edict()
    settings_model.loss_triangulation.visibility_threshold = 0.1
    settings_model.dept_top_percent = 0.85
    settings_model.loss_triangulation.w_corres = 0.9
    settings_model.loss_triangulation.w_corres_scale = 10.0

    settings_model.loss_weight = edict()
    settings_model.loss_weight.triangulation = -1

    settings_model.arch = edict()
    settings_model.arch.layers_feat = None

    settings_model.min_conf_valid_corr = 0.95

    settings_model.norm_threshold_adjuster = 0.5
    settings_model.rel_norm_threshold_adjuster = None

    settings_model.nablate_epp = 100
    settings_model.nablate_prj = 200
    settings_model.max_res_scale = 1.0
    settings_model.prj_th = 2.0
    settings_model.c3D_th = 0.050
    settings_model.topk = 128
    settings_model.noreroot = False

    settings_model.skip_epp_votes = True
    settings_model.skip_prj_votes = False
    settings_model.skip_c3D_votes = True
    return override_options(default_config, settings_model)
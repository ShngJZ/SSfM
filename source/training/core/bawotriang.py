import torch
import numpy as np
import cupy as cp
from source.training.core.triangulation_loss import padding_pose

epp_scoring_kernel = cp.RawKernel(r'''
template<typename T>
__device__ void ComputeError(const T *q,
                             const T *qp,
                             const T *E,
                             T &sum,
                             T &error) {
  // Compute Ex
  T Ex[3];
  for (int k = 0; k < 3; k++) {
    sum = 0.0;
    for (int l = 0; l < 3; l++) {
      sum += E[k * 3 + l] * q[l];
    }
    Ex[k] = sum;
  }
  // Compute x^TE
  T xE[3];
  for (int k = 0; k < 3; k++) {
    sum = 0.0;
    for (int l = 0; l < 3; l++) {
      sum += qp[l] * E[l * 3 + k];
    }
    xE[k] = sum;
  }
  // Compute xEx
  T xEx = 0.0;
  for (int k = 0; k < 3; k++) {
    xEx += qp[k] * Ex[k];
  }
  // Compute Sampson error
  T d = sqrt(Ex[0]*Ex[0]+Ex[1]*Ex[1]+xE[0]*xE[0]+xE[1]*xE[1]);
  error = xEx / d;

  if (error < 0.0) error = -error;
}

extern "C" __global__
void epp_scoring_function(
    const float* pts_source, 
    const float* pts_target, 
    const float* essen_mtx, 
    float* epp_score,
    int* to_compute_pair,
    float norm_threshold,
    const int npair,
    const int n_cand_pair,
    const int npts
    ) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    float sum;
    float error;
    for (; tid < npair * n_cand_pair; tid += blockDim.x * gridDim.x) {
        
        if (to_compute_pair[tid / n_cand_pair] == 0){
            continue;
        }
        
        for (int i = 0; i < npts; ++i){

            ComputeError<float>(
            &pts_source[tid / n_cand_pair * npts * 3 + i * 3], 
            &pts_target[tid / n_cand_pair * npts * 3 + i * 3], 
            &essen_mtx[tid * 9], 
            sum,
            error
            );

            if (error < norm_threshold){
                epp_score[tid] += 1;
            }
            
            
        }
    }
}
''', 'epp_scoring_function')


def epp_scoring_function(
        pts_source_normed,
        pts_target_normed,
        essen_mtx,
        norm_threshold,
        nfrm,
        to_compute_pair,
        epp_score=None
):
    npair, npts, _, _ = pts_source_normed.shape
    nenum, nablate = essen_mtx.shape[1], essen_mtx.shape[2]
    device = pts_source_normed.device

    if torch.sum(to_compute_pair == 0) == 0:
        assert epp_score is None
    else:
        assert epp_score is not None

    pts_source_normed, pts_target_normed = pts_source_normed.view([npair, npts, 3]), pts_target_normed.view([npair, npts, 3])
    essen_mtx = essen_mtx.view([npair, -1, 3, 3])

    n_cand_pair = essen_mtx.shape[1]
    if epp_score is None:
        epp_score = torch.zeros([npair, n_cand_pair], device=device)
    else:
        epp_score[to_compute_pair == 1] = 0

    pts_source_normed, pts_target_normed, essen_mtx, epp_score = pts_source_normed.contiguous(), pts_target_normed.contiguous(), essen_mtx.contiguous(), epp_score.contiguous()
    pts_source_normed, pts_target_normed, essen_mtx, epp_score = cp.asarray(pts_source_normed), cp.asarray(pts_target_normed), cp.asarray(essen_mtx), cp.asarray(epp_score)
    to_compute_pair = cp.asarray(to_compute_pair)

    epp_scoring_kernel(
        (8,), (1024,),
        (pts_source_normed, pts_target_normed, essen_mtx, epp_score, to_compute_pair, cp.float32(norm_threshold), cp.int32(npair), cp.int32(n_cand_pair), cp.int32(npts))
    )

    epp_score = torch.as_tensor(epp_score, device=device)
    epp_score = epp_score.view([npair, nenum, nablate])
    return epp_score

prj_scoring_kernel = cp.RawKernel(r'''
template<typename T>
__device__ void ComputeAndVote(
                            const T *pts_source,
                            const T *pts_target,
                            const T *depth,
                            T *fundm,
                            T *pure_rotation,
                            T *intrinsic_t, 
                            T *M,
                            T *prj_score,
                            T *debugger,
                            T nablate_prj,
                            const T max_res_scale,
                            const T prj_threshold
                            ) {
                 
  float sum;
  float swap;
  float vec1[2];
  float pts_target_eppline[2];
  
  float pts_target_eppline_border[2];
  float maxscale;
  float minscale;
  
  float mx[3];
  
  // Compute fundm @ pts_source, first two rows only, as third one is always 1
  for (int k = 0; k < 2; k++) {
    sum = 0.0;
    for (int l = 0; l < 3; l++) {
      sum += fundm[k * 3 + l] * pts_source[l];
    }
    vec1[k] = sum;
  }
  
  // Normalized epipolar line direction vector
  sum = sqrt(vec1[0] * vec1[0] + vec1[1] * vec1[1] + 1e-20);
  swap = vec1[0];
  vec1[0] = -vec1[1] / sum;
  vec1[1] = swap / sum;
  
  // Compute pure_rotation @ pts_source, one point on epipolar line
  for (int k = 0; k < 3; k++) {
        sum = 0.0;
        for (int l = 0; l < 3; l++) {
          sum += pure_rotation[k * 3 + l] * pts_source[l];
        }
        mx[k] = sum;
  }
  mx[0] = mx[0] / (mx[2] + 1e-10);
  mx[1] = mx[1] / (mx[2] + 1e-10);
  
  // Project point - epipolar point vector to epipolar line 
  // pts_target_eppline    
  // pts_target_eppline[0] = pts_target[0] - epppoint[0];
  // pts_target_eppline[1] = pts_target[1] - epppoint[1];
  pts_target_eppline[0] = pts_target[0] - mx[0];
  pts_target_eppline[1] = pts_target[1] - mx[1];
  sum = pts_target_eppline[0] * vec1[0] + pts_target_eppline[1] * vec1[1];
  sum = sum / (vec1[0] * vec1[0] + vec1[1] * vec1[1]);
  pts_target_eppline[0] = vec1[0] * sum;
  pts_target_eppline[1] = vec1[1] * sum;
  // pts_target_eppline[0] = pts_target_eppline[0] + epppoint[0];
  // pts_target_eppline[1] = pts_target_eppline[1] + epppoint[1];
  pts_target_eppline[0] = pts_target_eppline[0] + mx[0];
  pts_target_eppline[1] = pts_target_eppline[1] + mx[1];
  
  for (int k = 0; k < 3; k++) {
        sum = 0.0;
        for (int l = 0; l < 3; l++) {
          sum += M[k * 3 + l] * pts_source[l];
        }
        mx[k] = sum;
  }
  
  sum = sqrt((pts_target_eppline[0] - pts_target[0]) * (pts_target_eppline[0] - pts_target[0]) + (pts_target_eppline[1] - pts_target[1]) * (pts_target_eppline[1] - pts_target[1]) + 1e-20);
  
  if (sum < prj_threshold){
    swap = sqrt(prj_threshold * prj_threshold - sum * sum + 1e-20);
    vec1[0] = vec1[0] * swap;
    vec1[1] = vec1[1] * swap;
    
    pts_target_eppline_border[0] = pts_target_eppline[0] + vec1[0];
    pts_target_eppline_border[1] = pts_target_eppline[1] + vec1[1];
    maxscale = depth[0] * (mx[2] * pts_target_eppline_border[0] - mx[0]) / (intrinsic_t[0] - pts_target_eppline_border[0] * intrinsic_t[2] + 1e-20);
        
    pts_target_eppline_border[0] = pts_target_eppline[0] - vec1[0];
    pts_target_eppline_border[1] = pts_target_eppline[1] - vec1[1];
    minscale = depth[0] * (mx[2] * pts_target_eppline_border[0] - mx[0]) / (intrinsic_t[0] - pts_target_eppline_border[0] * intrinsic_t[2] + 1e-20);
    
    // vanishing point resides inside range, skip and record the anomaly                     
    maxscale = maxscale / max_res_scale * nablate_prj;
    minscale = minscale / max_res_scale * nablate_prj;
            
    maxscale = (int) (maxscale + 0.5);
    minscale = (int) (minscale + 0.5);
    
    if (maxscale < 0){
        maxscale = 0;
    }
    
    if (maxscale > nablate_prj){
        maxscale = nablate_prj;

    }
    
    if (minscale < 0){
        minscale = 0;
    }
    
    if (minscale > nablate_prj){
        minscale = nablate_prj;
    }
    
    if (minscale > maxscale){
        debugger[0]++;
    }
    
    if ((minscale != maxscale) && (maxscale == nablate_prj)){
        debugger[1]++;
    }
    
    // Vote
    for (int i = minscale; i < maxscale; i++) {
        prj_score[i]++;
    }
    
  }
  
}

extern "C" __global__
void prj_scoring_function(
    const float* pts_source, 
    const float* pts_target, 
    const float* depth,
    float* fundm, 
    float* pure_rotation,
    float* intrinsic_t, 
    float* M,
    const int* to_compute_pair,
    float* prj_score,
    float* debugger,
    const int npair,
    const int n_cand_pair,
    const int npts,
    int nablate_prj,
    const float max_res_scale,
    const float prj_threshold
    ) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int pair_idx;
    for (; tid < npair * n_cand_pair; tid += blockDim.x * gridDim.x) {
        
        if (to_compute_pair[tid / n_cand_pair] == 0){
            continue;
        }
        
        for (int i = 0; i < npts; ++i){
            
            pair_idx = tid - tid / n_cand_pair * n_cand_pair;
            
            ComputeAndVote<float>(
                &pts_source[tid / n_cand_pair * npts * 3 + i * 3], 
                &pts_target[tid / n_cand_pair * npts * 3 + i * 3], 
                &depth[tid / n_cand_pair * npts * 1 + i * 1],
                &fundm[tid / n_cand_pair * n_cand_pair * 9 + pair_idx * 9],
                &pure_rotation[tid / n_cand_pair * n_cand_pair * 9 + pair_idx * 9],
                &intrinsic_t[tid / n_cand_pair * n_cand_pair * 3 + pair_idx * 3],
                &M[tid / n_cand_pair * n_cand_pair * 9 + pair_idx * 9],
                &prj_score[tid / n_cand_pair * n_cand_pair * nablate_prj + pair_idx * nablate_prj],
                &debugger[tid / n_cand_pair * n_cand_pair * 2 + pair_idx * 2],
                (float) nablate_prj,
                max_res_scale,
                prj_threshold
            );

        }
    }
}
''', 'prj_scoring_function')

def prj_scoring(
        pts_source,
        pts_target,
        depth,
        fundm,
        Rc,
        ntc,
        pure_rotation,
        intrinsic_t,
        M,
        intr,
        nablate_prj,
        max_res_scale,
        prj_th,
        nfrm,
        to_compute_pair,
        prj_score
):
    npair, npts, _ = pts_source.shape
    nenum, nablate = ntc.shape[1], ntc.shape[2]
    device = ntc.device

    if torch.sum(to_compute_pair == 0) == 0:
        assert prj_score is None
    else:
        assert prj_score is not None

    intrinsic_t, M, pure_rotation = intrinsic_t.view([npair, nenum, nablate, 3]), M.view([npair, nenum, 1, 3, 3]), pure_rotation.view([npair, nenum, 1, 3, 3])
    M, pure_rotation = M.expand([-1, -1, nablate, -1, -1]), pure_rotation.expand([-1, -1, nablate, -1, -1])

    fundm, pure_rotation = fundm.view([npair, -1, 3, 3]), pure_rotation.contiguous().view([npair, -1, 3, 3])
    n_cand_pair = fundm.shape[1]

    # Init Debug and Projection Scores
    debugger = torch.zeros([npair, n_cand_pair, 2], device=device)
    if prj_score is None:
        prj_score = torch.zeros([npair, n_cand_pair, nablate_prj], device=device)
    else:
        prj_score[to_compute_pair == 1] = 0

    pts_source, pts_target, fundm, debugger, pure_rotation = pts_source.contiguous(), pts_target.contiguous(), fundm.contiguous(), debugger.contiguous(), pure_rotation.contiguous()
    pts_source, pts_target, fundm, debugger, pure_rotation = cp.asarray(pts_source), cp.asarray(pts_target), cp.asarray(fundm), cp.asarray(debugger), cp.asarray(pure_rotation)

    intrinsic_t, M, prj_score, depth = intrinsic_t.view([npair, n_cand_pair, 3]).contiguous(), M.contiguous().view([npair, n_cand_pair, 3, 3]).contiguous(), prj_score.contiguous(), depth.contiguous()
    intrinsic_t, M, prj_score, depth = cp.asarray(intrinsic_t), cp.asarray(M), cp.asarray(prj_score), cp.asarray(depth)
    to_compute_pair = cp.asarray(to_compute_pair)

    prj_scoring_kernel(
        (8,), (1024,),
        (pts_source, pts_target, depth, fundm, pure_rotation, intrinsic_t, M, to_compute_pair, prj_score, debugger,
            cp.int32(npair), cp.int32(n_cand_pair), cp.int32(npts), cp.int32(nablate_prj), cp.float32(max_res_scale), cp.float32(prj_th))
    )

    """
    # Points Around Epipolar Line and Points Outrange Max
    debugger = torch.Tensor(debugger).sum(dim=[0, 1])
    print("%d points around epipole and %d points outbound" % (debugger[0].item(), debugger[1].item()))

    while True:
        rnd1, rnd2, rnd3, rnd4 = np.random.randint(npair), np.random.randint(n_cand_pair), np.random.randint(npts), np.random.randint(nablate_prj)
        if prj_score[rnd1, rnd2, rnd4] > 100:
            break
    valid, pts_target_eppline_maxscale, pts_target_eppline_minscale, pts_target_eppline_ref, vec1, vec2, scale_th_target2prj_dist, maxscale, minscale, mx, counts = \
    scale_estimation(
        torch.as_tensor(fundm, device=device)[rnd1, rnd2].view([1, 3, 3]),
        torch.as_tensor(intrinsic_t, device=device)[rnd1, rnd2].view([1, 3, 1]),
        torch.as_tensor(M, device=device)[rnd1, rnd2].view([1, 3, 3]),
        torch.as_tensor(pure_rotation, device=device)[rnd1, rnd2].view([1, 3, 3]),
        torch.as_tensor(pts_source, device=device)[rnd1].view([1, npts, 3]),
        torch.as_tensor(pts_target, device=device)[rnd1].view([1, npts, 3]),
        torch.as_tensor(depth, device=device)[rnd1].view([1, npts, 1]),
        npts, prj_th, nablate_prj, max_res_scale
    )
    diff = (torch.Tensor(prj_score[rnd1, rnd2]).cuda() - counts.cuda()).abs().max()

    scale_ck = (rnd4 + 0.5) / nablate_prj * max_res_scale
    intr_s = torch.eye(4)
    intr_s[0:3, 0:3] = intr.squeeze()
    R_ck = Rc.view([npair, nenum, 1, 3, 3]).expand([-1, -1, nablate, -1, -1])
    R_ck = R_ck.contiguous().view([npair, n_cand_pair, 3, 3]).contiguous()
    R_ck = R_ck[rnd1, rnd2]
    ntc_ck = ntc.contiguous().view([npair, n_cand_pair, 3, 1]).contiguous()
    ntc_ck = ntc_ck[rnd1, rnd2]
    pose_ck = torch.eye(4)
    pose_ck[0:3, 0:3] = R_ck
    pose_ck[0:3, 3:4] = ntc_ck * scale_ck
    relpose_ck = intr_s @ pose_ck @ intr_s.inverse()

    pts_ss = torch.as_tensor(pts_source[rnd1] * depth[rnd1]).cuda()
    pts_ss = kornia.geometry.convert_points_to_homogeneous(pts_ss)
    pts_ss_prj = pts_ss @ relpose_ck.cuda().T
    pts_ss_prj = kornia.geometry.convert_points_from_homogeneous(kornia.geometry.convert_points_from_homogeneous(pts_ss_prj))
    dist_ck = pts_ss_prj - torch.as_tensor(pts_target[rnd1, :, 0:2]).cuda()
    dist_ck = torch.sum(dist_ck ** 2, dim=-1).sqrt()
    inlier1 = torch.sum(dist_ck < prj_th)
    inlier2 = prj_score[rnd1, rnd2, rnd4].item()
    print(inlier1, inlier2)
    """

    prj_score = torch.as_tensor(prj_score, device=device)
    prj_score = prj_score.view([npair, nenum, nablate, nablate_prj])
    return prj_score


def scale_estimation(fundm, intrinsic_t, M, pure_rotation, pts_source, pts_target, depthf, npts, scale_th, nablate_prj, max_res_scale):
    # This is a python version of the CUDA code
    import kornia
    eps, topk = 1e-10, 1
    device = fundm.device
    pts_source, pts_target, depthf = pts_source.view([1, npts, 3]), pts_target.view([1, npts, 3]), depthf.view([1, npts, 1])

    # Compute Epipolar Line and Epp point
    epp_line = pts_source @ fundm.transpose(-1, -2)
    point_from_rot = pts_source @ pure_rotation.transpose(-1, -2)
    point_from_rot = kornia.geometry.conversions.convert_points_from_homogeneous(point_from_rot)
    # point_on_line_check = (epp_line * kornia.geometry.conversions.convert_points_to_homogeneous(point_from_rot)).sum(dim=-1).abs()

    # Compute Target Projection to Line
    vec1x, vec1y, _ = torch.split(epp_line, 1, dim=-1)
    vec1 = torch.cat([-vec1y, vec1x], dim=-1)
    vec1 = vec1 / torch.sqrt(torch.sum(vec1 ** 2, dim=-1, keepdim=True))
    vec2 = (kornia.geometry.conversions.convert_points_from_homogeneous(pts_target) - point_from_rot)
    vec2prj = torch.sum(vec1 * vec2, dim=-1, keepdim=True) / torch.sum(vec1 ** 2, dim=-1, keepdim=True) * vec1
    pts_target_eppline = vec2prj + point_from_rot
    # point_on_line_check = (epp_line * kornia.geometry.conversions.convert_points_to_homogeneous(pts_target_eppline)).sum(dim=-1).abs()

    target2prj_dist = ((pts_target_eppline - kornia.geometry.conversions.convert_points_from_homogeneous(pts_target)) ** 2 + eps).sum(dim=-1, keepdim=True).sqrt()
    valid = target2prj_dist < scale_th

    pts_target_eppline_maxscale = pts_target_eppline + vec1 * (scale_th ** 2 - target2prj_dist ** 2).sqrt()
    # point_on_line_check = (epp_line * kornia.geometry.conversions.convert_points_to_homogeneous(pts_target_eppline_maxscale)).sum(dim=-1).abs()
    pts_target_eppline_minscale = pts_target_eppline - vec1 * (scale_th ** 2 - target2prj_dist ** 2).sqrt()
    # point_on_line_check = (epp_line * kornia.geometry.conversions.convert_points_to_homogeneous(pts_target_eppline_maxscale)).sum(dim=-1).abs()

    maxscale, mx = projection2scale(pts_source, pts_target_eppline_maxscale, depthf, intrinsic_t, M)
    minscale, __ = projection2scale(pts_source, pts_target_eppline_minscale, depthf, intrinsic_t, M)

    valid = valid * (maxscale > minscale)

    counts = torch.zeros([nablate_prj])
    minscale, maxscale = (minscale / max_res_scale * nablate_prj + 0.5).int(), (maxscale / max_res_scale * nablate_prj + 0.5).int()
    minscale, maxscale = torch.clamp(minscale, min=0, max=nablate_prj), torch.clamp(maxscale, min=0, max=nablate_prj)

    for i in range(len(valid[0])):
        if valid[0, i, 0]:
            for ii in range(int(minscale[0, i, 0].item()), int(maxscale[0, i, 0].item())):
                counts[ii] += 1

    return valid, pts_target_eppline_maxscale, pts_target_eppline_minscale, pts_target_eppline, \
        vec1, vec2, scale_th - target2prj_dist, maxscale, minscale, mx, counts

def projection2scale(pts_source, pts_target, depthf, intrinsic_t, M):
    eps = 1e-20
    xxf_target, yyf_target = torch.split(pts_target, 1, dim=-1)

    x, y, z = torch.split(intrinsic_t, 1, dim=1)

    m0x, m1x, m2x = torch.split(pts_source @ M.transpose(-1, -2), 1, dim=2)

    scale = depthf * (m2x * xxf_target - m0x) / (x - xxf_target * z + eps)
    # scale_prx = depthf * (m2x * xxf_target - m0x) / (x - xxf_target * z + self.eps)
    # scale_pry = depthf * (m2x * yyf_target - m1x) / (y - yyf_target * z + self.eps)
    return scale, pts_source @ M.transpose(-1, -2)


# ================================================================================= #
c3D_scoring_kernel = cp.RawKernel(r'''
template<typename T>
__device__ void ComputeAndVoteC3D(
                            const T *pts_source_3D,
                            const T *pts_target_3D,
                            const T *prjM_i2j_R,
                            const T *prjM_j,
                            const T *ntc,
                            T *c3D_score,
                            T *debugger,
                            T nablate_prj,
                            const T max_res_scale,
                            const T c3D_th
                            ) {

  float sum;
  float l3d_st[3];
  float s_center[3];
  
  float a, b, c, d, minscale, maxscale;
  
  // Compute prjM_i2j_R @ pts_source_3D, first two rows only, as third one is always 1
  for (int k = 0; k < 3; k++) {
    sum = 0.0;
    for (int l = 0; l < 3; l++) {
      sum += prjM_i2j_R[k * 3 + l] * pts_source_3D[l];
    }
    l3d_st[k] = sum;
  }
  
  // Compute prjM_j @ pts_target_3D, first two rows only, as third one is always 1
  for (int k = 0; k < 3; k++) {
    sum = 0.0;
    for (int l = 0; l < 3; l++) {
      sum += prjM_j[k * 3 + l] * pts_target_3D[l];
    }
    s_center[k] = sum;
  }
  
  a = ntc[0] * ntc[0] + ntc[1] * ntc[1] + ntc[2] * ntc[2];
  b = (l3d_st[0] * ntc[0] + l3d_st[1] * ntc[1] + l3d_st[2] * ntc[2]) - (s_center[0] * ntc[0] + s_center[1] * ntc[1] + s_center[2] * ntc[2]);
  b = 2.0 * b;
  c = (l3d_st[0] - s_center[0]) * (l3d_st[0] - s_center[0]) + (l3d_st[1] - s_center[1]) * (l3d_st[1] - s_center[1]) + (l3d_st[2] - s_center[2]) * (l3d_st[2] - s_center[2]);
  c = c - c3D_th * c3D_th;
  d = b * b - 4.0 * a * c;
  
  if (d > 0){
    minscale = - (b + sqrt(d)) / 2 / a;
    maxscale = - (b - sqrt(d)) / 2 / a;
    
    minscale = minscale / max_res_scale * nablate_prj;
    maxscale = maxscale / max_res_scale * nablate_prj;
    
    minscale = (int) (minscale + 0.5);
    maxscale = (int) (maxscale + 0.5);
    
    if (maxscale < 0){
        maxscale = 0;
    }
    
    if (maxscale > nablate_prj){
        maxscale = nablate_prj;

    }
    
    if (minscale < 0){
        minscale = 0;
    }
    
    if (minscale > nablate_prj){
        minscale = nablate_prj;
    }
    
    if (minscale > maxscale){
        sum = minscale;
        minscale = maxscale;
        maxscale = sum;
        debugger[0]++; // Document Anomaly Swap
    }
    
    for (int i = minscale; i < maxscale; i++) {
        c3D_score[i]++;
    }
    
    if ((minscale != maxscale) && (maxscale == nablate_prj)){
        debugger[1]++; // Document Anomaly Insufficient Ablation Range
    }
    
  }
  
}

extern "C" __global__
void c3D_scoring_function(
    const float* pts_source_3D, 
    const float* pts_target_3D, 
    const float* prjM_i2j_R, 
    const float* prjM_j,
    const float* ntc,
    float* c3D_score, 
    float* debugger,
    const int npair,
    const int n_cand_pair,
    const int npts,
    int nablate_prj,
    const float max_res_scale,
    const float c3D_th
    ) {
    
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int pair_idx;
    for (; tid < npair * n_cand_pair; tid += blockDim.x * gridDim.x) {

        for (int i = 0; i < npts; ++i){

            pair_idx = tid - tid / n_cand_pair * n_cand_pair;

            ComputeAndVoteC3D<float>(
                &pts_source_3D[tid / n_cand_pair * npts * 3 + i * 3], 
                &pts_target_3D[tid / n_cand_pair * npts * 3 + i * 3], 
                &prjM_i2j_R[tid / n_cand_pair * n_cand_pair * 9 + pair_idx * 9],
                &prjM_j[tid / n_cand_pair * n_cand_pair * 9 + pair_idx * 9],
                &ntc[tid / n_cand_pair * n_cand_pair * 3 + pair_idx * 3],
                &c3D_score[tid / n_cand_pair * n_cand_pair * nablate_prj + pair_idx * nablate_prj],
                // &debugger[tid / n_cand_pair * n_cand_pair * npts * 2 + pair_idx * npts * 2 + i * 2],
                &debugger[tid / n_cand_pair * n_cand_pair * 2 + pair_idx * 2],
                (float) nablate_prj,
                max_res_scale,
                c3D_th
            );

        }
    }
}
''', 'c3D_scoring_function')



def c3D_scoring(
        pts_source,
        pts_target,
        depth_viewi,
        depth_viewj,
        Rc,
        ntc,
        intr,
        nablate_prj,
        max_res_scale,
        c3D_th
):
    npair, npts, _ = pts_source.shape
    nenum, nablate = ntc.shape[1], ntc.shape[2]
    device = ntc.device

    Rc = Rc.expand([-1, -1, nablate, -1, -1]).contiguous()
    prjM_i2j_R = Rc @ intr.inverse()
    prjM_j = intr.inverse().expand([npair, nenum, nablate, 3, 3])

    prjM_i2j_R, prjM_j = prjM_i2j_R.view([npair, -1, 3, 3]), prjM_j.view([npair, -1, 3, 3])
    prjM_i2j_R, prjM_j = prjM_i2j_R.contiguous(), prjM_j.contiguous()
    prjM_i2j_R, prjM_j = cp.asarray(prjM_i2j_R), cp.asarray(prjM_j)
    n_cand_pair = prjM_i2j_R.shape[1]

    pts_source_3D, pts_target_3D = (pts_source * depth_viewi).contiguous(), (pts_target * depth_viewj).contiguous()
    pts_source_3D, pts_target_3D = cp.asarray(pts_source_3D), cp.asarray(pts_target_3D)

    c3D_score, debugger = torch.zeros([npair, n_cand_pair, nablate_prj], device=device), torch.zeros([npair, n_cand_pair, 2], device=device)
    c3D_score, debugger = cp.asarray(c3D_score), cp.asarray(debugger)

    Rc, ntc = Rc.view([npair, -1, 3, 3]).contiguous(), ntc.view([npair, -1, 3, 1]).contiguous()
    Rc, ntc = cp.asarray(Rc), cp.asarray(ntc)

    c3D_scoring_kernel(
        (8,), (1024,),
        (pts_source_3D, pts_target_3D, prjM_i2j_R, prjM_j, ntc, c3D_score, debugger,
            cp.int32(npair), cp.int32(n_cand_pair), cp.int32(npts), cp.int32(nablate_prj), cp.float32(max_res_scale), cp.float32(c3D_th))
    )
    """
    depth_viewi, pts_source = depth_viewi.contiguous(), pts_source.contiguous()
    depth_viewi, pts_source = cp.asarray(depth_viewi), cp.asarray(pts_source)

    intr = intr.expand([npair, nenum, nablate, -1, -1]).view([npair, -1, 3, 3]).contiguous()
    intr = cp.asarray(intr)
    
    while True:
        rnd1, rnd2 = np.random.randint(npair), np.random.randint(n_cand_pair)
        # if c3D_score[rnd1, rnd2].max() > 300:
        break
    l3d_st, s_center, a, b, c, d, tmin, tmax, minscale, maxscale, counts = \
    scale_estimation_c3D(
        torch.as_tensor(prjM_i2j_R, device=device)[rnd1, rnd2].view([1, 3, 3]),
        torch.as_tensor(prjM_j, device=device)[rnd1, rnd2].view([1, 3, 3]),
        torch.as_tensor(Rc, device=device)[rnd1, rnd2].view([1, 3, 3]),
        torch.as_tensor(ntc, device=device)[rnd1, rnd2].view([1, 3, 1]),
        torch.as_tensor(intr, device=device)[rnd1, rnd2].view([1, 3, 3]),
        torch.as_tensor(pts_source_3D, device=device)[rnd1].view([1, npts, 3]),
        torch.as_tensor(pts_target_3D, device=device)[rnd1].view([1, npts, 3]),
        torch.as_tensor(pts_source, device=device)[rnd1].view([1, npts, 3]),
        torch.as_tensor(depth_viewi, device=device)[rnd1].view([1, npts, 1]),
        npts, c3D_th, nablate_prj, max_res_scale
    )

    count_ref = torch.as_tensor(c3D_score, device=device)[rnd1, rnd2].view([1, nablate_prj])
    diff = (counts.cuda() - count_ref).abs().max()
    print("Difference between CUDA and Python imp is %d" % diff.item())
    """

    c3D_score = torch.as_tensor(c3D_score, device=device)
    c3D_score = c3D_score.view([npair, nenum, nablate, nablate_prj])
    return c3D_score



def scale_estimation_c3D(
        prjM_i2j_R, prjM_j, Rc, ntc, intr, pts_source_3D, pts_target_3D, pts_source, depth_viewi, npts, c3D_th, nablate_prj, max_res_scale
):
    # This is a python version of the CUDA code
    import kornia

    l3d_st = pts_source_3D @ prjM_i2j_R.transpose(-1, -2)

    # Point outside the images
    vec = ntc.view([1, 1, 3]).expand([-1, npts, -1])

    s_center = pts_target_3D @ prjM_j.transpose(-1, -2)

    a = torch.sum(vec ** 2, dim=-1, keepdim=True)
    b = 2.0 * (torch.sum(l3d_st * vec, dim=-1, keepdim=True) - torch.sum(s_center * vec, dim=-1, keepdim=True))
    c = torch.sum((l3d_st - s_center) ** 2, dim=-1, keepdim=True) - c3D_th ** 2
    d = b ** 2 - 4 * a * c

    valid = d > 0.0
    tmin = - (b + torch.sqrt(d)) / 2 / a
    tmax = - (b - torch.sqrt(d)) / 2 / a

    counts = torch.zeros([nablate_prj])
    minscale, maxscale = (tmin / max_res_scale * nablate_prj + 0.5).int(), (tmax / max_res_scale * nablate_prj + 0.5).int()
    minscale, maxscale = torch.clamp(minscale, min=0, max=nablate_prj), torch.clamp(maxscale, min=0, max=nablate_prj)

    for i in range(len(valid[0])):
        if valid[0, i, 0]:
            mins, maxs = int(minscale[0, i, 0].item()), int(maxscale[0, i, 0].item())
            if mins > maxs:
                from copy import copy
                swapper = lambda x, y: (copy(y), copy(x))
                mins, maxs = swapper(maxs, mins)
            assert mins <= maxs
            for ii in range(mins, maxs):
                counts[ii] += 1

    while True:
        rndidx = np.random.randint(len(counts))
        if counts[rndidx] > 30:
            break

    scale_ck = (rndidx + 0.5) / nablate_prj * max_res_scale
    poses_ck = padding_pose(torch.cat([Rc, ntc * scale_ck], dim=-1))
    prjM = poses_ck @ padding_pose(intr).inverse()
    pts_viewj_ck = prjM.unsqueeze(1) @ kornia.geometry.convert_points_to_homogeneous(pts_source_3D).unsqueeze(-1)
    pts_viewj_ck = kornia.geometry.convert_points_from_homogeneous(pts_viewj_ck.squeeze(-1))
    dist = torch.sqrt(torch.sum((pts_viewj_ck - s_center) ** 2, dim=-1))
    cnt = torch.sum(dist < c3D_th)
    cnt_ref = counts[rndidx]
    print("Difference between vote and computation: %d, %d, %d" % ((cnt - cnt_ref).abs().item(), cnt.item(), cnt_ref.item()))

    return l3d_st, s_center, a, b, c, d, tmin, tmax, minscale, maxscale, counts
import torch
import cv2, copy
import numpy as np
import kornia
def depth2scale(pts2d1, pts2d2, intrinsic, R, t, coorespondedDepth):
    intrinsic33 = intrinsic[0:3, 0:3]
    M = intrinsic33 @ R @ np.linalg.inv(intrinsic33)
    delta_t = (intrinsic33 @ t).squeeze()
    minval = 1e-6

    denom = (pts2d2[0, :] * (np.expand_dims(M[2, :], axis=0) @ pts2d1).squeeze() - (np.expand_dims(M[0, :], axis=0) @ pts2d1).squeeze()) ** 2 + \
            (pts2d2[1, :] * (np.expand_dims(M[2, :], axis=0) @ pts2d1).squeeze() - (np.expand_dims(M[1, :], axis=0) @ pts2d1).squeeze()) ** 2

    selector = (denom > minval)

    rel_d = np.sqrt(
        ((delta_t[0] - pts2d2[0, selector] * delta_t[2]) ** 2 +
         (delta_t[1] - pts2d2[1, selector] * delta_t[2]) ** 2) / denom[selector])
    alpha = np.mean(coorespondedDepth[selector]) / np.mean(rel_d)
    return alpha

def select_scale(scale_md, R, t, pts1_inliers, pts2_inliers, mdDepth_npf, intrinsic):
    intrinsicnp = np.eye(4)
    intrinsicnp[0:3, 0:3] = intrinsic
    numres = 100

    divrat = 5
    maxrat = 1
    pos = (np.exp(np.linspace(0, divrat, numres)) - 1) / (np.exp(divrat) - 1) * np.exp(maxrat) + 1
    neg = np.exp(-np.log(pos))
    tot = np.sort(np.concatenate([pos, neg, np.array([1e-5])]))

    scale_md_cand = scale_md * tot

    self_pose = np.eye(4)
    self_pose[0:3, 0:3] = R
    self_pose[0:3, 3:4] = t
    self_pose = np.expand_dims(self_pose, axis=0)
    self_pose = np.repeat(self_pose, axis=0, repeats=tot.shape[0])
    self_pose[:, 0:3, 3] = self_pose[:, 0:3, 3] * np.expand_dims(scale_md_cand, axis=1)

    pts3d = np.stack([pts1_inliers[:, 0] * mdDepth_npf, pts1_inliers[:, 1] * mdDepth_npf, mdDepth_npf, np.ones_like(mdDepth_npf)])
    pts3d = intrinsicnp @ self_pose @ np.linalg.inv(intrinsicnp) @ np.repeat(np.expand_dims(pts3d, axis=0), axis=0, repeats=tot.shape[0])
    rpjpts2dx = pts3d[:, 0, :] / (pts3d[:, 2, :] + 1e-12)
    rpjpts2dy = pts3d[:, 1, :] / (pts3d[:, 2, :] + 1e-12)

    rprjdist = np.sqrt((rpjpts2dx - np.expand_dims(pts2_inliers[:, 0], axis=0)) ** 2 + (rpjpts2dy - np.expand_dims(pts2_inliers[:, 1], axis=0)) ** 2)
    inlierth = 1
    rprj_inlierc = np.sum(rprjdist < inlierth, axis=1)

    best = np.argmax(rprj_inlierc)
    return scale_md_cand[best], best

def t2T(t):
    T = np.zeros([3, 3])
    T[0, 1] = -t[2]
    T[0, 2] = t[1]
    T[1, 0] = t[2]
    T[1, 2] = -t[0]
    T[2, 0] = -t[1]
    T[2, 1] = t[0]
    return T

def npose2pose(pts1, pts2, mdn, intrinsic, npose, inliers):
    R, t = npose[0:3, 0:3], npose[0:3, 3:4]

    # Estimate Scale
    inliers_mask = inliers == 1
    pts1_inliers = pts1[inliers_mask, :].T
    pts2_inliers = pts2[inliers_mask, :].T

    pts1_inliers = np.concatenate([pts1_inliers, np.ones([1, pts1_inliers.shape[1]])], axis=0)
    pts2_inliers = np.concatenate([pts2_inliers, np.ones([1, pts2_inliers.shape[1]])], axis=0)
    mdn_inlier = mdn[inliers_mask]

    intrinsic33 = intrinsic[0:3, 0:3]

    epppoint = intrinsic33 @ t
    epppoint[0, 0] = epppoint[0, 0] / epppoint[2, 0]
    epppoint[1, 0] = epppoint[1, 0] / epppoint[2, 0]
    epppoint[2, 0] = 1

    E_init = t2T(t) @ R
    F_init = np.linalg.inv(intrinsic33).T @ E_init @ np.linalg.inv(intrinsic33)
    eppline = F_init @ pts1_inliers

    vec1 = np.stack([-eppline[1, :], eppline[0, :]], axis=0)
    vec2 = pts2_inliers[0:2, :] - epppoint[0:2, :]
    vecp = np.sum(vec1 * vec2, axis=0) / np.sum(vec1[0:2, :] ** 2, axis=0) * vec1
    pts2_inliers_new = vecp + epppoint[0:2, :]

    # Initializtion of camera scale
    scale_md = depth2scale(
        pts1_inliers,
        pts2_inliers_new,
        intrinsic,
        R, t,
        mdn_inlier
    )

    scale_md, bestid = select_scale(
        scale_md,
        R, t,
        pts1, pts2,
        mdn,
        intrinsic
    )

    pose = np.eye(4)
    pose[0:3, 0:3] = R
    pose[0:3, 3:4] = scale_md * t
    pose = pose.astype(np.float32)
    return pose

import glob
import os
import pdb
import numpy as np
import cv2
HPATCHES_LOCAL = '/Users/cadar/Documents/hpatches-benchmark/'

def getGT(kps, H):

    if kps.shape[0] > 0:
        ones = np.ones(kps.shape[0])
        pts = np.vstack([kps.T,ones])
        warped = np.matmul(H, pts)
        warped = warped / warped[2]
        warped = warped.T
        return warped[:, :2]
    else:
        return np.empty(0)

def getViewpoint():
    folders = sorted(glob.glob(HPATCHES_LOCAL+"data/hpatches-sequences-release/v_*"))
    files = {}
    for fold in folders:
        data_name = fold.split("/")[-1]
        files[data_name] = sorted(glob.glob(os.path.join(fold, "*.ppm")))
        # print(files[data_name])

    return files

def getIlumination():
    folders = sorted(glob.glob(HPATCHES_LOCAL+"data/hpatches-sequences-release/i_*"))
    files = {}
    for fold in folders:
        data_name = fold.split("/")[-1]
        files[data_name] = sorted(glob.glob(os.path.join(fold, "*.ppm")))
        # print(files[data_name])

    return files

def getViewpointPatches():
    folders = sorted(glob.glob(HPATCHES_LOCAL+"data/hpatches-sequences/v_*"))
    files = {}
    for fold in folders:
        data_name = fold.split("/")[-1]
        files[data_name] = sorted(glob.glob(os.path.join(fold, "*.png")))
        # print(files[data_name])

    return files

def getIluminationPatches():
    folders = sorted(glob.glob(HPATCHES_LOCAL+"data/hpatches-sequences/i_*"))
    files = {}
    for fold in folders:
        data_name = fold.split("/")[-1]
        files[data_name] = sorted(glob.glob(os.path.join(fold, "*.png")))
        # print(files[data_name])

    return files

def plotKps(img, kps_np, color=(0,0,255)):
    s = 3
    ploted = img.copy()
    for kp in kps_np.astype(int):
        ploted = cv2.circle(ploted, tuple(kp), s, color, -1)
        ploted = cv2.circle(ploted, tuple(kp), s, (0,0,0), 1)
    return ploted

def plotGtMatch(img, kps, kps_gt):
    s = 4
    red=(0,0,255)
    green=(0,255, 0)
    ploted = img.copy()
    for i in range(kps_gt.shape[0]):

        ploted = cv2.line(ploted, tuple(kps_gt[i].astype(int)), tuple(kps[i].astype(int)), (255,0,0))

        ploted = cv2.circle(ploted, tuple(kps[i].astype(int)), s, red, -1)
        ploted = cv2.circle(ploted, tuple(kps[i].astype(int)), s, (0,0,0), 1)

        ploted = cv2.circle(ploted, tuple(kps_gt[i].astype(int)), s, green, -1)
        ploted = cv2.circle(ploted, tuple(kps_gt[i].astype(int)), s, (0,0,0), 1)


    return ploted
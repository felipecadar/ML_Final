import cv2
import numpy as np


def extractPatch(img, pos, w=50):
    f = w//2
    mx = img.shape[0]
    my = img.shape[1]
    x = int(pos[1])
    y = int(pos[0])

    x0 = max(x-f, 0)
    x1 = min(x+f, mx-1)
    y0 = max(y-f, 0)
    y1 = min(y+f, my-1)

    patch = img[x0:x1, y0:y1, :]

    px = (w - (x1-x0)) // 2
    py = (w - (y1-y0)) // 2

    px0 = px
    px1 = w - px0 - patch.shape[0]

    py0 = py
    py1 = w - py0 - patch.shape[1]

    pad = np.pad(patch, ((px0, px1), (py0,py1), (0,0)), 'constant', constant_values=0)

    # pdb.set_trace()
    return pad



bf = None
def makeMatch(kps1, desc1, kps2, desc2, idx=False, dmatch=False):

    # BFMatcher with default params
    global bf
    if bf is None:
        bf = cv2.BFMatcher()

    if len(desc1) == 0 or len(desc2) == 0:
        if dmatch:
            return dmatches
        return [], []

    matches = bf.knnMatch(desc1,desc2,k=2)

    mkpts1 = []
    mkpts2 = []
    # Apply ratio test
    idx1 = []
    idx2 = []
    dmatches = []
    for m,n in matches:
        if m.distance < 0.9*n.distance:
            dmatches.append(m)
            mkpts1.append(kps1[m.queryIdx])
            mkpts2.append(kps2[m.trainIdx])
            idx1.append(m.queryIdx)
            idx2.append(m.trainIdx)
    if dmatch:
        return dmatches

    if idx:
        return np.array(idx1), np.array(idx2)

    return np.array(mkpts1), np.array(mkpts2)

def evalMatch(mkpts2, gtkpts, total, K=10):
    if len(mkpts2) == 0:
        return np.zeros(K), np.zeros(K)

    diff = mkpts2.astype(int) - gtkpts.astype(int)
    diff = np.linalg.norm(diff, axis=1)
    
    ms = np.zeros(K)
    mma = np.zeros(K)
    for i, k in enumerate(range(K)):
        acc_at_k = (np.abs(diff) <= k).sum()
        ms[i] = acc_at_k/total
        mma[i] = acc_at_k/len(mkpts2)

    return ms, mma

def cvToNp(kps):
    return np.array([[kp.pt[0], kp.pt[1]] for kp in kps])


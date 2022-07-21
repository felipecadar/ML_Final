import cv2
import numpy as np
import os


def readKps(fpath):
    kps = []
    with open(fpath, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            size, angle, x, y, octave = line.replace("\n", "").split(",")
            kps.append(cv2.KeyPoint(x=float(x), y=float(y), size=float(size), octave=int(octave)))

    return kps

def writeCvKps(cv_kps, fname):
    with open(f"{fname}.kps", "w") as f:
        f.write('size, angle, x, y, octave\n')
        for kp in cv_kps:
            f.write('%.2f, %.3f, %.2f, %.2f, %d\n'%(1, kp.angle, kp.pt[0], kp.pt[1], kp.octave))

def writeDesc(desc, fname):
    # np.savetxt(fname+".desc", desc)
    np.savez_compressed(fname+".desc", desc=desc)

def readist(fname):
    with open(fname, 'r') as f:
        return [x.replace("\n", '') for x in f.readlines()]
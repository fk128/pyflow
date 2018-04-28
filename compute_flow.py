# Author: Deepak Pathak (c) 2016

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import argparse
import os
from os import listdir
from os.path import isfile, join
from blend_modes import blend_modes
from scipy import ndimage
# from __future__ import unicode_literals
import numpy as np
from PIL import Image

import pyflow


def main():
    # Flow Options:
    alpha = 0.012
    ratio = 0.5
    minWidth = 20
    nOuterFPIterations = 1
    nInnerFPIterations = 1
    nSORIterations = 20
    colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

    parser = argparse.ArgumentParser(
        description='compute flow for sequence of jpg images and save as npz')
    parser.add_argument(
        '--src',
        help='location of input images')
    parser.add_argument(
        '--out',
        help='destination to save the computed flow for each frame')
    parser.add_argument(
        '--psize',
        type=float,
        default=1,
        help='percent to resize the input image')
    args = parser.parse_args()

    imgFiles = [join(args.src, f) for f in listdir(args.src) if f.endswith('.jpg') and not f.startswith('.')]

    if len(imgFiles) == 0:
        print('No images provided.')
        exit(0)
    else:
        print(imgFiles)

    f1 = imgFiles[0]

    im1 = Image.open(f1)
    wpercent = args.psize  # (basewidth/float(im1.size[0]))
    wsize = int((float(im1.size[0]) * float(wpercent)))
    hsize = int((float(im1.size[1]) * float(wpercent)))

    # im1 = im1.resize((basewidth,hsize), Image.ANTIALIAS)

    if not os.path.exists(args.out):
        os.mkdir(args.out)

    for i in range(1, len(imgFiles)):
        f2 = imgFiles[i]

        print(f2)

        im1 = Image.open(f1)
        im2 = Image.open(f2)

        im1 = im1.resize((wsize, hsize), Image.ANTIALIAS)
        im2 = im2.resize((wsize, hsize), Image.ANTIALIAS)

        im1 = np.array(im1).astype(float) / 255.
        im2 = np.array(im2).astype(float) / 255.

        f1 = f2

        s = time.time()
        u, v, im2W = pyflow.coarse2fine_flow(
            im2, im1, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
            nSORIterations, colType)
        e = time.time()
        print('Time Taken: %.2f seconds for image of size (%d, %d, %d)' % (
            e - s, im1.shape[0], im1.shape[1], im1.shape[2]))

        z = np.zeros(u.shape)

        # flow = np.concatenate((u[..., None], v[..., None], z[..., None]), axis=2)

        # print(flow.max())
        # print(flow.min())

        np.savez(os.path.join(args.out, 'flow' + str(i) + '.npz'), u=u, v=v)


if __name__ == '__main__':
    main()

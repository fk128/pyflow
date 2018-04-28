# Author: Deepak Pathak (c) 2016

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import argparse
from os import listdir
import os
from os.path import isfile, join
from blend_modes import blend_modes
from scipy import ndimage
from scipy.ndimage import zoom
# from __future__ import unicode_literals
import numpy as np
from PIL import Image, TiffImagePlugin
import pyflow
from utils import printProgressBar

TiffImagePlugin.WRITE_LIBTIFF = True


def transform(im, field0, field1):
    skip = int(im.shape[1] / field0[0].shape[0])
    # print(skip)
    field0 = skip * zoom(field0, skip)
    field1 = skip * zoom(field1, skip)
    df = np.meshgrid(range(0, im.shape[1]), range(0, im.shape[0]))
    df[0] = df[0] + field0
    df[1] = df[1] + field1
    df[0], df[1] = df[1], df[0]

    out = np.ndarray(im.shape)
    for j in range(im.shape[2]):
        out[:, :, j] = ndimage.map_coordinates(im[:, :, j], df)
    return out, df


def transform0(im, df):
    out = np.ndarray(im.shape)
    for j in range(im.shape[2]):
        out[:, :, j] = ndimage.map_coordinates(im[:, :, j], df)
    return out


def transform1(im, field):
    df = np.meshgrid(range(0, im.shape[1]), range(0, im.shape[0]))
    df[0] = df[0] + field[0]
    df[1] = df[1] + field[1]
    df[0], df[1] = df[1], df[0]

    out = ndimage.map_coordinates(im, df)
    return out


def main():
    parser = argparse.ArgumentParser(
        description='Apply flow to images as save output as tiff')
    parser.add_argument(
        '--src',
        help='')
    parser.add_argument(
        '--layer',
        type=str,
        nargs='+',
        help='')
    parser.add_argument(
        '--out',
        default='.',
        help='')
    args = parser.parse_args()

    flowFiles = [join(args.src, f) for f in listdir(args.src) if f.endswith('.npz') and not f.startswith('.')]

    if not os.path.exists(args.out):
        os.mkdir(args.out)

    overlayLayers = [np.array(Image.open(layer)).astype(float) / 255. for layer in args.layer]

    names = [os.path.splitext(os.path.basename(layer))[0] for layer in args.layer]
    print(names)

    for i in range(0, len(flowFiles)):

        printProgressBar(i + 1, len(flowFiles), prefix='Progress:', suffix='Complete', length=50)
        f2 = flowFiles[i]

        data = np.load(f2)
        u = data['u']
        v = data['v']
        # print(u.shape)

        # out = Image.fromarray(np.uint8(im2W*255))
        # out.save('flow'+str(i)+'.png')
        # print(v.shape)
        # print(overlayLayer.shape)

        if i == 0:
            aU = u
            aV = v
        else:
            aU += u
            aV += v

        outOverlayLayer, df = transform(overlayLayers[0], aU, aV)
        Image.fromarray(np.uint8(outOverlayLayer * 255), mode='RGBA').save(
            os.path.join(args.out, names[0] + '_' + str(i) + '.tif'), compression="tiff_lzw")
        for j in range(1, len(overlayLayers)):
            outOverlayLayer = transform0(overlayLayers[j], df)
            Image.fromarray(np.uint8(outOverlayLayer * 255), mode='RGBA').save(
                os.path.join(args.out, names[j] + '_' + str(i) + '.tif'), compression="tiff_lzw")


if __name__ == '__main__':
    main()

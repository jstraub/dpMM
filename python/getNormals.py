# Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>
# Licensed under the MIT license. See the license file LICENSE.

import numpy as np
from js.data.rgbd.rgbdframe import RgbdFrame

import os
import argparse

# eg: 
# python getNormals.py -i ~/workspace/research/vpCluster/data/MIT_hallway_1_rgb


parser = argparse.ArgumentParser(description = 'obtain Normals from various file formats')
parser.add_argument('-i','--input', default="", help='input file')
#parser.add_argument('-nyu', action='store_true', help='switch to process the NYU dataset')
args = parser.parse_args()


rgbd = RgbdFrame(540)
rgbd.load(args.input)
rgbd.saveNormals('./normals.csv',False,'guy')


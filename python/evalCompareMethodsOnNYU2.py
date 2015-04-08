# Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>
# Licensed under the MIT license. See the license file LICENSE.
import numpy as np
import cv2
import os
import fnmatch



cfg=dict()
cfg['path'] = '/data/vision/scratch/fisher/jstraub/dpMM/nyu2'
cfg['base'] = ['DpNiw' , 'DpNiwSphereFull', 'spkm'];

index = open('/data/vision/fisher/data1/nyu_depth_v2/index.txt')
for i,name in enumerate(index):
  name =name[:-1]
  found = [None for base in cfg['base']]
  for file in os.listdir(cfg['path']):
    if not fnmatch.fnmatch(file, '*lbls.png'):
      continue
#    print file
    for j,base in enumerate(cfg['base']):
#      print  '{}*{}*.png'.format(name,base)
      if fnmatch.fnmatch(file, '{}*{}-*.png'.format(name,base)):
        found[j] = file
#        print file
  if not any(f is None for f in found): #found[0] is None and not found[1] is None:
    print found
    I = cv2.imread(os.path.join(cfg['path'],found[0]))
    for f in found[1::]:
      I = np.r_[I,cv2.imread(os.path.join(cfg['path'],f))]
#    print found
#    I0 = cv2.imread(os.path.join(cfg['path'],found[0]))
#    I1 = cv2.imread(os.path.join(cfg['path'],found[1]))
#    print I0.shape
#    print np.c_[I0,I1].shape
#    print np.r_[I0,I1].shape
    cv2.imshow(' vs. '.join(cfg['base']),I)
    cv2.waitKey()

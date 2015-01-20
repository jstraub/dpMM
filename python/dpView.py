# Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>
# Licensed under the MIT license. See the license file LICENSE.

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import cv2
import scipy.io
import subprocess as subp

import os, re, time
import argparse

import mayavi.mlab as mlab
from vpCluster.rgbd.rgbdframe import RgbdFrame
from vpCluster.manifold.sphere import Sphere
from js.utils.config import Config2String
from js.utils.plot.pyplot import SaveFigureAsImage

parser = argparse.ArgumentParser(description = 'DpMM modeling and viewer')
parser.add_argument('-s','--start', type=int, default=0, help='start image Nr')
parser.add_argument('-e','--end', type=int, default=0, help='end image Nr')
parser.add_argument('-K0', type=int, default=1, help='initial number of MFs')
parser.add_argument('-b','--base', default='DpNiwSphereFull', help='base distribution/algorithm')
parser.add_argument('-nyu', action='store_true', help='switch to process the NYU dataset')
args = parser.parse_args()

cfg=dict()
cfg['path'] = '/home/jstraub/workspace/research/vpCluster/data/nyu2/'
cfg['path'] = '/home/jstraub/workspace/research/vpCluster/data/'
cfg['path'] = '~/workspace/research/vpCluster/data/'
cfg['outputPath'] = '/data/vision/scratch/fisher/jstraub/dpMM/'
cfg['path'] = '/data/vision/scratch/fisher/jstraub/dpMM/nyu2/'
#cfg['base'] = 'DpNiwSphereFull';
#cfg['base'] = 'spkm';
#cfg['base'] = 'DpNiw';
cfg['base'] = args.base;
cfg['K'] = args.K0
cfg['T'] = 600
cfg['delta'] = 18.
cfg['nu'] =   3 + 10000.0 
seed = 214522
reRun = True
algo = 'guy' #'sobel'#'guy'
#mode = ['multi']
mode = ['multiFromFile']
mode = ['multi']
mode = ['single','disp']

if args.nyu:
  mode = ['multiFromFile']

if 'single' in mode:
  name = '2013-09-27.10:33:47' #
  name = '2013-10-01.19:25:00' # my room
  name = 'living_room_0000'
  name = 'study_room_0004_uint16'
  name = 'study_room_0005_uint16'
  name = 'home_office_0001_uint16'
  name = '2boxes_1'
  name = 'kitchen_0004'
  name = 'office_0008_uint16'
  name = '3boxes_moreTilted_0' #segments really well - has far distance!!! [k=4]
  name = 'table_1'
  name = 'kitchen_0016_252'
  names = [name]
elif 'multi' in mode:
  names = []
  for root,dirs,files in os.walk(cfg['path']):
    for f in files:
      ff = re.split('_',f)
      if ff[-1] == 'd.png':
        names.append('_'.join(ff[0:-1]))
        print 'adding {}'.format(names[-1])
##  names = ['home_office_0001_358','3boxes_moreTilted_0','couches_0','MIT_hallway_1','stairs_5','office_0008_17','stairs_5','MIT_hallway_0','kitchen_0007_132'] 
  names = ['kitchen_0015_252', 'living_room_0058_1301', 'bedroom_0085_1084', 'kitchen_0033_819', 'conference_room_0002_342', 'kitchen_0048_879']

elif 'multiFromFile' in mode:
  cfg['evalStart'] = args.start
  cfg['evalEnd'] = args.end
  indexPath = '/data/vision/fisher/data1/nyu_depth_v2/index.txt'
  cfg['path'] = '/data/vision/fisher/data1/nyu_depth_v2/extracted/'
  cfg['outputPath'] = '/data/vision/scratch/fisher/jstraub/dpMM/nyu2/'
  names =[]
  with open(indexPath) as f:
    allNames = f.read().splitlines() #readlines()
  for i in range(len(allNames)):
    if cfg['evalStart'] <= i and i <cfg['evalEnd']:
      names.append(allNames[i])
      print '@{}: {}'.format(len(names)-1,names[-1])
  print names
else:
  print 'no files in list'
  exit(1)

if 'disp' in mode:
  figm0 = mlab.figure(bgcolor=(1,1,1))
  figm1 = mlab.figure(bgcolor=(1,1,1))
  fig0 = plt.figure()

rndInds = np.random.permutation(len(names))
for ind in rndInds:
  name = names[ind]

  outName = cfg['outputPath']+name+'_'+Config2String(cfg).toString()
  if 'multiFromFile' in mode and os.path.isfile(outName):
    print 'skipping '+outName+' since it is already existing'
    continue;
  
  print 'processing '+cfg['path']+name
  rgbd = RgbdFrame(460.0) # correct: 540
  rgbd.load(cfg['path']+name)
  if 'disp' in mode:
    rgbd.showRgbd(fig=fig0)
  rgbd.getPc()
  print np.max(rgbd.d)
  nAll = rgbd.getNormals(algo=algo)
  n = nAll[rgbd.mask,:].T
  print n.shape
  D = n.shape[0]
  N = n.shape[1]
  
  dataPath = cfg['path']+name+'_normals.csv'
  np.savetxt(dataPath,n)
   
  alpha = np.ones(cfg['K'])*1. #*100.; 
  if cfg['base'] == 'NiwSphereUnifNoise':
    alpha[cfg['K']-1] = 1.0;
  nu = cfg['nu']#+N/10.

  if cfg['base'] == 'DpNiw':
    kappa = 1.0
    thetaa = np.zeros(D)
    Delta = nu * (cfg['delta']*np.pi)/180.0 * np.eye(D)
    params = np.array([nu,kappa])
    params = np.r_[params,thetaa.ravel(),Delta.ravel()]
  else:
    Delta = nu* (cfg['delta']*np.pi)/180.0 * np.eye(D-1)
    params = np.array([nu])
    params = np.r_[params,Delta.ravel()]

  #args = ['../build/dpSubclusterSphereGMM',
  args = ['../build/dpmmSampler',
    '-N {}'.format(N),
    '-D {}'.format(D),
    '-K {}'.format(cfg['K']),
    '-T {}'.format(cfg['T']),
    '--alpha '+' '.join([str(a) for a in alpha]),
    '--base '+cfg['base'],
    '--seed {}'.format(seed),
    '-i {}'.format(dataPath),
    '-o {}'.format(outName+'.lbl'),
    '--params '+' '.join([str(p) for p in params])]
  if reRun:
    print ' '.join(args)
    print ' --------------------- '
    time.sleep(1);
    err = subp.call(' '.join(args),shell=True)
    if err: 
      print 'error when executing'
      if not 'multiFromFile' in mode:
        raw_input()
      continue
  
  print outName+'.lbl'

  z = np.loadtxt(outName+'.lbl',delimiter=' ').astype(int)
  logLike = np.loadtxt(outName+'.lbl_jointLikelihood.csv',delimiter=' ')
  T = z.shape[0]
  for t in range(T):
    print t,logLike[t],np.bincount(z[t,:])
  for t in range(T):
    print t,logLike[t],np.bincount(z[t,:]/2)
  
  if cfg['base'] in ['DpNiw','DpNiwSphereFull']:
    zz = z[-1,:]/2
    t = T-1
    K = (np.max(z[-1,:])+1)/2
    means = np.loadtxt(outName+'.lbl_means.csv')                                    
    for k in range(K):
      for j in range(k+1,K):
        muk = means[t*3:(t+1)*3,3*k+2]
        muj = means[t*3:(t+1)*3,3*j+2]
        print np.arccos(muk.dot(muj))*180./np.pi
        if np.arccos(muk.dot(muj))*180./np.pi < 5:
          zz[zz==j] = k
  elif cfg['base'] in ['spkm', 'kmeans']:
    zz = z[-1,:]
    t = T-1
    K = (np.max(z[-1,:])+1)

  figL = plt.figure()
  I = np.zeros(rgbd.mask.shape)
  I[rgbd.mask] = zz + 1
  plt.imshow(I,cmap=cm.spectral,figure = figL)
#  plt.imshow(I,cmap=cm.hsv,figure = figL)
  SaveFigureAsImage(outName+'lbls.png',figL)
  

  if 'disp' in mode:
    figL.show()
#    plt.show()
    figm2 = rgbd.showWeightedNormals(algo=algo)
    fig = rgbd.showAxialSigma()
    fig = rgbd.showLateralSigma(theta=30.0)
    #fig = rgbd.bilateralDepthFiltering(theta=30.0)
    figm0 = rgbd.showPc(showNormals=True,algo=algo)
    figm1 = rgbd.showNormals()
    figm2 = rgbd.showNormals(as2D=True); figm2.show()
    M = Sphere(2)
    M.plotFanzy(figm1,1.0) 
    mlab.show(stop=True)
  elif  'multiFromFile' in mode and 'disp' in mode:
    figm0 = rgbd.showPc(figm=figm0,showNormals=True,algo=algo)
    figm1 = rgbd.showNormals(figm=figm1)
    M = Sphere(2)
    M.plotFanzy(figm1,1.0) 
    mlab.show(stop=True)
    mlab.clf(figm0)
    mlab.clf(figm1)
  
  plt.close(figL)


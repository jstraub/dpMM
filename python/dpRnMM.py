import numpy as np
import subprocess as subp

import matplotlib.pyplot as plt
import mayavi.mlab as mlab
from matplotlib.patches import Ellipse

import pdb, re, time
import os.path

from js.utils.plot.colors import colorScheme
from js.utils.config import Config2String

from vpCluster.manifold.karcherMean import karcherMeanSphere_propper
from vpCluster.manifold.sphere import Sphere

dataPath = './allSignalsV5.csv' #data from temperature sensor of cellphone
dataPath = None;
dataPath = './sphericalAssociatedPress.csv';
dataPath = '/data/vision/fisher/data1/wikipediaWordVectors/vectors_wiki.normalized_OnlyVect_colVects.csv'; # 200D
dataPath = './normals.csv';
dataPath = '/data/vision/fisher/data1/wikipediaWordVectors/vectors_wiki.20_onlyVects_colVects.csv'; #20D
dataPath = '/data/vision/fisher/data1/wikipediaWordVectors/vectors_wiki.20_prune100_onlyVects_colVects.csv'; #20D
dataPath = './rndSphereDataIw.csv';
dataPath = './rndSphereData.csv';
dataPath = './syntMMF.csv';
dataPath = './rndSphereDataIwUncertain.csv';
dataPath = './rndNonSpherical.csv';
dataPath = './rndSphereData2IwUncertain.csv';

cfg = dict()
cfg['base'] = 'kmeans';
cfg['base'] = 'DpNiw';

if dataPath is None:
  dataPath = './sphereData.csv'
  N = 4000
  D = 2
  x=np.random.standard_normal(size=[D,N/2])*0.1+1.*np.ones([D,N/2])
  #pdb.set_trace()
  x=np.c_[x,-1.*np.ones([D,N/2])+np.random.standard_normal([D,N/2])*0.1]
  x=x/np.sqrt((x**2).sum(axis=0))
  np.savetxt(dataPath,x,delimiter=' ')
  N = x.shape[1]
else:
  x=np.loadtxt(dataPath,delimiter=' ')
#  x = x[:,0:200]
  N = x.shape[1]
  D = x.shape[0]
  print np.sqrt((x**2).sum(axis=0))

if D < 3:
  fig = plt.figure()
  if D==2:
    plt.plot(x[0,:],x[1,:],'.')
  elif D==1:
    plt.plot(x[0,:],np.ones(x.shape[1]),'.')
  fig.show()

cfg['K'] = 1;
cfg['T'] = 500;
alpha = np.ones(cfg['K'])*1. #*100.; 
nu = D+1.0 #+N/10.

kappa = 1.0
thetaa = np.abs(np.mean(x,axis=1))
print thetaa
Delta = nu * np.eye(D)* (1.0*np.pi)/180.0
params = np.array([nu,kappa])
params = np.r_[params,thetaa.ravel(),Delta.ravel()]
args = ['../build/dpStickGMM',
  '-N {}'.format(N),
  '-D {}'.format(D),
  '-K {}'.format(cfg['K']),
  '-T {}'.format(cfg['T']),
  '--alpha '+' '.join([str(a) for a in alpha]),
  #'--base NiwSphereUnifNoise',
  '--base '+cfg['base'],
  '-i {}'.format(dataPath),
  '-o {}'.format(re.sub('csv','lbl',dataPath)),
  '--params '+' '.join([str(p) for p in params])]


print ' '.join(args)
print ' --------------------- '
time.sleep(3);
subp.call(' '.join(args),shell=True)

z = np.loadtxt(re.sub('csv','lbl',dataPath),dtype=int,delimiter=' ')
logLike = np.loadtxt(re.sub('csv','lbl',dataPath)+'_jointLikelihood.csv',delimiter=' ')
T = z.shape[0]
for t in range(T):
  print logLike[t],np.bincount(z[t,:])
for t in range(T):
  print logLike[t],np.bincount(z[t,:]/2)

if D <=2:
  fig = plt.figure()
  plt.subplot(2,1,1)
  plt.plot(np.max(z,axis=1)+1)
  plt.title('number of clusters')
  
  def covEllipse(x,sigmaMult=3.0):
    from scipy.linalg import eig
    mean = np.mean(x,axis=1)
    cov = np.cov(x) 
    print cov
    e,V=eig(cov)
    imax = np.argmax(e.real)
    imin = np.argmin(e.real)
    ang = np.arctan2(V[1,imax],V[0,imax])
    return Ellipse(xy=(mean[0],mean[1]),width=sigmaMult*np.sqrt(e[imax]),height=sigmaMult*np.sqrt(e[imin]),angle=ang/np.pi*180.0)
  
  ax = plt.subplot(2,1,2)
  counts = np.bincount(z[-1,:])
  K = float(np.max(z[-1,:]))
  for k,c in enumerate(counts):
    if D==2 and (z[-1,:]==k).sum()>0:
      plt.plot(x[0,z[-1,:]==k],x[1,z[-1,:]==k],'.',color=(k/K,1.-k/K,0))
      if (z[-1,:]==k).sum() > 1:
        ax.add_patch(covEllipse(x[:,z[-1,:]==k]))
    elif D==1 and (z[-1,:]==k).sum()>0:
      if (z[-1,:]==k).sum() > 1:
        plt.plot(x[0,z[-1,:]==k],np.ones((z[-1,:]==k).sum()),'.',color=(k/K,1.-k/K,0))
        plt.plot(np.mean(x[0,z[-1,:]==k]),np.ones(1)*2.,'x',color=(k/K,1.-k/K,0))
    elif D>2 and (z[-1,:]==k).sum()>0:
      print '---------- k={} ------------- '.format(k)
      print 'cov: \n{}'.format(np.cov(x[:,z[-1,:]==k]))
      print 'mean: \n{}'.format(np.mean(x[:,z[-1,:]==k]))
  plt.show()
elif D==3:
  figm = mlab.figure(bgcolor=(1,1,1))
  counts = np.bincount(z[-1,:])
  print counts
  if counts.sum() > 3000:
    mode = 'point'
  else:
    mode = 'sphere'
  t =0
  while True:
    print "-------- @t={} --------------".format(t)
    mlab.clf(figm)
    K = 2*((int(np.max(z[t,:]))+1)/2)
    zAll = np.floor(z[t,:]/2)
    print np.bincount(z[t,:])
    for k in range(K/2):
      if (zAll==k).sum() >1:
        mlab.points3d(x[0,z[t,:]==2*k],x[1,z[t,:]==2*k],x[2,z[t,:]==2*k],color=colorScheme('label')[k%7],figure=figm,scale_factor=0.05,mode='2dcross')
        mlab.points3d(x[0,z[t,:]==2*k+1],x[1,z[t,:]==2*k+1],x[2,z[t,:]==2*k+1],color=colorScheme('label')[k%7],figure=figm,scale_factor=0.03,mode='2dcircle')
    key = raw_input("press j,l, <space>:")
    if key =='j':
      t-=1
    elif key == 'l':
      t+=1
    elif key == 'E':
      t=T-1
    elif key == 'q':
      break
    elif key == ' ':
      mlab.show(stop=True)
    else:
      t+=1
    t=t%T

  mlab.show(stop=True)
#else:
#  fig = plt.figure()
#  plt.imshow(np.cov(x),interpolation='nearest')
#  plt.colorbar()
#  plt.title('full')
#  fig.show()
#
#  counts = np.bincount(z[-1,:])
#  K = float(np.max(z[-1,:]))
#  for k,c in enumerate(counts):
#    if (z[-1,:]==k).sum()>0:
#      print '---------- k={} ------------- '.format(k)
##      print 'cov: \n{}'.format(np.cov(x[:,z[-1,:]==k]))
##      print 'mean: \n{}'.format(np.mean(x[:,z[-1,:]==k],axis=1))
#      if K<100:
#        fig = plt.figure()
#        plt.imshow(np.cov(x[:,z[-1,:]==k]),interpolation='nearest')
#        plt.colorbar()
#        plt.title('k={}'.format(k))
#        fig.show()
  

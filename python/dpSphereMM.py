# Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>
# Licensed under the MIT license. See the license file LICENSE.
import numpy as np
from scipy.linalg import eig, logm
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

import matplotlib as mpl                                                        
mpl.rc('lines',linewidth=3.)

dataPath = './allSignalsV5.csv' #data from temperature sensor of cellphone
dataPath = None;
dataPath = './sphericalAssociatedPress.csv';
dataPath = '/data/vision/fisher/data1/wikipediaWordVectors/vectors_wiki.normalized_OnlyVect_colVects.csv'; # 200D
dataPath = '/data/vision/fisher/data1/wikipediaWordVectors/vectors_wiki.20_onlyVects_colVects.csv'; #20D
dataPath = './rndSphereDataIw.csv';
dataPath = './rndSphereData.csv';
dataPath = './rndNonSpherical.csv';
dataPath = './rndSphereDataElipticalCovs.csv';
dataPath = './rndSphereData1IwUncertain.csv';
dataPath = './syntMMF.csv';
dataPath = './rndSphereData2IwUncertain.csv';
dataPath = './rndSphereData1IwUncertain.csv';
dataPath = './rndSphereDataElipticalCovs.csv';
dataPath = './rndSphereData2IwUncertain.csv';
dataPath = './rndSphereDataIwUncertain.csv';
dataPath = './rndSphereData3IwUncertain.csv';
dataPath = './syntMMF.csv';
dataPath = './rndSphereData4IwUncertain.csv';
dataPath = './rndSphereData1IwUncertain.csv';
dataPath = './rndSphereDataElipticalCovs1.csv';
dataPath = './rndSphereData2IwUncertain.csv';
dataPath = './rndSphereDataElipticalCovs_test0.csv'; # smaller 10k with 30 classes
dataPath = './rndSphereDataIwUncertain.csv';
dataPath = './normals.csv';
dataPath = './rndSphereData2IwUncertain.csv';
dataPath = './rndSphereData4IwUncertain.csv';
dataPath = './rndSphereDataElipticalCovs3.csv'; # smaller 10k with 30 classes
dataPath = './rndSphereData1IwUncertain.csv';
dataPath = './rndSphereData5ElipticalCovs.csv'; # smaller 10k with 30 classes

dataPath = './rndSphereDataElipticalCovs4.csv'; # smaller 10k with 30 classes
dataPath = './rndSphereDataElipticalCovs3.csv'; # smaller 10k with 30 classes
dataPath = './rndSphereDataIwUncertain.csv';
dataPath = '/data/vision/fisher/data1/wikipediaWordVectors/vectors_wiki.normalized_OnlyVect_colVects.csv'; 
dataPath = '/data/vision/fisher/data1/wikipediaWordVectors/vectors_wiki.20_prune100_onlyVects_colVects.csv'; #20D

cfg = dict()
cfg['base'] = 'NiwSphereCuda';
cfg['base'] = 'spkmKarcher';
cfg['base'] = 'NiwSphereUnifNoise';
cfg['base'] = 'kmeans';
cfg['base'] = 'DpNiw';
cfg['base'] = 'DpNiwTangent';
cfg['base'] = 'NiwSphere';
cfg['base'] = 'DpNiwSphere';
cfg['base'] = 'spkm';
cfg['base'] = 'DpNiwSphereFull';

plotSubClusters = False
reRun = False
seed = 121352 #234345

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
  N = x.shape[1]
  D = x.shape[0]
  norms = np.sqrt((x**2).sum(axis=0))
  print norms
  for d in  range(D):
    x[d,:] /= norms


if D < 3:
  fig = plt.figure()
  if D==2:
    plt.plot(x[0,:],x[1,:],'.')
  elif D==1:
    plt.plot(x[0,:],np.ones(x.shape[1]),'.')
  fig.show()

if dataPath in ['./normals.csv','./syntMMF.csv', './rndSphereData1IwUncertain.csv', './rndSphereData2IwUncertain.csv', './rndSphereData3IwUncertain.csv', './rndSphereData4IwUncertain.csv', './rndSphereDataElipticalCovs_test0.csv']:
  cfg['K'] = 1;
  cfg['T'] = 100;
  alpha = np.ones(cfg['K'])*1. #*100.; 
  if cfg['base'] == 'NiwSphereUnifNoise':
    alpha[cfg['K']-1] = 1.0;
  nu = D+1.0 #+N/10.

  if cfg['base'] == 'DpNiw':
    kappa = 1.0
    thetaa = np.zeros(D)
    Delta = nu * np.eye(D)
    params = np.array([nu,kappa])
    params = np.r_[params,thetaa.ravel(),Delta.ravel()]
  else:
    Delta = nu* (1.*np.pi)/180.0 * np.eye(D-1)
    params = np.array([nu])
    params = np.r_[params,Delta.ravel()]

  outName,_ = os.path.splitext(dataPath)
  outName += '_'+Config2String(cfg).toString()
  
  #args = ['../build/dpSubclusterSphereGMM',
  args = ['../build/dpStickGMM',
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
elif dataPath in ['./rndSphereData.csv','./rndSphereDataIw.csv','./rndSphereDataIwUncertain.csv','../build/rndSphereData.csv', './rndSphereDataElipticalCovs.csv', './rndSphereDataElipticalCovs0.csv', './rndSphereDataElipticalCovs1.csv', './rndSphereDataElipticalCovs2.csv', './rndSphereDataElipticalCovs3.csv', './rndSphereDataElipticalCovs4.csv']:
  cfg['K'] = 10;
  cfg['T'] = 400;
  alpha = np.ones(cfg['K']) *1. #*100.; 
  if cfg['base'] == 'NiwSphereUnifNoise':
    alpha[cfg['K']-1] = 1.0;
  nu = D+1. #+N/10.
  Delta = nu* (1.*np.pi)/180.0 * np.eye(D-1)
  params = np.array([nu])
  params = np.r_[params,Delta.ravel()]

  if cfg['base'] in ['DpNiw']:
    Delta = nu* (0.1*np.pi)/180. * np.eye(D)
    kappa = 1.0 
    thetaa = np.zeros(D)
    params = np.array([nu,kappa])
    params = np.r_[params,thetaa.ravel(),Delta.ravel()] 

  outName,_ = os.path.splitext(dataPath)
  outName += '_'+Config2String(cfg).toString()
  
  #args = ['../build/dpSubclusterSphereGMM',
  args = ['../build/dpStickGMM',
    '-N {}'.format(N),
    '-D {}'.format(D),
    '-K {}'.format(cfg['K']),
    '-T {}'.format(cfg['T']),
    '--alpha '+' '.join([str(a) for a in alpha]),
    #'--base NiwSphereUnifNoise',
    '--base '+cfg['base'],
    '-i {}'.format(dataPath),
    '-o {}'.format(outName+'.lbl'),
    '--params '+' '.join([str(p) for p in params])]

elif dataPath in ['/data/vision/fisher/data1/wikipediaWordVectors/vectors_wiki.normalized_OnlyVect_colVects.csv', 
    '/data/vision/fisher/data1/wikipediaWordVectors/vectors_wiki.20_onlyVects_colVects.csv',
    '/data/vision/fisher/data1/wikipediaWordVectors/vectors_wiki.20_prune100_onlyVects_colVects.csv']:
   
  cfg['K'] = 20; #97;
  cfg['T'] = 660;
  cfg['alpha'] = 1.;
  alpha = np.ones(cfg['K'])*cfg['alpha']
  nu = D+1.0 #+ 100.
  Delta = nu*(1.*np.pi)/180.0
  params = np.array([nu, Delta])

  outName,_ = os.path.splitext(dataPath)
  outName += '_'+Config2String(cfg).toString()
  
  #args = ['../build/dpSubclusterSphereGMM',
  args = ['../build/dpStickGMM',
    '-N {}'.format(N),
    '-D {}'.format(D),
    '-K {}'.format(cfg['K']),
    '-T {}'.format(cfg['T']),
    '--alpha '+' '.join([str(a) for a in alpha]),
    '--base '+cfg['base'],
    '-i {}'.format(dataPath),
    '-o {}'.format(outName+'.lbl'),
    '--brief '+' '.join([str(p) for p in params])]

elif dataPath in ['./sphericalAssociatedPress.csv']:
  cfg['K'] = 8;
  cfg['T'] = 100;
  alpha = np.ones(cfg['K'])*100; 
  nu = D+4.0+10
  Delta = nu*(12.*np.pi)/180.0
  params = np.array([nu, Delta])
  outName,_ = os.path.splitext(dataPath)
  outName += '_'+Config2String(cfg).toString()
  args = ['../build/dpStickGMM',
    '-N {}'.format(N),
    '-D {}'.format(D),
    '-K {}'.format(cfg['K']),
    '-T {}'.format(cfg['T']),
    '--alpha '+' '.join([str(a) for a in alpha]),
    '--base '+cfg['base'],
    '-i {}'.format(dataPath),
    '-o {}'.format(outName+'.lbl'),
    '--brief '+' '.join([str(p) for p in params])]
elif dataPath in ['./rndNonSpherical.csv']:
  cfg['K'] = 1;
  cfg['T'] = 100;
  alpha = np.ones(cfg['K'])*1 #*100.; 
  nu = D+1.0 #+N/10.

  kappa = 1.0
  thetaa = np.zeros(D)
  Delta = nu * np.eye(D)
  params = np.array([nu,kappa])
  params = np.r_[params,thetaa.ravel(),Delta.ravel()]
  outName,_ = os.path.splitext(dataPath)
  outName += '_'+Config2String(cfg).toString()
  args = ['../build/dpStickGMM',
    '-N {}'.format(N),
    '-D {}'.format(D),
    '-K {}'.format(cfg['K']),
    '-T {}'.format(cfg['T']),
    '--alpha '+' '.join([str(a) for a in alpha]),
    #'--base NiwSphereUnifNoise',
    '--base '+cfg['base'],
    '-i {}'.format(dataPath),
    '-o {}'.format(outName+'.lbl'),
    '--params '+' '.join([str(p) for p in params])]

else:
  raise NotImplementedError

if reRun:
  print ' '.join(args)
  print ' --------------------- '
  time.sleep(3);
  err = subp.call(' '.join(args),shell=True)
  if err: 
    print 'error when executing'
    raw_input()

z = np.loadtxt(outName+'.lbl',dtype=int,delimiter=' ')
logLike = np.loadtxt(outName+'.lbl_jointLikelihood.csv',delimiter=' ')
T = z.shape[0]
for t in range(T):
  print t,logLike[t],np.bincount(z[t,:])
for t in range(T):
  print t,logLike[t],np.bincount(z[t,:]/2)


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
  if counts.sum() > 30000:
    mode = 'point'
  else:
    mode = 'sphere'
  K = 2*((int(np.max(z[-1,:]))+1)/2)
  M = Sphere(3)
  M.plot(figm,1.)
  if cfg['base'] == 'NiwSphereUnifNoise':
    for k in range(K-1):
      if (z[-1,:]==k).sum() >0:
        mlab.points3d(x[0,z[-1,:]==k],x[1,z[-1,:]==k],x[2,z[-1,:]==k],color=colorScheme('label')[k%7],figure=figm,scale_factor=0.03,mode=mode)
        mu = karcherMeanSphere_propper(np.array([0.,0.,1.]),x[:,z[-1,:]==k]) 
#        print mu
#        mlab.plot3d([0.0,mu[0]],[0.0,mu[1]],[0.0,mu[2]],figure=figm,color=colorScheme('label')[k%7])
        xT = M.LogTo2D(mu,x[:,z[-1,:]==k])
        M.plotCov(figm,np.cov(xT),mu)
    k=K-1
    if (z[-1,:]==k).sum() >0:
      mlab.points3d(x[0,z[-1,:]==k],x[1,z[-1,:]==k],x[2,z[-1,:]==k],color=(0.4,0.4,0.4),figure=figm,scale_factor=0.05,mode=mode)
  elif cfg['base'] in ['DpNiw','DpNiwSphere','DpNiwSphereFull','DpNiwTangent']:
    t =0
    means = np.loadtxt(outName+'.lbl_means.csv')
    covs = np.loadtxt(outName+'.lbl_covs.csv')
    dd = D-1
    while True:
      print "-------- @t={} --------------".format(t)
      mlab.clf(figm)
      M.plot(figm,1.) #,linewidth=1)
      K = 2*((int(np.max(z[t,:]))+1)/2)
      zAll = np.floor(z[t,:]/2)
      print np.bincount(z[t,:])
      for k in range(K/2):
        if (zAll==k).sum() >1:
          mask = max(1,int((zAll==k).sum()/300))
#          mlab.points3d(x[0,z[t,:]==2*k],x[1,z[t,:]==2*k],x[2,z[t,:]==2*k],color=colorScheme('label')[k%7],figure=figm,scale_factor=0.03,mode='sphere')
#          mlab.points3d(x[0,z[t,:]==2*k+1],x[1,z[t,:]==2*k+1],x[2,z[t,:]==2*k+1],color=colorScheme('label')[k%7],figure=figm,scale_factor=0.03,mode='sphere')
          mlab.points3d(x[0,z[t,:]==2*k],x[1,z[t,:]==2*k],x[2,z[t,:]==2*k],
              color=colorScheme('label')[k%7],figure=figm,scale_factor=0.05,mode='2dcross',mask_points=mask)
          mlab.points3d(x[0,z[t,:]==2*k+1],x[1,z[t,:]==2*k+1],x[2,z[t,:]==2*k+1],
              color=colorScheme('label')[k%7],figure=figm,scale_factor=0.03,mode='2dcircle',mask_points=mask)
          if plotSubClusters:
#          mu = karcherMeanSphere_propper(np.array([0.,0.,1.]),x[:,zAll==k]) 
#          print mu
            mu = means[t*3:(t+1)*3,3*k] 
            print mu
            mlab.plot3d([0.0,mu[0]],[0.0,mu[1]],[0.0,mu[2]],figure=figm,color=colorScheme('label')[k%7])
#          if (z[t,:]==2*k).sum() >1:
#            xT = M.LogTo2D(mu,x[:,z[t,:]==2*k])
#            M.plotCov(figm,np.cov(xT),mu)
            M.plotCov(figm,covs[t*dd:(t+1)*dd,3*k*dd:(3*k+1)*dd],mu)
            mu = means[t*3:(t+1)*3,3*k+1] 
            print mu
            mlab.plot3d([0.0,mu[0]],[0.0,mu[1]],[0.0,mu[2]],figure=figm,color=colorScheme('label')[k%7])
#          if (z[t,:]==2*k+1).sum() >1:
#            xT = M.LogTo2D(mu,x[:,z[t,:]==2*k+1])
#            M.plotCov(figm,np.cov(xT),mu)
            M.plotCov(figm,covs[t*dd:(t+1)*dd,(3*k+1)*dd:(3*k+2)*dd],mu)
          # plot upper clusters
          mu = means[t*3:(t+1)*3,3*k+2] 
          print mu
          mlab.plot3d([0.0,mu[0]],[0.0,mu[1]],[0.0,mu[2]],figure=figm,color=colorScheme('label')[k%7])
          M.plotCov(figm,covs[t*dd:(t+1)*dd,(3*k+2)*dd:(3*k+3)*dd],mu)
#          xT = M.LogTo2D(mu,x[:,zAll==k])
#          M.plotCov(figm,np.cov(xT),mu)
      key = raw_input("press j,l, <space>:")
      if key =='j':
        t-=1
      elif key == 'l' or key == '':
        t+=1
      elif key == 'E':
        t=T-1
      elif key == 'q':
        break
      elif key == ' ':
        mlab.show(stop=True)
      else:
        try:
          tt = int(key)
        except ValueError:
          tt = t 
        t = tt
      t=t%T

  else:
    for k in range(K):
      if (z[-1,:]==k).sum() >1:
        mlab.points3d(x[0,z[-1,:]==k],x[1,z[-1,:]==k],x[2,z[-1,:]==k],color=colorScheme('label')[k%7],figure=figm,scale_factor=0.05,mode=mode)
        mu = karcherMeanSphere_propper(np.array([0.,0.,1.]),x[:,z[-1,:]==k]) 
        print mu
        mlab.plot3d([0.0,mu[0]],[0.0,mu[1]],[0.0,mu[2]],figure=figm,color=colorScheme('label')[k%7])
        xT = M.LogTo2D(mu,x[:,z[-1,:]==k])
        M.plotCov(figm,np.cov(xT),mu)
  mlab.show(stop=True)
#else:


def rotationFromEye(R):
  # Log map from SOn to son
#  print np.trace(R), 0.5*(np.trace(R) -1.)
#  theta = np.arccos(min(1.0,max(-1.0,0.5*(np.trace(R) -1.))))
#  print theta
#  lnR = 0.5*theta/np.sin(theta) * (R-R.T)
#x = Log_{p}(q) = p logm(  p^{-1}q)
  W = logm(R)
  W = 0.5*(W-W.T) # antisymmetrize it
  print W
  print np.sqrt((np.triu(W)**2).sum())
  print np.sqrt(0.5*(W**2).sum())
  theta = (np.sqrt((np.triu(W)**2).sum()).real)
  if theta >= np.pi:
    theta -= 2*np.pi
  return theta

#counts = np.bincount(z[-1,:])
#K = int(np.max(z[-1,:]))
#zAll = np.floor(z[-1,:])
K = 2*((int(np.max(z[-1,:]))+1)/2)
zAll = np.floor(z[-1,:]/2)
M = Sphere(D)
north = np.zeros(D)
north[0] = 1.0;
eigs = np.zeros((K/2,D-1))
cors = np.zeros((K/2,D-1))
thetas = np.zeros(K/2)
means = np.loadtxt(outName+'.lbl_means.csv')
covs = np.loadtxt(outName+'.lbl_covs.csv')
dd = D-1
t = T-1
for k in range(K/2):
  if (zAll==k).sum()>1:  
    # upper cluster mean and covariance
    mu = means[t*D:(t+1)*D,3*k+2] #karcherMeanSphere_propper(north,x[:,zAll==k]) 
    cov = covs[t*dd:(t+1)*dd,(3*k+2)*dd:(3*k+3)*dd] 
#    xT = M.LogTo2D(mu,x[:,zAll==k])
#    xT = np.nan_to_num(xT)
#    print xT
#    if xT.shape[1] < 2:
#      continue
#    cov = np.cov(xT)
    e,V = eig(cov)
    eigs[k,:] = e.real
    print '---------- k={} ------------- '.format(k)
    print 'Eig(cov): \n{}'.format(e.real)
    print '# points {}'.format((zAll==k).sum())
    print 'cond. number = eig_max/eig_min: {}'.format(np.max(e)/np.min(e))
    print 'eig_max - eig_min: {}'.format(np.max(e) - np.min(e))
    print 'eig_max {} eig_min: {}'.format(np.max(e), np.min(e))
    thetas[k] = rotationFromEye(V)
    print 'rotation from identity: \n{}'.format(thetas[k]*180./np.pi)
    std = np.sqrt(e.real)
    if np.max(std)/np.min(std) > 1.:
      print 'large condition number {} -> saving data'.format(np.max(std)/np.min(std))
      np.savetxt(outName+'largeCovForTest{}_cov.csv'.format(k),cov)
      np.savetxt(outName+'largeCovForTest{}_mu.csv'.format(k),mu)
      np.savetxt(outName+'largeCovForTest{}_z.csv'.format(k),z[-1,:],fmt='%d')

#    print 'mean: \n{}'.format(mu)
    if K<50:
      fig = plt.figure()
      plt.subplot(1,2,1)
      plt.stem(np.arange(D-1),e.real)
      plt.title('eig_max/eig_min: \n{}'.format(np.max(e.real)/np.min(e.real)))
      plt.xlabel('dimension')
      plt.ylabel('eigenvalue')
      plt.xlim([-0.2,D-1.8])
      plt.subplot(1,2,2)
      plt.imshow(cov,interpolation='nearest')
      plt.colorbar()
      plt.title('k={}; #points={}'.format(k,(zAll==k).sum()))
      fig.show()

print np.bincount(z[-1,:])

# convert to std and angles
stds = np.sqrt(eigs)
stds *= 180./np.pi

fig = plt.figure(figsize=(10, 7), dpi=80, facecolor='w', edgecolor='k')
plt.hist(stds.ravel(),bins=100)
plt.title('hist. over all stds in all clusters')
plt.xlabel('std [deg]')
plt.savefig(outName+'_histAllStd.png',figure=fig)                                
fig.show()

fig = plt.figure(figsize=(10, 7), dpi=80, facecolor='w', edgecolor='k')
plt.hist(np.max(stds,axis=1).ravel(),bins=100)
plt.title('hist. over all max(std) per cluster ')
plt.xlabel('max(std) per cluster [deg]')
plt.savefig(outName+'_histMaxStd.png',figure=fig)                                
fig.show()

fig = plt.figure(figsize=(10, 7), dpi=80, facecolor='w', edgecolor='k')
plt.hist(np.min(stds,axis=1).ravel(),bins=100)
plt.title('hist. over all min(std) per cluster')
plt.xlabel('min(std) per cluster [deg]')
plt.savefig(outName+'_histMinStd.png',figure=fig)                                
fig.show()

fig = plt.figure(figsize=(10, 7), dpi=80, facecolor='w', edgecolor='k')
plt.hist((np.max(stds,axis=1) - np.min(stds,axis=1)).ravel(),bins=100)
plt.title('hist. over all max(stds)-min(stds) per cluster')
plt.xlabel('max(stds)-min(stds) per cluster [deg]')
plt.savefig(outName+'_histMaxMinusMinStd.png',figure=fig)                                
fig.show()

fig = plt.figure(figsize=(10, 7), dpi=80, facecolor='w', edgecolor='k')
plt.hist((np.max(stds,axis=1) / np.min(stds,axis=1)).ravel(),bins=30)
plt.title('hist. over max(std)/min(std) per cluster')
plt.xlabel('max(std)/min(std) [deg]')
plt.savefig(outName+'_histMaxOverMinStd.png',figure=fig)                                
fig.show()

fig = plt.figure(figsize=(10, 7), dpi=80, facecolor='w', edgecolor='k')
plt.plot(np.min(stds,axis=1).ravel(), np.max(stds,axis=1).ravel(),'x')
plt.xlabel('min(std) per cluster [deg]')
plt.ylabel('max(std) per cluster [deg]')
plt.savefig(outName+'_minMaxStd.png',figure=fig)                                
fig.show()


print thetas
print np.max(stds,axis=1).ravel()/np.min(stds,axis=1).ravel()
print np.max(stds,axis=1).ravel()
print np.min(stds,axis=1).ravel()

fig = plt.figure(figsize=(10, 7), dpi=80, facecolor='w', edgecolor='k')
plt.plot(thetas*180./np.pi,np.max(stds,axis=1).ravel()/np.min(stds,axis=1).ravel(),'x')
plt.xlabel('angle of cov. per cluster [deg]')
plt.ylabel('max(std)/min(std) per cluster [deg]')
plt.savefig(outName+'_angleMaxOverMinStd.png',figure=fig)                                
fig.show()

fig = plt.figure(figsize=(10, 7), dpi=80, facecolor='w', edgecolor='k')
plt.plot(thetas*180./np.pi,np.max(stds,axis=1).ravel(),'x')
plt.xlabel('angle of cov. per cluster [deg]')
plt.ylabel('max(std) per cluster [deg]')
plt.savefig(outName+'_angleMaxStd.png',figure=fig)                                
fig.show()


pdb.set_trace()

raw_input()

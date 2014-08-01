import numpy as np
import subprocess as subp

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mayavi.mlab as mlab

from matplotlib.patches import Ellipse

import pdb, re
import os.path
import time

from js.utils.plot.colors import colorScheme
from js.utils.config import Config2String

from vpCluster.manifold.karcherMean import karcherMeanSphere_propper
from vpCluster.manifold.sphere import Sphere

import matplotlib as mpl                                                        
mpl.rc('font',size=25) 
mpl.rc('lines',linewidth=4.)


def mutualInfo(z,zGt):
  ''' assumes same number of clusters in gt and inferred labels '''
  N = float(z.size)
  Kgt = int(np.max(zGt)+1)
  K = int(np.max(z)+1)
  print Kgt, K
  mi = 0.0
  for j in range(K):
    for k in range(Kgt):
      Njk = np.logical_and(z==j,zGt==k).sum()
      Nj = (z==j).sum()
      Nk = (zGt==k).sum()
      if Njk > 0:
#        print '{} {} {} {} {} -> += {}'.format(N, Njk,Nj,Nk, N*Njk/(Nj*Nk), Njk/N * np.log(N*Njk/(Nj*Nk)))
        mi += Njk/N * np.log(N*Njk/(Nj*Nk))
  return mi
def entropy(z):
  ''' assumes same number of clusters in gt and inferred labels '''
  N = float(z.size)
  K = int(np.max(z)+1)
  H = 0.0
  for k in range(K):
    Nk = (z==k).sum()
    if Nk > 0:
#        print '{} {} {} {} {} -> += {}'.format(N, Njk,Nj,Nk, N*Njk/(Nj*Nk), Njk/N * np.log(N*Njk/(Nj*Nk)))
      H -= Nk/N * np.log(Nk/N)
  return H

dataPath = './rndSphereData.csv';
dataPath = './rndSphereDataIw.csv';
dataPath = './rndSphereDataElipticalCovs.csv';
dataPath = './rndSphereDataElipticalCovs1.csv';
dataPath = './rndSphereDataElipticalCovs2.csv'; # 10k datapoints with 30 classes
dataPath = './rndSphereDataElipticalCovs3.csv'; # 10k datapoints with 30 classes

# for final eval
dataPath = './rndSphereDataElipticalCovs4.csv'; # 10k datapoints with 30 classes less spread
dataPath = './rndSphereDataIwUncertain.csv';

if os.path.isfile(re.sub('.csv','_gt.lbl',dataPath)):
  zGt = np.loadtxt(re.sub('.csv','_gt.lbl',dataPath),dtype=int,delimiter=' ')
  Kgt = np.max(zGt)+1
else:
  print "groundtruth not found"

cfg = dict()
bases = ['NiwSphereUnifNoise','spkm','spkmKarcher','NiwSphere','kmeans'];
bases = ['spkm','spkmKarcher','kmeans','NiwSphere'];
bases = ['DpNiw'];
bases = ['spkm'];
bases = ['DpNiwSphereFull','DpNiw','DpNiwTangent','DpNiwSphere'];
bases = ['DpNiwSphereFull'];
bases = ['NiwSphere','DpNiwSphereFull','DpNiwSphere']
bases = ['spkm','spkmKarcher','kmeans','NiwSphere']
bases = ['kmeans','NiwSphere']
bases = ['spkm','spkmKarcher','kmeans','NiwSphere','DpNiwSphereFull','DpNiwSphere']
bases = ['spkm','spkmKarcher','kmeans']
bases = ['spkm','spkmKarcher','kmeans','DpNiwSphereFull']
bases = ['DpNiwSphereFull']
bases = ['spkm','spkmKarcher','kmeans','NiwSphere']
bases = ['DpNiwSphereFull','DpNiw','DpNiwTangent','DpNiwSphere'];
bases = ['spkm'];
bases = ['DpNiwSphereFull'];
bases = ['spkm','kmeans','NiwSphere','DpNiwSphereFull']
bases = ['DpNiwSphereFull','DpNiw','DpNiwTangent','DpNiwSphere'];
bases = ['spkm'];
bases = ['spkm','kmeans','NiwSphere']
bases = ['spkm','kmeans','NiwSphere','DpNiw']
bases = ['DpNiwSphereFull'];
bases = ['spkm','kmeans','NiwSphere','DpNiw','DpNiwSphereFull']
bases = ['spkm','kmeans','DpNiw','DpNiwSphereFull']
bases = ['spkm_15','spkm_30','spkm_45','kmeans_15','kmeans_30','kmeans_45','DpNiw','DpNiwSphereFull']
bases = ['spkm_30','kmeans_30','DpNiw_1','DpNiw_10','DpNiwSphereFull_1','DpNiwSphereFull_10']
bases = ['DpNiwSphereFull_1','DpNiw_1'];
bases = ['spkm_27','kmeans_27']
bases = ['spkm_30','kmeans_30','DpNiw_1','DpNiw_10','DpNiwSphereFull_1','DpNiwSphereFull_10']

x=np.loadtxt(dataPath,delimiter=' ')
N = x.shape[1]
D = x.shape[0]

K = 30 # 15 30 45
reRun = False

cfg['K'] = K;
cfg['T'] = 400; #200

nmis = np.zeros((len(bases),cfg['T']))
vMeasures = np.zeros((len(bases),cfg['T']))
mis = np.zeros((len(bases),cfg['T']))
Ns = np.zeros((len(bases),cfg['T']))

for i,base in enumerate(bases):
  cfg['base']=base
  if cfg['base'] in ["DpNiwSphere","DpNiwSphereFull",'DpNiw','DpNiwTangent']:
    info = cfg['base'].split('_')
    cfg['base'] = info[0]
    cfg['K'] = int(info[1])
#    cfg['K'] = 1 #10
    alpha = np.ones(cfg['K']) *1.; 
  else:
    # get K from base string
    print cfg['base']
    info = cfg['base'].split('_')
    cfg['base'] = info[0]
    cfg['K'] = int(info[1])
    alpha = np.ones(cfg['K']) * 10.;  
  if cfg['base'] == 'NiwSphereUnifNoise':
    alpha[cfg['K']-1] = 1.0;
  nu = D+1. #+N/10.
  Delta = nu* (1.*np.pi)/180. * np.eye(D-1)
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
    '--seed {}'.format(int(time.time()*100000) - 100000*int(time.time())),
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

  if reRun:
    print ' '.join(args)
    print ' --------------------- '
    err = subp.call(' '.join(args),shell=True)
    if err:
      print 'error when executing'
      raw_input()

  z = np.loadtxt(outName+'.lbl',dtype=int,delimiter=' ')
  print z.shape
  if cfg['base'] in ["DpNiwSphere","DpNiwSphereFull","DpNiw",'DpNiwTangent']:
    z = np.floor(z/2)  
  # compute MI and entropies - if not already computed and stored 
  MI = np.zeros(cfg['T'])
  Hz = np.zeros(cfg['T'])
  Hgt = np.zeros(cfg['T'])
  if not reRun and os.path.exists('./'+outName+'_MI.csv'):
    MI = np.loadtxt(outName+'_MI.csv')
    Hgt = np.loadtxt(outName+'_Hgt.csv')
    Hz = np.loadtxt(outName+'_Hz.csv')
    Ns[i,:] = np.loadtxt(outName+'_Ns.csv')
  else:
    for t in range(cfg['T']):
      MI[t] = mutualInfo(z[t,:],zGt)
      Hz[t] = entropy(z[t,:])
      Hgt[t] = entropy(zGt)
      Ns[i,t] = int(np.max(z[t,:])+1)
    np.savetxt(outName+'_MI.csv',MI);
    np.savetxt(outName+'_Hgt.csv',Hgt);
    np.savetxt(outName+'_Hz.csv',Hz);
    np.savetxt(outName+'_Ns.csv',Ns[i,:]);

  for t in range(cfg['T']):
    nmis[i,t] = MI[t] / np.sqrt(Hz[t]*Hgt[t])
#    nmis[i,t] = MI*2. / (Hz+Hgt)
    mis[i,t] = MI[t]
    vMeasures[i,t] = 2.* MI[t] / (Hz[t]+Hgt[t])
    print nmis[i,t], 2.*MI[t], Hz[t], Hgt[t]

baseMap={'spkm':'spkm','kmeans':'k-means','NiwSphere':'DirSNIW', \
  'DpNiw':'DP-GMM','DpNiwSphere':'DpSNIW opt','DpNiwSphereFull':'DP-SGMM'}

#cl = cm.gnuplot2(np.arange(len(bases)))
cl = cm.hsv(np.arange(255))
cl = cm.brg(np.arange(255))
cl = cm.gist_rainbow(np.arange(255))
cl = cm.gnuplot2(np.arange(255))
cl = cm.gnuplot(np.arange(255))
cl = cm.spectral(np.arange(255))
print cl
I = len(bases) +1

print nmis.shape
fig = plt.figure(figsize=(14, 9), dpi=80, facecolor='w', edgecolor='k')
for i,base in enumerate(bases):
  if not re.search('_',base) is None:
    info = base.split('_')
    base = info[0]
    Ki = int(info[1])
    plt.plot(np.arange(cfg['T']),nmis[i,:],label=baseMap[base]+' (k={})'.format(Ki),c=cl[(i+1)*255/I])
  else:
    plt.plot(np.arange(cfg['T']),nmis[i,:],label=baseMap[base],c=cl[(i+1)*255/I])
  print i*255/len(bases)
plt.xlabel('iterations')
plt.ylabel('NMI')
plt.ylim([0,1])
plt.legend(loc='lower right')
#plt.title(dataPath)
plt.savefig(outName+'_NMI.png',figure=fig)

fig = plt.figure(figsize=(14, 9), dpi=80, facecolor='w', edgecolor='k')
for i,base in enumerate(bases):
  if not re.search('_',base) is None:
    info = base.split('_')
    base = info[0]
    Ki = int(info[1])
    plt.plot(np.arange(cfg['T']),vMeasures[i,:],label=baseMap[base]+' (k={})'.format(Ki),c=cl[(i+1)*255/I])
  else:
    plt.plot(np.arange(cfg['T']),vMeasures[i,:],label=baseMap[base],c=cl[(i+1)*255/I])
plt.xlabel('iterations')
plt.ylabel('V-Measure')
plt.ylim([0,1])
plt.legend(loc='lower right')
#plt.title(dataPath)
plt.savefig(outName+'_VMeasure.png',figure=fig)

fig = plt.figure(figsize=(14, 9), dpi=80, facecolor='w', edgecolor='k')
for i,base in enumerate(bases):
  if not re.search('_',base) is None:
    info = base.split('_')
    base = info[0]
    Ki = int(info[1])
    plt.plot(np.arange(cfg['T']),mis[i,:],label=baseMap[base]+' (k={})'.format(Ki),c=cl[(i+1)*255/I])
  else:
    plt.plot(np.arange(cfg['T']),mis[i,:],label=baseMap[base],c=cl[(i+1)*255/I])
plt.xlabel('iterations')
plt.ylabel('MI')
plt.legend(loc='lower right')
#plt.title(dataPath)
plt.savefig(outName+'_MI.png',figure=fig)

fig = plt.figure(figsize=(14, 9), dpi=80, facecolor='w', edgecolor='k')
plt.plot(np.arange(1),np.ones(1)*K)
#plt.plot(np.arange(1),np.ones(1)*K)
#plt.plot(np.arange(1),np.ones(1)*K)
#plt.plot(np.arange(cfg['T']),np.ones(cfg['T'])*15,label='spkm, k-means (k=15)')
plt.plot(np.arange(cfg['T']),np.ones(cfg['T'])*30,label='spkm, k-means',c=cl[(1+1)*255/I])
#plt.plot(np.arange(cfg['T']),np.ones(cfg['T'])*45,label='spkm, k-means (k=45)')
for i,base in enumerate(bases):
  if not re.search('_',base) is None:
    info = base.split('_')
    base = info[0]
    if base in ['DpNiw','DpNiwSphere','DpNiwSphereFull']:
      Ki = int(info[1])
      plt.plot(np.arange(cfg['T']),Ns[i,:],label=baseMap[base]+' (k={})'.format(Ki),c=cl[(i+1)*255/I])
plt.plot(np.arange(cfg['T']),np.ones(cfg['T'])*Kgt,'--',label='ground-truth',c=cl[0])
plt.xlabel('iterations')
plt.ylabel('# clusters')
plt.ylim([0,np.max(Ns)+1])
plt.legend(loc='lower right')
#plt.title(dataPath)
plt.savefig(outName+'_nClusters.png',figure=fig)

plt.show()

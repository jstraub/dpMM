# Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>
# Licensed under the MIT license. See the license file LICENSE.
import numpy as np
from numpy import sin,cos
from scipy.linalg import  norm, det, qr, expm, eig
import pickle

import mayavi.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from manifold.sphere import *

import time
import ipdb
from rgbd.rgbdframe import RgbdFrame
#from vp.vanishingpoint import *
from js.utils.plot.colors import colorScheme
import js.utils.plot.pyplot as pltut                                            


def sphericalHist(n):
  xedges,yedges = np.linspace(-np.pi,np.pi,36),np.linspace(-np.pi,np.pi,36)
#  beta = np.nan_to_num(np.arcsin(n[:,1]))
#  alpha = np.arcsin(n[:,0]/np.cos(beta)) 
##  alpha = (np.arcsin(n[:,0]/np.cos(beta)) + np.arccos(n[:,2]/np.cos(beta)))*0.5
#  alpha = np.nan_to_num(alpha)
#  H, xedges, yedges = np.histogram2d(alpha, beta, bins=(xedges,yedges),
#      normed=True)
  
  xmid = (xedges[:-1] + xedges[1::])*0.5
  ymid = (yedges[:-1] + yedges[1::])*0.5
  halpha, hbeta = np.meshgrid(xmid,ymid)
  halpha, hbeta = halpha.ravel(), hbeta.ravel()
  h3d = np.c_[np.sin(halpha)*np.cos(hbeta), np.sin(hbeta), 
      np.cos(halpha)*np.cos(hbeta)]
  H = np.zeros((35,35))
#  ipdb.set_trace()
  for i,ni in enumerate(n):
    ang = np.arccos(h3d.dot(ni))
    imin = np.argmin(ang)
    H[imin/35,imin%35] += 1.
  H /= float(n.shape[0])

  fig = plt.figure()
  plt.imshow(H, interpolation='nearest', origin='low',
                      extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
  plt.colorbar()
  fig.show()
  
#  ipdb.set_trace()
  print np.sqrt(np.einsum('...i,...i', h3d,h3d))[:,np.newaxis]

  H3 = np.c_[H.ravel(),H.ravel(),H.ravel()]

  h3d[H3>0.] *= (1. + H3[H3>0.])
  h3d[H3==0.] *= 0.0

  print np.sqrt(np.einsum('...i,...i', h3d,h3d))[:,np.newaxis]
  
  S = Sphere(3)
  figm = mlab.figure(bgcolor=(1,1,1))
  mlab.points3d(h3d[:,0],h3d[:,1],h3d[:,2],H3[:,0],scale_factor=10.)
  S.plotFanzy(figm,1.0,linewidth=1)
#  mlab.show(stop=True)

  return H3, h3d

mode = ['CG','robust']

name = 'hall_2'
name = '2013-10-01.19:25:00'
name = 'MIT_hallway_1'
name = 'corridor_3_1'
name = 'corridor_3_2'
name = '3boxes_0'
name = 'lines_3'
name = 'lines_4'
name = 'lines_1'
name = 'lines_2'
name = 'corridor_3_0'

dataPath='../data/'
f=525.0
rgbd = RgbdFrame(f,f)
rgbd.load(dataPath+name)
#### gradients in image
rgb_dx, rgb_dy, rgb_E, rgb_phi = rgbd.getRgbGrad()

g = np.c_[rgb_dx.ravel(),rgb_dy.ravel()]
print g.shape
g[:,0] /= rgb_E.ravel() # 1.0/np.sqrt((g**2).sum(axis=1))
g[:,1] /= rgb_E.ravel() # 1.0/np.sqrt((g**2).sum(axis=1))
print g

eThresh = 250000

fig = plt.figure()
plt.imshow(rgbd.rgb)
fig.show()
fig = plt.figure()
plt.subplot(2,2,1)
plt.hist(rgb_phi.ravel()[rgb_E.ravel() > eThresh],100)
plt.title('hist over angles in gradient image')
plt.subplot(2,2,2)
plt.hist(rgb_E.ravel(),100)
plt.title('hist over intensity in gradient image')
plt.subplot(2,2,3)
phiMasked = rgb_phi
phiMasked[rgb_E < eThresh] = 0
plt.imshow(phiMasked)
plt.title('angle in gradient image')
plt.colorbar()
fig.show()

fig = plt.figure()
plt.subplot(2,2,1)
plt.imshow(rgb_dx)
plt.title('dx in gradient image')
plt.colorbar()
plt.subplot(2,2,2)
plt.imshow(rgb_dy)
plt.title('dy in gradient image')
plt.colorbar()
plt.subplot(2,2,3)
plt.imshow(rgb_E)
plt.title('intensity in gradient image')
plt.colorbar()
plt.subplot(2,2,4)
plt.imshow(rgb_phi)
plt.title('angle in gradient image')
plt.colorbar()
fig.show()


### compute interpretation planes [Barnard 1983]
x,y = np.meshgrid(np.arange(640)-320.5,np.arange(480)-240.5)
ps = np.c_[x.ravel(),y.ravel(),np.ones(640*480)*f]
#pDeltas = ps + np.c_[rgb_dx.ravel(),rgb_dy.ravel(),np.ones(640*480)*f]
# rotate by 90 degree to get from gradients to line directions
pDeltas = ps + np.c_[np.cos(rgb_phi.ravel()+np.pi*0.0)*1.,
    -np.sin(rgb_phi.ravel()+np.pi*0.0)*1.,np.zeros_like(rgb_phi.ravel())]
mask = rgb_E.ravel() > eThresh
ps = ps[mask,:]
pDeltas = pDeltas[mask,:]

#fig=plt.figure()
#plt.quiver(ps[::0,0],ps[::10,1],(pDeltas-ps)[::10,0],(pDeltas-ps)[::10,1])
#plt.show()

#ps /= np.sqrt(np.einsum('...i,...i', ps, ps))[:,np.newaxis]
#pDeltas /= np.sqrt(np.einsum('...i,...i', pDeltas, pDeltas))[:,np.newaxis]
phi = np.cross(ps,pDeltas)
phi /= np.sqrt(np.einsum('...i,...i', phi, phi))[:,np.newaxis]
print np.allclose(np.sqrt(np.einsum('...i,...i', phi, phi)),np.ones(phi.shape[0]))

print phi.shape[0]*(phi.shape[0]-1)*0.5
S = (1./phi.shape[0])*phi.T.dot(phi)
print S
e,V = eig(S)
print V
print e

reRun = False
N,D = phi.shape
cfg = {'K':5,'T':600, 'alpha': [1.0], 'base':'DpNiwSphereFull', 'seed':1123212, 
    'dataPath':'../data/','delta':1.}
nu = 10.
Delta = nu* (cfg['delta']*np.pi)/180.0 * np.eye(D-1)
params = np.array([nu])
params = np.r_[params,Delta.ravel()]

np.savetxt(cfg['dataPath']+name+'_normals.csv',phi.T)
#np.savetxt(cfg['dataPath']+name+'_normalsRows.csv',phi)

args = ['~/workspace/research/dpMM/build/dpStickGMM',
  '-N {}'.format(N),
  '-D {}'.format(D),
  '-K {}'.format(cfg['K']),
  '-T {}'.format(cfg['T']),
  '--alpha '+' '.join([str(a) for a in cfg['alpha']]),
  '--base '+cfg['base'],
  '--seed {}'.format(cfg['seed']),
  '-i {}'.format(cfg['dataPath']+name+'_normals.csv'),
  '-o {}'.format(cfg['dataPath']+name+'.lbl'),
  '--params '+' '.join([str(p) for p in params])]
if reRun:
  print ' '.join(args)
  print ' --------------------- '
  time.sleep(1);
  import subprocess as subp
  err = subp.call(' '.join(args),shell=True)
  if err: 
    print 'error when executing'
    raw_input()

print name+'.lbl'
z = np.loadtxt(cfg['dataPath']+name+'.lbl',delimiter=' ').astype(int)[-1]
K = z.max()+1
logLike = np.loadtxt(cfg['dataPath']+name+'.lbl_jointLikelihood.csv',delimiter=' ')
mfColor = colorScheme('label')

S = Sphere(3)
figm = mlab.figure(bgcolor=(1,1,1))
S.plotFanzy(figm,1.0,linewidth=1)
Ns = [(z==k).sum() for k in range(K)]
print  Ns
for k in range(K): #np.argsort(Ns)[-1:-5:-1]:
  if (z==k).sum() > 0:
    print k, (z==k).sum()
    mlab.points3d(phi[z==k,0],phi[z==k,1],phi[z==k,2],color=mfColor[k%7],mode='point')
plotCosy(figm,V)

lines = np.zeros((480,640,3))
#inds = np.arange(480*640)[mask]
indsc,indsr = np.meshgrid(np.arange(640),np.arange(480))
indsc,indsr = indsc.ravel()[mask], indsr.ravel()[mask]
for k in range(K): #np.argsort(Ns)[-1:-5:-1]:
  if (z==k).sum() > 0:
    print mask.shape
    print (z==k).shape
#    ipdb.set_trace()
#    mask_k = np.zeros_like(mask,dtype=bool)
#    mask_k[mask][z==k] = True
    lines[indsr[z==k],indsc[z==k],:] = mfColor[k%7]
lines = pltut.linear(lines,scale_min=0,scale_max=1.0)
fig = plt.figure()
plt.imshow(lines,aspect='equal')

vp = V[0:2,0]*f/V[2,0] + np.array([320,240])
vp1 = V[0:2,1]*f/V[2,1] + np.array([320,240])
vp2 = V[0:2,2]*f/V[2,2] + np.array([320,240])
print vp,vp1,vp2
plt.plot([vp[0]],[vp[1]],'or')
plt.plot([vp1[0]],[vp1[1]],'or')
#plt.plot([vp2[0]],[vp2[1]],'or')

V = -V

vp = V[0:2,0]*f/V[2,0] + np.array([320,240])
vp1 = V[0:2,1]*f/V[2,1] + np.array([320,240])
vp2 = V[0:2,2]*f/V[2,2] + np.array([320,240])
print vp,vp1,vp2
plt.plot([vp[0]],[vp[1]],'or')
plt.plot([vp1[0]],[vp1[1]],'or')
fig.show()
plt.show()

mlab.show(stop=True)
#ipdb.set_trace()


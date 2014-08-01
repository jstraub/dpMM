import numpy as np
import subprocess as subp

import matplotlib.pyplot as plt

import pdb, re

dataPath = None;
dataPath = './allSignalsV5.csv' #data from temperature sensor of cellphone
dataPath = './sphereData.csv' 

if dataPath is None:
  N = 4000
  D = 1
  x=np.random.standard_normal(size=[D,N/2])
  #pdb.set_trace()
  x=np.c_[x,10*np.ones([D,N/2])+np.random.standard_normal([D,N/2])]
  np.savetxt('./data.csv',x,delimiter=' ')
  N = x.shape[1]
  dataPath = './data.csv'
else:
  x=np.loadtxt(dataPath,delimiter=' ')
  N = x.shape[1]
  D = x.shape[0]

if D < 3:
  fig = plt.figure()
  if D==2:
    plt.plot(x[0,:],x[1,:],'.')
  elif D==1:
    plt.plot(x[0,:],np.ones(x.shape[1]),'.')
  fig.show()

T=300
alpha = 0.01
nu = D+3.0
kappa = D+3.0
Delta = np.eye(D)*1.0
theta = np.ones(D)*0.0#np.mean(x, axis=1) #np.ones(D)*0.0
params = np.array([nu,kappa])
params = np.r_[params,theta.ravel(),Delta.ravel()]

#args = ['../build/dpStickGMM',
args = ['../build/dpSubclusterGMM',
  '-N {}'.format(N),
  '-D {}'.format(D),
  '-T {}'.format(T),
  '-a {}'.format(alpha),
  '-b NIW',
  '-i {}'.format(dataPath),
  '-o {}'.format(re.sub('csv','lbl',dataPath)),
  '-p '+' '.join([str(p) for p in params])]

print ' '.join(args)
print ' --------------------- '
subp.call(' '.join(args),shell=True)

z = np.loadtxt(re.sub('csv','lbl',dataPath),dtype=int,delimiter=' ')

if D <3:
  fig = plt.figure()
  plt.subplot(2,1,1)
  plt.plot(np.max(z,axis=1)+1)
  plt.title('number of clusters')
  
  def covEllipse(x,sigmaMult=3.0):
    from scipy.linalg import eig
    from matplotlib.patches import Ellipse
    mean = np.mean(x,axis=1)
    cov = np.cov(x) 
    e,V=eig(cov)
    imax = np.argmax(e.real)
    imin = np.argmin(e.real)
    ang = np.arctan2(V[1,imax],V[0,imax])
    print 'mean={}; max axis={}; min axis={}, angle={}'.format(mean, 
        sigmaMult*np.sqrt(e[imax]).real, sigmaMult*np.sqrt(e[imin]).real,
        ang/np.pi*180.0)
    return Ellipse(xy=(mean[0],mean[1]),width=sigmaMult*np.sqrt(e[imax]).real,
        height=sigmaMult*np.sqrt(e[imin]).real,angle=ang/np.pi*180.0)
  
  ax = plt.subplot(2,1,2)
  counts = np.bincount(z[-1,:])
  K = float(np.max(z[-1,:]))
  print K
  for k,c in enumerate(counts):
    if D==2 and (z[-1,:]==k).sum()>0:
      plt.plot(x[0,z[-1,:]==k],x[1,z[-1,:]==k],'.',color=(k/K,1.-k/K,0))
      if (z[-1,:]==k).sum() > 1:
        print 'plotting ellipse'
        plt.gca().add_patch(covEllipse(x[:,z[-1,:]==k]))
    elif D==1 and (z[-1,:]==k).sum()>0:
      if (z[-1,:]==k).sum() > 1:
        plt.plot(x[0,z[-1,:]==k],np.ones((z[-1,:]==k).sum()),'.',color=(k/K,1.-k/K,0))
        plt.plot(np.mean(x[0,z[-1,:]==k]),np.ones(1)*2.,'x',color=(k/K,1.-k/K,0))
else:
  fig = plt.figure()
  plt.imshow(np.cov(x),interpolation='nearest')
  plt.colorbar()
  plt.title('full')
  fig.show()

  counts = np.bincount(z[-1,:])
  K = float(np.max(z[-1,:]))
  for k,c in enumerate(counts):
    if (z[-1,:]==k).sum()>0:
      print '---------- k={} ------------- '.format(k)
      print 'cov: \n{}'.format(np.cov(x[:,z[-1,:]==k]))
      print 'mean: \n{}'.format(np.mean(x[:,z[-1,:]==k],axis=1))

      fig = plt.figure()
      plt.imshow(np.cov(x[:,z[-1,:]==k]),interpolation='nearest')
      plt.colorbar()
      plt.title('k={}'.format(k))
      fig.show()
  
plt.show()

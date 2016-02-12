# Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>
# Licensed under the MIT license. See the license file LICENSE.
import numpy as np
import subprocess as subp

import matplotlib.pyplot as plt

import re, argparse, os

parser = argparse.ArgumentParser(description = 'DpMM sampler')
parser.add_argument('-i','--input', default="../data/rndSphereDataIwUncertain.csv", help='input csv file N rows and D columns.')
parser.add_argument('-b','--base', default="DpNiwSphereFull",help="Basemeasure to use. Supported right now are: DpNiw (DP-GMM) and DpNiwSphereFull (DP-TGMM).")
parser.add_argument('-T', type=int, default=100, help='number of sampler iterations')
args = parser.parse_args()

dataPath = args.input
base = args.base

x=np.loadtxt(dataPath,delimiter=' ')
N = x.shape[1]
D = x.shape[0]

T=args.T        # number of sampler iterations
alpha = 1.0     # alpha concentration parameter of the DP
K = 1           # initial number of clusters
nu = D+1.01 


if base == 'DpNiw':
  kappa = D+3.0
  theta = np.ones(D)*0.0#np.mean(x, axis=1) #np.ones(D)*0.0
  Delta = nu* (1.*np.pi)/180. * np.eye(D)
  params = np.array([nu,kappa])
  params = np.r_[params,theta.ravel(),Delta.ravel()]
elif base == 'DpNiwSphereFull':
  Delta = nu* (1.*np.pi)/180. * np.eye(D-1)
  params = np.array([nu])
  params = np.r_[params,Delta.ravel()]

args = [os.path.dirname(os.path.realpath(__file__))+'/../build/dpmmSampler',
  '-N {}'.format(N),
  '-D {}'.format(D),
  '-K {}'.format(K),
  '-T {}'.format(T),
  '-a {}'.format(alpha),
  '--base '+base,
  '-i {}'.format(dataPath),
  '-o {}'.format(re.sub('csv','lbl',dataPath)),
  '-p '+' '.join([str(p) for p in params])]

print ' '.join(args)
print ' --------------------- '
ret = subp.call(' '.join(args),shell=True)
print ret
if not ret:
  print "Please make sure the sampler is compiled and located at ./build/dpmmSampler"

z = np.loadtxt(re.sub('csv','lbl',dataPath),dtype=int,delimiter=' ')
z = z/2 # to get from subcluster indices to cluster indices
Ks = np.array([np.unique(z[t,:]).size for t in range(T)])

logLikes = np.loadtxt(re.sub('csv','lbl',dataPath)+'_jointLikelihood.csv',delimiter=' ')
print logLikes.shape
print logLikes

fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(np.arange(T), Ks)
plt.ylabel("number of clusters K")
plt.xlabel("iteration")
plt.subplot(2,1,2)
plt.plot(np.arange(T+1), logLikes)
plt.ylabel("log likelihood")
plt.xlabel("iteration")
plt.show()

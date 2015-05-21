import time
import numpy as np 
import subprocess as subp

rootPath = "/scratch/amps/synthdata/"
rootPath = "/home/jstraub/workspace/research/dpMMshared/data/synthdata/"
dataPath = "train-001.log"
heldoutPath = "test-001.log"
outName = dataPath+"_out"

if True:
  x = np.loadtxt(rootPath+dataPath)
  ho = np.loadtxt(rootPath+heldoutPath)
  import matplotlib.pyplot as plt
  plt.figure()
  plt.plot(x[:,0],x[:,1],'.')
  plt.plot(ho[:,0],ho[:,1],'r.')
  plt.show()

reRun = True

cfg =dict()
cfg['K'] = 1;
cfg['T'] = 300
cfg['J'] = 1
cfg['N'] = 10000
cfg['Nho'] = 1000
cfg['D'] = D = 2
cfg['base'] = "DpNiw"

alpha = [1.]*cfg['K']

nu = 4.
Delta = nu * np.eye(D)
kappa = 0.000001
thetaa = np.zeros(D)
params = np.array([nu,kappa])
params = np.r_[params,thetaa.ravel(),Delta.ravel()] 

args = ['../build/dpmmSampler',
  '--seed {}'.format(int(time.time()*100000) - 100000*int(time.time())),
#  '-s', # print silhouette
  '-N {}'.format(cfg['N']),
  '--Nho {}'.format(cfg['Nho']),
  '-D {}'.format(cfg['D']),
  '-K {}'.format(cfg['K']),
  '-T {}'.format(cfg['T']),
  '--alpha '+' '.join([str(a) for a in alpha]),
  #'--base NiwSphereUnifNoise',
  '--base '+cfg['base'],
  '-i {}'.format(rootPath+dataPath),
  '--heldout {}'.format(rootPath+heldoutPath),
  '-o {}'.format(outName+'.lbl'),
  '--params '+' '.join(['{:.9f}'.format(p) for p in params])]
if reRun:
  print ' '.join(args)
  print ' --------------------- '
  time.sleep(3)
  err = subp.call(' '.join(args),shell=True)

hoLogLike = np.loadtxt(outName+'.lbl'+"_hoLogLike.csv")
print hoLogLike[:,0].T
#import matplotlib.pyplot as plt
#plt.figure()



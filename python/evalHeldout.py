import time
import numpy as np 
import subprocess as subp

rootPath = "/scratch/amps/synthdata/"
dataPath = "train-000.log"
heldoutPath = "test-000.log"
outName = dataPath+"_out"

reRun = True

cfg =dict()
cfg['K'] = 1;
cfg['T'] = 100
cfg['J'] = 1
cfg['N'] = 10000
cfg['Nho'] = 1000
cfg['D'] = D = 2
cfg['base'] = "DpNiw"

alpha = [1.]*cfg['K']

nu = 4.
Delta = nu * np.eye(D)
kappa = 0.1
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
  '--params '+' '.join([str(p) for p in params])]
if reRun:
  print ' '.join(args)
  print ' --------------------- '
  time.sleep(3)
  err = subp.call(' '.join(args),shell=True)


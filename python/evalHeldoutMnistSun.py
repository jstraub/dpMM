import time
import numpy as np 
import subprocess as subp

# render and plot without X
#import matplotlib as mpl
#mpl.use('Agg')

reRun = False
reRun = True
plotHo = True
plotHo = False

cfg =dict()
cfg['K'] = 1;
cfg['T'] = 2000
cfg['J'] = 1
cfg['D'] = D = 20
cfg['base'] = "DpNiw"

alpha = [5.]*cfg['K']

nu = D+2.
Delta = nu * np.eye(D) * 1e6
kappa = 0.001
thetaa = np.zeros(D)
params = np.array([nu,kappa])
params = np.r_[params,thetaa.ravel(),Delta.ravel()] 

if plotHo:
  fig = plt.figure(1)
#  hoLogLikes = np.zeros
rootPath = "/scratch/amps/mnistsun/"
rootPath = "/data/vision/scratch/fisher/jstraub/amps/mnistsun/"
for dataSet in ['mnist20','sun20','lf3']:
  dataPath = dataSet+"_train.txt"
  x = np.loadtxt(rootPath+dataPath)
  heldoutPath = dataSet+"_test.txt"
  ho = np.loadtxt(rootPath+heldoutPath)
  cfg['N'] = x.shape[0]
  cfg['Nho'] = ho.shape[0]

  outName = rootPath+'results/'+dataPath+"_out"
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
    '--rowData',
    '-i {}'.format(rootPath+dataPath),
    '--heldout {}'.format(rootPath+heldoutPath),
    '-o {}'.format(outName+'.lbl'),
    '--params '+' '.join(['{:.9f}'.format(p) for p in params])]
  if reRun:
    print ' '.join(args)
    print ' --------------------- '
    time.sleep(2)
    err = subp.call(' '.join(args),shell=True)
  if plotHo:
    hoLogLike = np.loadtxt(outName+'.lbl'+"_hoLogLike.csv")
    #print hoLogLike[:,0].T
    plt.figure(1)
    plt.plot(np.cumsum(hoLogLike[:,1])/1000.,hoLogLike[:,0])
if plotHo:
  plt.savefig(rootPath+'results/'+dataSet+'hologLikes.png',figure=fig)


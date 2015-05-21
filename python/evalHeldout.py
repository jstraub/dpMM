import time
import numpy as np 
import subprocess as subp

if False:
  x = np.loadtxt(rootPath+dataPath)
  ho = np.loadtxt(rootPath+heldoutPath)
  import matplotlib.pyplot as plt
  plt.figure()
  plt.plot(x[:,0],x[:,1],'.')
  plt.plot(ho[:,0],ho[:,1],'r.')
  plt.show()

reRun = True
reRun = False
plotZ = False
plotHo = False

cfg =dict()
cfg['K'] = 40;
cfg['T'] = 1000
cfg['J'] = 1
cfg['N'] = 10000
cfg['Nho'] = 1000
cfg['D'] = D = 2
cfg['base'] = "DpNiw"

genSynth = False
if genSynth:
#  import ipdb 
  x = np.c_[np.random.randn(2,100)*0.2,np.random.randn(2,100)*0.2+4.].T
  x = x[np.random.permutation(200),:]
#  ipdb.set_trace()
  np.savetxt("./synth.csv",x[:180,:],fmt='%.18e', delimiter=' ')
  np.savetxt("./synthHo.csv",x[180::,:],fmt='%.18e', delimiter=' ')
  rootPath = './'
  dataPath = 'synth.csv'
  heldoutPath = 'synthHo.csv'
  cfg['K'] = 1;
  cfg['T'] = 100
  cfg['J'] = 1
  cfg['N'] = 180
  cfg['Nho'] = 20

alpha = [1.]*cfg['K']

nu = 4.
Delta = nu * np.eye(D)
kappa = 0.001
thetaa = np.zeros(D)
params = np.array([nu,kappa])
params = np.r_[params,thetaa.ravel(),Delta.ravel()] 

if plotZ or plotHo:
  import matplotlib.pyplot as plt
if plotHo:
  fig = plt.figure()
for i in range(30):
  if not genSynth:
    rootPath = "/scratch/amps/"
    dataPath = "train-{:03d}.log".format(i)
    heldoutPath = "test-{:03d}.log".format(i)
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
    time.sleep(3)
    err = subp.call(' '.join(args),shell=True)

  if plotHo:
    hoLogLike = np.loadtxt(outName+'.lbl'+"_hoLogLike.csv")
    print hoLogLike[:,0].T
    plt.plot(np.arange(hoLogLike.shape[0]),hoLogLike[:,0])
  if plotZ:
    x = np.loadtxt(rootPath+dataPath)
    z = np.loadtxt(outName+'.lbl')
    K = np.unique(z).max()+1
    fig2 = plt.figure()
    for k in range(K):
      if np.count_nonzero(z==k) >0:
        plt.plot(x[z==k,0],x[z==k,1])
    plt.savefig(outName+'.lbl.png',figure=fig2)
plt.savefig('test.png',figure=fig)


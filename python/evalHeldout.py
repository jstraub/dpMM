import time
import numpy as np 
import subprocess as subp

# render and plot without X
#import matplotlib as mpl
#mpl.use('Agg')

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
plotZ = True 
plotZ = False 
plotHo = False
plotHo = True

cfg =dict()
cfg['K'] = 10;
cfg['T'] = 500
cfg['J'] = 1
cfg['N'] = 100000
cfg['Nho'] = 10000
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

alpha = [5.5]*cfg['K']

nu = 4.
Delta = nu * np.eye(D)
kappa = 0.001
thetaa = np.zeros(D)
params = np.array([nu,kappa])
params = np.r_[params,thetaa.ravel(),Delta.ravel()] 

if plotZ or plotHo:
  import matplotlib.pyplot as plt
if plotHo:
  fig = plt.figure(1)
#  hoLogLikes = np.zeros
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
    #print hoLogLike[:,0].T
    plt.figure(1)
    plt.plot(np.cumsum(hoLogLike[:,1])/1000.,hoLogLike[:,0])
  if plotZ:
    T = cfg['T']
    x = np.loadtxt(rootPath+dataPath)
    z = np.loadtxt(outName+'.lbl').astype(np.int)[-1,:]/2
    means = np.loadtxt(outName+'.lbl_means.csv')
    covs = np.loadtxt(outName+'.lbl_covs.csv')
    K = np.unique(z).max()+1
    print "K=",K
    fig2 = plt.figure(2)
    for k in range(K):
      if np.count_nonzero(z==k) >0:
        plt.plot(x[z==k,0],x[z==k,1],'.')
        mean = means[(T-1)*2:T*2, k*3+2];
        print mean
        plt.plot(mean[0],mean[1],'xr')
        cov = covs[(T-1)*2:T*2, (k*3+2)*D:(k*3+3)*D];
        L = np.linalg.cholesky(cov)
        print cov
        c = np.zeros((2,40))
        for  i,theta in enumerate(np.linspace(0,np.pi*2,40)):
          c[0,i] = np.cos(theta)
          c[1,i] = np.sin(theta)
        c = (3*(L.dot(c)).T + mean).T
        plt.plot(c[0,:],c[1,:],'r')
    plt.savefig(outName+'.lbl.png',dpi=300,figure=fig2)
if plotHo:
  plt.figure(1)
  plt.savefig(rootPath+'results/hologLikes.png',figure=fig)


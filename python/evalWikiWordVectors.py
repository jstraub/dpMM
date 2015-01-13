# Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>
# Licensed under the MIT license. See the license file LICENSE.

import numpy as np
from scipy.linalg import eig, logm
import subprocess as subp

import matplotlib.pyplot as plt
import mayavi.mlab as mlab
from matplotlib.patches import Ellipse

import ipdb, re, time
import os.path

from js.utils.plot.colors import colorScheme
from js.utils.config import Config2String

from vpCluster.manifold.karcherMean import karcherMeanSphere_propper
from vpCluster.manifold.sphere import Sphere

from pytagcloudLocal import create_tag_image, make_tags

import matplotlib as mpl                                                        
mpl.rc('font',size=20)                                                          
mpl.rc('lines',linewidth=3.)


x=np.loadtxt(dataPath,delimiter=' ')
N = x.shape[1]
D = x.shape[0]
norms = np.sqrt((x**2).sum(axis=0))
print norms
for d in  range(D):
  x[d,:] /= norms

dataPath = './vectors_wiki.20_prune100_onlyVects_colVects.csv'; #20D

cfg=dict()
cfg['base'] = 'DpNiwSphereFull';
cfg['K'] = 20;
cfg['T'] = 660;
cfg['alpha'] = 1.;
plotWordClouds = False
plot2D = False

outName,_ = os.path.splitext(dataPath)
outName += '_'+Config2String(cfg).toString()
print outName

#zAll = np.loadtxt(outName+'.lbl',dtype=int,delimiter=' ')
#logLike = np.loadtxt(outName+'.lbl_jointLikelihood.csv',delimiter=' ')
#T = z.shape[0]
#z = zAll[-1,:]

pathWords = '/data/vision/fisher/expres1/jstraub/results/wordVectorClustering/vectors_wiki.20_prune100_onlyWords.txt'                          
pathInds = '/data/vision/fisher/expres1/jstraub/results/wordVectorClustering/wikipediaWordVectors/vectors_wiki.20_prune100_onlyVects_colVects_alpha_1.0-K_20-base_DpNiwSphereFull-T_600.lblmlInds.csv'
pathLogLikes = '/data/vision/fisher/expres1/jstraub/results/wordVectorClustering/wikipediaWordVectors/vectors_wiki.20_prune100_onlyVects_colVects_alpha_1.0-K_20-base_DpNiwSphereFull-T_600.lblmlLogLikes.csv'

fid=open(pathWords)                                                             
words=fid.readlines()                                                           
words = [word[:-1] for word in words] 
inds = np.loadtxt(pathInds)
logLikes = np.loadtxt(pathLogLikes)

M = Sphere(D)
Ks=[0,10,11,12,13,14,15,16,17,18,19,1,20]
Ks=[71]
Ks = np.arange(71,96)
Ks = np.arange(74,75)
Ks = np.arange(0,96)
for i,k in enumerate(Ks):
  z = np.loadtxt(outName+'largeCovForTest{}_z.csv'.format(k),dtype=int)
  zFull = z.copy();
  z = np.floor(z/2)
  cov = np.loadtxt(outName+'largeCovForTest{}_cov.csv'.format(k))
  mu  = np.loadtxt(outName+'largeCovForTest{}_mu.csv'.format(k))
  inds = np.arange(N)[z==k]
  print k, (z==k).sum()
  xs_k = x[:,z==k]
  x_k = M.LogTo2D(mu,xs_k)
  subClInd = zFull[z==k]%2

#  if (z==k).sum() < 200 or (z==k).sum() > 300:
#    continue

  e,V = eig(cov)
  std = np.sqrt(e.real)
  iEigs = np.argsort(e);

  if plotWordClouds:
    vTop = np.c_[V[:,iEigs[-1]], V[:,iEigs[-2]], V[:,iEigs[-3]]]
    xProj = vTop.T.dot(x_k)*180.0/np.pi
  
    mlab.figure(bgcolor=(1,1,1))
    mlab.points3d(xProj[0,:],xProj[1,:],xProj[2,:],mode='point')
    s = 1.
    for k,ind in enumerate(inds): 
      print words[ind]
      mlab.text3d(xProj[0,k],xProj[1,k],xProj[2,k],
          words[ind],color=(0,0,0),scale=s)
    mlab.show(stop=True)

  if plot2D:
    iEigs = np.argsort(e);
    vTop = np.c_[V[:,iEigs[-1]], V[:,iEigs[-2]]]
    p = (vTop.T.dot(x_k)*180.0/np.pi).T
    
    xSort = np.argsort(p[:,0])
    ySort = np.argsort(p[:,1])
    xWords = [words[ind] for ind in inds[xSort]]
    yWords = [words[ind] for ind in inds[ySort]]

  
    fig = plt.figure()
    plt.subplot(2,1,1)
    plt.plot(p[subClInd==0,0],p[subClInd==0,1],'xr')
    plt.plot(p[subClInd==1,0],p[subClInd==1,1],'xb')
    for l,word in enumerate(xWords[0:12]):
      plt.text(p[xSort[l],0],p[xSort[l],1],word)
    for l,word in enumerate(xWords[-13:-1]):
      plt.text(p[xSort[-l-1],0],p[xSort[-l-1],1],word)
    plt.subplot(2,1,2)
    plt.plot(p[subClInd==0,0],p[subClInd==0,1],'xr')
    plt.plot(p[subClInd==1,0],p[subClInd==1,1],'xb')
    for l,word in enumerate(yWords[0:12]):
      plt.text(p[ySort[l],0],p[ySort[l],1],word)
    for l,word in enumerate(yWords[-13:-1]):
      plt.text(p[ySort[-l-1],0],p[ySort[-l-1],1],word)
    fig.show()

    fig = plt.figure()
    for l,word in enumerate(xWords[0:12]):
      plt.text(-1,l-6,word)
      plt.plot(-1,l-6,'x')
    for l,word in enumerate(xWords[-13:-1]):
      plt.text(1,l-6,word)
      plt.plot(1,l-6,'x')
    for l,word in enumerate(yWords[0:12]):
      plt.text(0,-l-1,word)
      plt.plot(0,-1-l,'x')
    for l,word in enumerate(yWords[-13:-1]):
      plt.text(0,13-l,word)
      plt.plot(0,13-l,'x')
    fig.show()

    wordcounts =[]
    for l,w in enumerate(xWords[0:50]): 
      wordcounts.append((w,(len(xWords)-l)**5))#((x_k[0,l]**2).sum())*180.0/np.pi))
    create_tag_image(make_tags(wordcounts, maxsize=60), 
        'cloud_large_xWordsBegin.png', size=(1000, 600), fontname='Cantarell')
    wordcounts =[]
    for l,w in enumerate(xWords[-1:-51]): 
      wordcounts.append((w,(len(xWords)-l)**5))#((x_k[0,l]**2).sum())*180.0/np.pi))
    create_tag_image(make_tags(wordcounts, maxsize=60), 
        'cloud_large_xWordsEnd.png', size=(1000, 600), fontname='Cantarell')
    wordcounts =[]
    for l,w in enumerate(yWords[0:50]): 
      wordcounts.append((w,(len(yWords)-l)**5)) #((x_k[1,l]**2).sum())*180.0/np.pi))
    create_tag_image(make_tags(wordcounts, maxsize=60), 
        'cloud_large_yWordsBegin.png', size=(1000, 600), fontname='Cantarell')
    wordcounts =[]
    for l,w in enumerate(yWords[-1:-51]): 
      wordcounts.append((w,(len(yWords)-l)**5)) #((x_k[1,l]**2).sum())*180.0/np.pi))
    create_tag_image(make_tags(wordcounts, maxsize=60), 
        'cloud_large_yWordsEnd.png', size=(1000, 600), fontname='Cantarell')

#  wordcounts =[]
#  for l,ind in enumerate(inds): 
#    if len(words[ind]) > 3 and not words[ind] == 'domingo':
#      wordcounts.append((words[ind].encode('iso-8859-1'),
#        ((x_k[:,l]**2).sum())*180.0/np.pi))
#
##      print words[ind]
#  tags = make_tags(wordcounts, maxsize=60)
##  for font in ["Nobile", "Old Standard TT", "Cantarell", "Reenie Beanie", "Cuprum", "Molengo", "Neucha", "Yanone Kaffeesatz", "Cardo", "Neuton", "Inconsolata", "Crimson Text", "Josefin Sans", "Droid Sans", "Lobster", "IM Fell DW Pica", "Vollkorn", "Tangerine", "Coustard", "PT Sans Regular"]:
#  font = 'Cuprum' #'Cantarell'
#  try:
#    create_tag_image(tags, outName+'cloud_large_'+font+'_{}.png'.format(k), 
#      size=(1200, 800)) #, fontname=font)
#  except:
#    pass

  iEigs = np.argsort(e)[-3:-1]
  iEigs = [np.argmax(e), np.argmin(e)]
  iEigs = [np.argsort(e)[-1], np.argsort(e)[-2]]
  print iEigs
  print np.sort(e)
  vals = np.sort( np.sqrt((x_k**2).sum(axis=0))/np.pi*180.0)
#  print vals
  iGlobal = np.argsort( vals )
  wordsGlobal = []
  for j,ind in enumerate(inds[iGlobal]):
    wordsGlobal.append(words[ind])
#    if j < 40:
#      print words[ind]+":"+"{}".format(int(np.floor(vals[j])))
#(x_k**2).sum(axis=0)(x_k**2).sum(axis=0)  print np.sort( (x_k**2).sum(axis=0) )

  for l,iEig in enumerate(iEigs):
#    iMax = np.argmax(e)
    print '------------ eigVal {} at dim {} ---------- '.format(e[iEig],iEig)
    v = V[:,iEig]
    xProj = v.dot(x_k)
    iSorted = np.argsort(xProj)
#    print inds
#    print inds[iSorted]
#    print xProj[iSorted]
    wordsSorted = []
    for j,ind in enumerate(inds[iSorted]):
      wordsSorted.append(words[ind])
  #    print xProj[iSorted[j]],words[ind]
#    print wordsSorted
    mid = np.argmin(np.abs(xProj[iSorted]))
#    print mid
    nr = min(24,len(wordsSorted)/3)
    print wordsSorted[0:nr]
#    for i,word in enumerate(wordsSorted[0:nr]):
#      print word+':{}'.format(int(nr-i)+10)
    print wordsGlobal[0:nr]
    for i,word in enumerate(wordsGlobal[0:nr]):
      print word+':{}'.format(int(nr-i)+10)
    print wordsSorted[-1:(-nr-1):-1]
#    for i,word in enumerate(wordsSorted[-1:-1-nr:-1]):
#      print word+':{}'.format(int(nr-i)+10)
#    print xProj[iSorted[-nr:-1]]

   
#  ipdb.set_trace()




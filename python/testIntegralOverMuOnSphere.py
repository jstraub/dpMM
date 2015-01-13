# Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>
# Licensed under the MIT license. See the license file LICENSE.

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from vpCluster.manifold.sphere import Sphere
from vpCluster.gibbs.distributions import Gaussian
import pdb

def zeroMeanGauss1D(x,Sigma):
  return 1./np.sqrt(2*np.pi*Sigma) * np.exp(-0.5*x**2/Sigma)

D = 2
M = Sphere(2-1)

N=1000

angles = np.linspace(-np.pi,np.pi,N)
anglesEval = np.linspace(-np.pi,np.pi,N)
#anglesEval += np.pi
#anglesEval %= 2.*np.pi
#pdb.set_trace()

print angles
pdf = np.zeros_like(anglesEval)

q = np.zeros((2,N))
pdfPlt = np.zeros_like(q)
for i in range(N):
  q[0,i] = np.cos(anglesEval[i])
  q[1,i] = np.sin(anglesEval[i])

Nsig = 1
for sig in np.linspace(0.1,0.1,Nsig):
  Sigma = (sig*np.pi)/180.
  for alpha in angles:
    mu = np.array([np.cos(alpha),np.sin(alpha)])
    Rnorth = np.array([[np.cos(-alpha),-np.sin(-alpha)],
      [np.sin(-alpha),np.cos(-alpha)]])
    for i,theta in enumerate(anglesEval):
      x = M.Log_p(mu,q[:,i][:,np.newaxis])
      xn = Rnorth.dot(x)
      assert(np.fabs(xn[0]) < 1e-5)
    #    pdf[i] += np.exp(gauss.logPdf(xn[1])) #/ (2*np.pi)
      pdf[i] += zeroMeanGauss1D(xn[1],Sigma) #/ (2*np.pi)
    #    print Rnorth.dot(x).T,np.exp(gauss.logPdf(xn[1]))
pdf /= angles.size * Nsig
for i,theta in enumerate(anglesEval):
  pdfPlt[:,i] = q[:,i]* (1.+pdf[i])
  #  pdb.set_trace()

print pdf
print pdf.sum()

print 'plotting'
plt.figure()
plt.subplot(2,1,1)
plt.plot(anglesEval,pdf,label='pdf marginalized over mu')
plt.plot(anglesEval,np.ones_like(anglesEval)/(np.pi*2.),label='uniform')
plt.xlim([np.min(anglesEval),np.max(anglesEval)])
plt.legend()

plt.subplot(2,1,2)
plt.plot(anglesEval,pdf)
plt.xlim([np.min(anglesEval),np.max(anglesEval)])

plt.figure()
plt.plot(q[0,:],q[1,:],'r-',label='sphere')
plt.plot([0,mu[0]],[0,mu[1]],'r-x',label='mean')
plt.plot(pdfPlt[0,:],pdfPlt[1,:],'b-',label='pdf on sphere')
plt.legend()

plt.show()

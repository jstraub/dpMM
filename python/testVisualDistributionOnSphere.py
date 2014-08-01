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
Sigma = (360.*np.pi)/180.


angles = np.linspace(0,2.*np.pi,N)

pdf = np.zeros_like(angles)
q = np.zeros((2,N))
pdfPlt = np.zeros_like(q)

theta = 180.*np.pi/180.
mu = np.array([np.cos(theta),np.sin(theta)])
Rnorth = np.array([[np.cos(-theta),-np.sin(-theta)],
  [np.sin(-theta),np.cos(-theta)]])

for i in range(N):
  q[0,i] = np.cos(angles[i])
  q[1,i] = np.sin(angles[i])
  x = M.Log_p(mu,q[:,i][:,np.newaxis])
  xn = Rnorth.dot(x)
  print xn
  pdf[i] += zeroMeanGauss1D(xn[1],Sigma) #/ (2*np.pi)
  pdfPlt[:,i] = q[:,i]* (1.+pdf[i])
print pdf 
plt.figure()
plt.plot(q[0,:],q[1,:],'r-',label='sphere')
plt.plot([0,mu[0]],[0,mu[1]],'r-x',label='mean')
plt.plot(pdfPlt[0,:],pdfPlt[1,:],'b-',label='pdf on sphere')
plt.legend()

plt.figure()
plt.plot(angles,pdf)
plt.xlabel('angle from mean')
plt.ylabel('Gaussian pdf')

plt.show()

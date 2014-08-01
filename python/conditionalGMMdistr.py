import numpy as np
from scipy.linalg import det
import matplotlib.pyplot as plt

def covEllipse(mean,cov,sigmaMult=3.0):
  from scipy.linalg import eig
  from matplotlib.patches import Ellipse
  e,V=eig(cov)
  imax = np.argmax(e.real)
  imin = np.argmin(e.real)
  ang = np.arctan2(V[1,imax],V[0,imax])
  return Ellipse(xy=(mean[0],mean[1]),width=sigmaMult*np.sqrt(e[imax]),height=sigmaMult*np.sqrt(e[imin]),angle=ang/np.pi*180.0)

def conditionalGauss(mu,S,a):
  # conditional moments
  S_xGy  = S[0,0] - S[0,1]**2 / S[1,1]
  mu_xGy = mu[0] + S[0,1]/S[1,1] *(a-mu[1]) 
  return mu_xGy, S_xGy

def gauss(x,mu,S):
  return 1.0/np.sqrt(2.*np.pi*S)*np.exp(-0.5*(mu-x)**2/S)


S1 = np.array([[1.0,0.5],[.5,2.0]])
mu1 = np.array([0.,0.])
a = 1.0                           # value of y we condition on

# conditional moments
mu_xGy, S_xGy  = conditionalGauss(mu1,S1,a)

x = np.linspace(-4.0,4.0,100)

plt.figure()
ax = plt.subplot(2,1,1)
ax.add_patch(covEllipse(mu1,S1))
plt.plot(x, np.ones(x.size)*a,'r')
plt.xlim([-4,4])
plt.ylim([-3,3])

plt.subplot(2,1,2)
plt.plot(x,gauss(x,mu_xGy,S_xGy))
plt.xlim([-4,4])


pi = np.array([0.3,0.7])
S2 = np.array([[1.0,-0.5],[-.5,2.0]])
mu2 = np.array([3.,0.])
mu2_xGy, S2_xGy  = conditionalGauss(mu2,S2,a)

plt.figure()
ax=plt.subplot(2,1,1)
ax.add_patch(covEllipse(mu1,S1))
ax.add_patch(covEllipse(mu2,S2))
plt.plot(x, np.ones(x.size)*a,'r')
plt.xlim([-4,4])
plt.ylim([-3,3])

plt.subplot(2,1,2)
plt.plot(x, pi[1]*gauss(x,mu2_xGy,S2_xGy) + pi[0]*gauss(x,mu_xGy,S_xGy))
plt.plot(x, gauss(x,mu_xGy+mu2_xGy,S_xGy+S2_xGy),'r')
plt.show()

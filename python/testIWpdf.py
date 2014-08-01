
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig

from vpCluster.gibbs.distributions import IW

D=3
nu = D+1.
Delta = nu* 0.1*np.eye(D)
iw = IW(Delta,nu)

N=1000
pdf=np.zeros(N)
for i,sig in enumerate(np.linspace(0.001,1.0,N)):
  Sigma = sig*np.eye(D)
  pdf[i] = iw.logPdf(Sigma)

plt.figure()
plt.plot(np.linspace(0.001,1.0,N),pdf)
plt.xlabel('sigma')
plt.ylabel('logPdf under IW')

plt.figure()
plt.plot(np.linspace(0.001,1.0,N),np.exp(pdf))
plt.xlabel('sigma')
plt.ylabel('pdf under IW')

plt.show()


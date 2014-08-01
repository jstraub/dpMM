import numpy as np
from scipy.linalg import eig
from vpCluster.manifold.sphere import Sphere
from vpCluster.manifold.karcherMean import karcherMeanSphereWeighted_propper
from vpCluster.manifold.karcherMean import karcherMeanSphere_propper

D = 3
M = Sphere(D-1)
N = 100
# generate data
q = np.random.normal(size=(D,N))*0.1 + np.ones((D,N))
q /= np.sqrt((q**2).sum(axis=0))
# around which point we linearize
p = np.zeros(D)
p[0] = 1.
p = karcherMeanSphere_propper(p,q)
print 'karcher mean'
print p
##################### (1) ######################################
# compute cov in TpS^D using eig
################################################################
x = M.Log_p(p,q)
S = x.dot(x.T)
# find eig of scatter matrix
v,U = eig(S)
# one of the eigenvalues should be 0
iNull = np.argmin(np.abs(v.real))
if np.abs(v[iNull]) > 1e-6:
  raise ValueError(v)
vLower = np.delete(v,iNull,0)
#Ulower = np.delete(U,iNull,1)
#print v
#print U
#print Ulower
#print Ulower.dot(np.diag(vLower.real).dot(Ulower.T))
# U contains all the rotation 
# -> v's are describing the variance in the tangent plane
Slower = np.diag(vLower.real)
print 'covariance using eig'
print Slower
##################### (2) ######################################
# compute cov in TpS^D using rotation of tangent points to north
################################################################
x = M.LogTo2D(p,q)
Slower = x.dot(x.T)
print 'covariance using rotation to north'
print Slower



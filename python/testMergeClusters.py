import numpy as np
import mayavi.mlab as mlab
from vpCluster.manifold.sphere import Sphere
from vpCluster.manifold.karcherMean import karcherMeanSphereWeighted_propper

###############################################################################
# merging to clusters on the sphere
###############################################################################
M = Sphere(2)
# counts of normals in the two clusters
Nk = np.array([100,100])
# means of the clusters
mus=np.c_[np.array([0.,0.,1.0]), np.array([0.0,1.0,0.0])]
print mus
print Nk
# sufficient statistics of the zero-mean Gaussians in the tangent plane
S1 = Nk[0] * np.eye(2)*(5./180.)*np.pi
S2 = Nk[1] * np.eye(2)*(1./180.)*np.pi
# find the mean of the merged cluster
p = karcherMeanSphereWeighted_propper(mus[:,0],mus,Nk)
print p
# find the sufficient statistics of the combined cluster
mus_p = M.LogTo2D(p,mus)
S = S1 + S2 + Nk[0]*np.outer(mus_p[:,0],mus_p[:,0]) \
  + Nk[1]*np.outer(mus_p[:,1],mus_p[:,1])
print S
# plot the three clusters with covariances
figm = mlab.figure(bgcolor=(1,1,1))
M.plotCov(figm,S1/Nk[0],mus[:,0])
M.plotCov(figm,S2/Nk[1],mus[:,1])
M.plotCov(figm,S/Nk.sum(),p)
M.plot(figm,1.0)
mlab.show(stop=True)



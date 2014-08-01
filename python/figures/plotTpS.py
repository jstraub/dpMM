
import mayavi.mlab as mlab
import numpy as np
from scipy.linalg import solve, inv, norm

from vpCluster.manifold.sphere import *


def plotGaussianTpS2(R):
  # plot the tangent space
  Rr,Th = np.meshgrid(np.linspace(0,0.5,100),np.linspace(-np.pi,np.pi,100))
#  X,Y = np.meshgrid(np.linspace(-0.5,0.5,100),np.linspace(-.5,.5,100))
  X,Y = Rr*np.cos(Th),Rr*np.sin(Th)
  # color according to covariance S
  S = np.eye(2)*0.03
  pts = np.c_[X.ravel(),Y.ravel()]
  C = -0.5*(pts.T*np.dot(inv(S),pts.T)).sum(axis=0) 
  C = np.exp(np.reshape(C,X.shape))
  Z = C*0.5 + 1.001

  p3 = np.c_[X.ravel(), Y.ravel(), Z.ravel()].T
  print p3.shape
  p3 = R.dot(p3)
  X = np.reshape(p3[0,:],X.shape)
  Y = np.reshape(p3[1,:],Y.shape)
  Z = np.reshape(p3[2,:],Z.shape)

  return X,Y,Z,C

def plotTangentSpace(R,rad=1.0):

  # plot the tangent space
  Rr,Th = np.meshgrid(np.linspace(0,rad,100),np.linspace(-np.pi,np.pi,100))
  X,Y = Rr*np.cos(Th),Rr*np.sin(Th)
#  X,Y = np.meshgrid(np.linspace(-1.,1.0,100),np.linspace(-1.,1.0,100))
  Z = np.ones(X.shape)
  p3 = np.c_[X.ravel(), Y.ravel(), Z.ravel()].T
  p3 = R.dot(p3)
  X = np.reshape(p3[0,:],X.shape)
  Y = np.reshape(p3[1,:],Y.shape)
  Z = np.reshape(p3[2,:],Z.shape)

  return X,Y,Z

mfColor = []
mfColor.append((232/255.0,65/255.0,32/255.0)) # red
mfColor.append((32/255.0,232/255.0,59/255.0)) # green
mfColor.append((32/255.0,182/255.0,232/255.0)) # tuerkis
mfColor.append((232/255.0,139/255.0,32/255.0)) # orange

saveFigs = False

figm = mlab.figure(bgcolor=(1.0,1.0,1.0))

o = np.array([0.05,0.05,0.05])

# plot the sphere
M = Sphere(2)
#M.plot(figm,1.0)
#M.plotFanzy(figm,1.0)
M.plot(figm,1.0)
R0 = np.eye(3)

# plot the MF
#plotMF(figm,np.eye(3))

# compute p, q and x
p = np.array([0.0,0.0,1.0])
t = np.linspace(0.0,np.pi/3.5,200)
q = np.zeros((3,t.size))
q[2,:] = np.cos(t)
q[1,:] = np.sin(t)
x = M.Log_p(p,q)

# plot the end points p,q,x
s = 0.1
mlab.points3d([p[0]],[p[1]],[p[2]],color=mfColor[1],scale_factor=s)
mlab.points3d(q[0,-1],q[1,-1],q[2,-1],color=mfColor[1],scale_factor=s)

#s = 0.3
#mlab.text3d(q[0,-1]-o[0], q[1,-1]+o[0], q[2,-1]-o[2],'q',color=(0,0,0),scale=s)
#mlab.text3d(p[0]+o[0], p[1]+o[1], p[2]+o[2],'p',color=(0,0,0),scale=s)

mlab.points3d(q[0,:],q[1,:],q[2,:],color=mfColor[0],mode='sphere',scale_factor=0.03)
#mlab.show(stop=True)

# plot the tangent space
X,Y,Z = plotTangentSpace(R0)
mlab.mesh(X,Y,Z,color=mfColor[2], opacity=0.5)

# plot the trace along manifold and in tangent space
mlab.points3d(x[0,:]+p[0],x[1,:]+p[1],x[2,:]+p[2],color=mfColor[0],mode='sphere',scale_factor=0.03)
# plot the angle
#mlab.points3d(q[0,:],q[1,:],q[2,:],color=(0.0,0.0,1.0),mode='sphere',scale_factor=0.01)

s = 0.1
mlab.points3d(x[0,-1]+p[0],x[1,-1]+p[1],x[2,-1]+p[2],color=mfColor[1],scale_factor=s)

# plot descriptive text
#s = 0.3
#mlab.text3d(x[0,-1]+p[0]+o[0], x[1,-1]+p[1]+o[1], x[2,-1]+p[2]+o[2],'x',color=(0,0,0),scale=s)
if saveFigs:
  mlab.savefig('./TpS.png',figure=figm,size=(1200,1200))

figm = mlab.figure(bgcolor=(1.0,1.0,1.0))
M.plot(figm,1.0)
X,Y,Z = plotTangentSpace(R0)
mlab.mesh(X,Y,Z,color=mfColor[2], opacity=0.5)
X,Y,Z,C = plotGaussianTpS2(R0)
mlab.mesh(X,Y,Z,scalars=C, colormap='hot', opacity=1., figure=figm)
if saveFigs:
  mlab.savefig('./TpSGaussian.png',figure=figm,size=(1200,1200))

figm = mlab.figure(bgcolor=(1.0,1.0,1.0))
s = 0.1
M.plot(figm,1.0)
theta = np.pi/4.
Rx = np.array([[1,0,0],[0,np.cos(theta),np.sin(theta)],[0,-np.sin(theta),np.cos(theta)]])
Rxn = np.array([[1,0,0],[0,np.cos(-theta),np.sin(-theta)],[0,-np.sin(-theta),np.cos(-theta)]])
X,Y,Z = plotTangentSpace(R0,1.0)
mlab.mesh(X,Y,Z,color=mfColor[2], opacity=0.5)
mlab.points3d([0],[0],[1],color=mfColor[2],scale_factor=s)
mlab.points3d([0],[theta],[1],color=mfColor[3],scale_factor=s)
mlab.points3d([0],[-theta],[1],color=mfColor[3],scale_factor=s)

X,Y,Z = plotTangentSpace(Rx,0.6)
mlab.mesh(X,Y,Z,color=mfColor[3], opacity=0.5)
mlab.points3d([Rx[0,2]],[Rx[1,2]],[Rx[2,2]],color=mfColor[3],scale_factor=s)
X,Y,Z = plotTangentSpace(Rxn,0.6)
mlab.mesh(X,Y,Z,color=mfColor[3], opacity=0.5)
mlab.points3d([Rxn[0,2]],[Rxn[1,2]],[Rxn[2,2]],color=mfColor[3],scale_factor=s)
#X,Y,Z,C = plotGaussianTpS2(R0)
#mlab.mesh(X,Y,Z,scalars=C, colormap='hot', opacity=1., figure=figm)
if saveFigs:
  mlab.savefig('./TpS2clusters.png',figure=figm,size=(1200,1200))

mlab.show()

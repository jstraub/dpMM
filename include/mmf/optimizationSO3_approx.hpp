/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <stdint.h>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/QR>
#include <unsupported/Eigen/MatrixFunctions>

// CUDA runtime
#include <cuda_runtime.h>
// Utilities and system includes
//#include <helper_functions.h>
#include <nvidia/helper_cuda.h>

//#include <mmf/defines.h>

using namespace Eigen;

template<typename T>
class OptSO3ApproxCpu
{
  public:
  OptSO3ApproxCpu(T t_max = 5.0f, T dt = 0.05f)
    : t_max_(t_max), dt_(dt), t_(0), R_(Matrix3f::Identity()) 
  { };

  virtual ~OptSO3ApproxCpu()
  { };

//  virtual T conjugateGradient(Matrix<T,3,3>& R, uint32_t maxIter=100);
  virtual T conjugateGradient(Matrix<T,3,3>& R, 
      const Matrix<T,Dynamic,Dynamic>& qKarch, 
      const Matrix<T,Dynamic,1>& Ns, 
      uint32_t maxIter=100);

  /* return a skew symmetric matrix from A */
  Matrix3f enforceSkewSymmetry(const Matrix3f &A) const
  {return 0.5*(A-A.transpose());};

  const Matrix<T,3,3>& R() const {return R_;};
  /* matrix of the 6 directions of the MF */
  Matrix<T,Dynamic,Dynamic> M() const;

protected:
  T t_max_, dt_;
  uint32_t t_; // timestep
  Matrix<T,3,3> R_; // previous rotation
  Matrix<T,3,6> qKarch_; // karcher means for all axes
  Matrix<T,1,6> Ns_; // number of normals for each axis

  virtual void conjugateGradientPostparation_impl(Matrix<T,3,3>& R);
  virtual T conjugateGradient_impl(Matrix<T,3,3>& R, T res0,
      uint32_t maxIter=100);
  /* 
   * evaluate cost function for a given assignment of npormals to axes
   */
  virtual T evalCostFunction(Matrix<T,3,3>& R);
  /* compute Jacobian */
  virtual void computeJacobian(Matrix<T,3,3>&J, Matrix<T,3,3>& R);

  /* 
   * updates G and H from rotation R and jacobian J
   */
  virtual void updateGandH(Matrix<T,3,3>& G, Matrix<T,3,3>& G_prev,
      Matrix<T,3,3>& H, const Matrix<T,3,3>& R, const Matrix<T,3,3>& J,
      const Matrix<T,3,3>& M_t_min, bool resetH);
  /* 
   * performs line search starting at R in direction of H returns min
   * of cost function and updates R, and M_t_min
   */
  virtual T linesearch(Matrix<T,3,3>& R, Matrix<T,3,3>& M_t_min, const
      Matrix<T,3,3>& H, T t_max=1.0f, T dt=0.1f);
  /* convert a Rotation matrix R to a MF representaiton of the axes */
  void Rot2M(Matrix<T,3,3>& R, T *mu);
};


/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <stdint.h>
#include <string>
#include <vector>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/QR>
#include <unsupported/Eigen/MatrixFunctions>

// CUDA runtime
#include <cuda_runtime.h>
// Utilities and system includes
//#include <mmf/defines.h>
#include <nvidia/helper_cuda.h>

#include <jsCore/clDataGpu.hpp>
#include <jsCore/timer.hpp>

using namespace Eigen;
using namespace std;

extern void robustSquaredAngleCostFctGPU(float *h_cost, float *d_cost,
    float *d_x, float* d_weights, uint32_t *d_z, float *d_mu, float sigma_sq, 
    int N);

extern void robustSquaredAngleCostFctAssignmentGPU(float *h_cost, float *d_cost,
  uint32_t *h_W, uint32_t *d_W, float *d_x, float* d_weights, 
  uint32_t *d_z, float *d_mu, float sigma_sq, int N);

extern void robustSquaredAngleCostFctJacobianGPU(float *h_J, float *d_J,
    float *d_x, float *d_weights, uint32_t *d_z, float *d_mu, float sigma_sq, 
    int N);

extern void meanInTpS2GPU(float *h_p, float *d_p, float *h_mu_karch,
    float *d_mu_karch, float *d_q, uint32_t *d_z, float* d_weights,int N);

extern void sufficientStatisticsOnTpS2GPU(float *h_p, float *d_p, float
    *h_Rnorths, float *d_Rnorths, float *d_q, uint32_t *d_z ,int N, float
    *h_SSs, float *d_SSs);

extern void loadRGBvaluesForMFaxes();

namespace mmf{

class OptSO3
{
protected:
  float *d_cost, *d_J, *d_mu_;
//  float *h_errs_;
  uint32_t *d_N_;
  float t_max_, dt_;

  jsc::ClDataGpu<float> cld_;

public:
//  float *d_errs_;
//  float *d_q_
  float *d_weights_;
  const float sigma_sq_;
//  uint32_t *d_z, *h_z;

  float dtPrep_; // time for preparation (here assignemnts)
  float dtCG_; // time for cunjugate gradient

  uint32_t t_; // timestep
  Matrix3f Rprev_; // previous rotation

  // t_max and dt define the number linesearch steps, and how fine
  // grained to search 
  OptSO3(float sigma, float t_max = 1.0f, float dt = 0.1f, 
      float *d_weights =NULL)
    : d_cost(NULL), d_J(NULL), d_mu_(NULL), d_N_(NULL),
      t_max_(t_max),dt_(dt), cld_(3,6),
//    d_q_(d_q),
    d_weights_(d_weights),
    sigma_sq_(sigma*sigma), dtPrep_(0.0f), dtCG_(0.0f),
    t_(0), Rprev_(Matrix3f::Identity())
  {init();};

  virtual ~OptSO3();

  double D_KL_axisUnif();
  float *getErrs(int N);

  // uses uncompressed normals
  virtual void updateExternalGpuNormals(float* d_q, uint32_t N, 
      uint32_t step, uint32_t offset = 0) {
    cld_.updateData(d_q,N,step,offset); };
  virtual double conjugateGradientCUDA(Matrix3f& R, uint32_t maxIter=100);
  
  /* return a skew symmetric matrix from A */
  Matrix3f enforceSkewSymmetry(const Matrix3f &A) const
  {return 0.5*(A-A.transpose());};

  float dtPrep(void) const { return dtPrep_;};
  float dtCG(void) const { return dtCG_;};

  const VectorXf& counts() {return this->cld_.counts();};
  const VectorXu& z() {return this->cld_.z();};
  const spMatrixXf& x() {return this->cld_.x();};
  uint32_t N() const {return this->cld_.N();};

protected:
  virtual float conjugateGradientPreparation_impl(Matrix3f& R, uint32_t& N);
  virtual float conjugateGradientCUDA_impl(Matrix3f& R, float res0, uint32_t N, uint32_t maxIter=100);
  virtual void conjugateGradientPostparation_impl(Matrix3f& R){;};
  /* evaluate cost function for a given assignment of npormals to axes */
  virtual float evalCostFunction(Matrix3f& R);
  /* compute Jacobian */
  virtual void computeJacobian(Matrix3f&J, Matrix3f& R, float N);
  /* recompute assignment based on rotation R and return residual as well */
  virtual float computeAssignment(Matrix3f& R, uint32_t& N);
  /* updates G and H from rotation R and jacobian J
   */
  virtual void updateGandH(Matrix3f& G, Matrix3f& G_prev, Matrix3f& H, 
      const Matrix3f& R, const Matrix3f& J, const Matrix3f& M_t_min,
      bool resetH);
  /* performs line search starting at R in direction of H
   * returns min of cost function and updates R, and M_t_min
   */
  virtual float linesearch(Matrix3f& R, Matrix3f& M_t_min, const
      Matrix3f& H, float N, float t_max=1.0f, float dt=0.1f);
  /* mainly init GPU arrays */
  virtual void init();
  /* copy rotation to device */
  void Rot2Device(Matrix3f& R);
  /* convert a Rotation matrix R to a MF representaiton of the axes */
  void Rot2M(Matrix3f& R, float *mu);
  //deprecated
  void rectifyRotation(Matrix3f& R);
};
}

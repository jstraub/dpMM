/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>                    
 * Licensed under the MIT license. See the license file LICENSE.                
 */
#pragma once

#include <Eigen/Dense>
#include <dpMM/basemeasure.hpp>
#include <dpMM/sphere.hpp>

using namespace Eigen;

template<typename T>
class UnifSphere : public BaseMeasure<T>
{
public:

  uint32_t D_; // dimensionality of the space in which the sphere lies
  Sphere<T> S_;

  UnifSphere(uint32_t D);
  ~UnifSphere();

  virtual baseMeasureType getBaseMeasureType() const {return(UNIF_SPHERE); }

  virtual BaseMeasure<T>* copy();

  virtual T logLikelihood(const Matrix<T,Dynamic,1>& x) const;
  virtual T logLikelihood(const Matrix<T,Dynamic,Dynamic>& x, uint32_t i) const 
    {return logLikelihood(x.col(i));};
  virtual void posterior(const Matrix<T,Dynamic,Dynamic>& x, 
      const VectorXu& z, uint32_t k);
  virtual T logPdfUnderPrior() const;
  void print() const {cout<<"Unif Sphere in D="<<D_<<endl;};
  virtual uint32_t getDim() const {return(D_);};
private:

};

typedef UnifSphere<double> UnifSphered;
typedef UnifSphere<float> UnifSpheref;

// ---------------------------------------------------------------------------
template<typename T>
UnifSphere<T>::UnifSphere(uint32_t D)
  : D_(D), S_(D)
{};
template<typename T>
UnifSphere<T>::~UnifSphere()
{};

template<typename T>
BaseMeasure<T>* UnifSphere<T>::copy()
{
  return new UnifSphere<T>(D_);
};

template<typename T>
T UnifSphere<T>::logLikelihood(const Matrix<T,Dynamic,1>& x) const
{
  return - S_.logSurfaceArea();
};

template<typename T>
void UnifSphere<T>::posterior(const Matrix<T,Dynamic,Dynamic>& x,
    const VectorXu& z, uint32_t k)
{
  // nothing seince we have no priors
};

template<typename T>
T UnifSphere<T>::logPdfUnderPrior() const 
{
  return 0.0;
};


/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>, Randi Cabezas <rcabezas@csail.mit.edu>                    
 * Licensed under the MIT license. See the license file LICENSE.                
 */

#pragma once

#include <Eigen/Dense>

#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/distributions/inverse_gamma.hpp>

#include <boost/random/chi_squared_distribution.hpp>
#include <boost/random/normal_distribution.hpp>

#include "distribution.hpp"
#include "vmf.hpp"

using namespace Eigen;
using std::endl;
using std::cout;
using std::vector;

template<typename T>
class vMFpriorFull : public Distribution<T>
{
public:
  vMFpriorFull(const Matrix<T,Dynamic,1>& m0, T t0, T a0, T b0);
  vMFpriorFull(const vMFpriorFull<T>& vmfPriorFull);
  ~vMFpriorFull();

  vMFpriorFull<T> posterior(const Matrix<T,Dynamic,Dynamic>& x, const
      VectorXu& z, uint32_t k); 
  vMFpriorFull<T> posterior() const;  

  void resetSufficientStatistics();
  void getSufficientStatistics(const Matrix<T,Dynamic,Dynamic> &x, 
    const VectorXu& z, uint32_t k);

  vMF<T> sample();
  vMF<T> sampleFromPosterior();

  T logPdf(const vMF<T>& vmf) const;
  T logPdfMarginalized() const; // log pdf of SS under NIW prior
//  T logPdfUnderPriorMarginalizedMerged(const NIW<T>& other) const;
  
  
  vMF<T> vmf0_; // prior on the mean
  T a0_;
  T b0_;

  // sufficient statistics
  Matrix<T,Dynamic,1> xSum_;
  T count_;

privat:
};
// -----------------------------------------------------------------------

template<typename T> 
vMFpriorFull<T>::vMFpriorFull(const Matrix<T,Dynamic,1>& m0, T t0, T a0, T
    b0)
: vmf0_(m0,t0), a0(a0_), b0(b0_), xSum_(Matrix<T,Dynamic,1>::Zero(m0.rows())), count_(0.)
{
};

template<typename T> 
vMFpriorFull<T>::vMFpriorFull(const vMFpriorFull<T>& vmfPriorFull)
: vmf0_(vmfPriorFull.vmf0_), a0_(vmfPriorFull.a0_), b0_(vmfPriorFull.b0_),
  xSum_(vmfPriorFull.xSum_), count_(vmfPriorFull.count_)
{
};


template<typename T> 
vMFpriorFull<T>::~vMFpriorFull()
{
};

template<typename T> 
vMFpriorFull<T> vMFpriorFull<T>::posterior(const 
    Matrix<T,Dynamic,Dynamic>& x, const VectorXu& z, uint32_t k)
{
  getSufficientStatistics(x,z,k);
  return posterior();
};

template<typename T>
vMFpriorFull<T> vMFpriorFull<T>::posterior() const
{
  Matrix<T,Dynamic,1> xi = t0_*m0_ +;
  const T a = a0_ + count_;
  const T b = b0_ + 
  return vMFpriorFull(xSum_, count_);
};

template<typename T>
void vMFpriorFull<T>::resetSufficientStatistics()
{
  xSum_.setZero(D_);
  count_ = 0;
};

template<typename T>
void vMFpriorFull<T>::getSufficientStatistics(const
    Matrix<T,Dynamic,Dynamic> &x, const VectorXu& z, uint32_t k)
{
  this->resetSufficientStatistics();
  // TODO: be carefull here when parallelizing since all are writing to the same 
  // location in memory
#pragma omp parallel for
  for (int32_t i=0; i<z.size(); ++i)
  {
    if(z(i) == k)
    {      
#pragma omp critical
      {
        xSum_ += x.col(i);
        count_++;
      }
    }
  }
#ifndef NDEBUG
  cout<<" -- updating ss "<<count_<<endl;
  cout<<"xSum="<<xSum_.transpose()<<endl;
  posterior().print();
#endif
};

/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>, Randi Cabezas <rcabezas@csail.mit.edu>                    
 * Licensed under the MIT license. See the license file LICENSE.
 */
 

#pragma once

#include <Eigen/Dense>

#include "basemeasure.hpp"
#include "niw.hpp"

/*
 * vmf base measure; uses monte carlo integration for p(x|hyperparams)
 * http://eprints.pascal-network.org/archive/00007206/01/iMMM.pdf
 */
template<typename T>
class vMFbase : public BaseMeasure<T>
{
public:
  vMFbase();
  vMFbase(const vMFbase<T>& vmf);
  ~vMFbase();

  virtual BaseMeasure<T>* copy();
  virtual vMFbase<T>* copyNative();

  T logLikelihood(const Matrix<T,Dynamic,1>& x) const;
  T logLikelihood(const Matrix<T,Dynamic,Dynamic>& x, uint32_t i) const 
    {return logLikelihood(x.col(i));};
  void posterior(const Matrix<T,Dynamic,Dynamic>& x, const VectorXu& z, 
    uint32_t k);
  void posterior(const vector<Matrix<T,Dynamic,Dynamic> >&x, const VectorXu& z, 
    uint32_t k);
  void sample();

  T logPdfUnderPrior() const;
  T logPdfUnderPriorMarginalized() const;
  T logPdfUnderPriorMarginalizedMerged(const shared_ptr<NiwSampled<T> >& other) const;

  virtual NiwSampled<T>* merge(const NiwSampled<T>& other);
  void fromMerge(const NiwSampled<T>& niwA, const NiwSampled<T>& niwB);

  void print() const;
  virtual uint32_t getDim() const {return(uint32_t(normal_.D_));}; 

  const Matrix<T,Dynamic,Dynamic>& scatter() const {return niw0_.scatter();};
  const Matrix<T,Dynamic,1>& mean() const {return niw0_.mean();};
  T count() const {return niw0_.count();};
//  T& count() {return niw0_.count_;};
  const Matrix<T,Dynamic,1>& getMean() const {return normal_.mu_;};

  const Matrix<T,Dynamic,Dynamic>& Sigma() const {return normal_.Sigma();};

private:
  vMF<T> vmf_;
  vMFpriorFull<T> vmfPrior_;

};

// ------------------------- impl -------------------------------------------

template<typename T>
vMFbase<T>::vMFbase()
{
};

template<typename T>
vMFbase<T>::vMFbase(const vMFbase<T>& vmf)
{
};


template<typename T>
vMFbase<T>::~vMFbase()
{};

template<typename T>
virtual BaseMeasure<T>* vMFbase<T>::copy()
{
  return new vMFbase<T>(*this);
};

template<typename T>
virtual vMFbase<T>* vMFbase<T>::copyNative()
{
  return new vMFbase<T>(*this);
};

template<typename T>
T vMFbase<T>::logLikelihood(const Matrix<T,Dynamic,1>& x) 
{
  return vmfData.logPdf(x);
};

template<typename T>
void vMFbase<T>::posterior(const Matrix<T,Dynamic,Dynamic>& x, const VectorXu& z, 
    uint32_t k)
{ 
  vmfPrior_.getSufficientStatistics(x,z,k);
  // compute posterior parameters
  const Matrix<T,Dynamic,1> xi = t0_*m0_ + vmfData_.mu_.transpose()*xSum_;
  const T t = xi.norm();
  const Matrix<T,Dynamic,1> mu = xi/t;
  const T a = a0_ + count_;
  const T b = b0_ + vmfData_.mu_.transpose()*xSum_;

  // sample new mean on the sphere
  vmfData_.mu(vmfPrior_.posterior(xSum_).sample());
  

};

template<typename T>
void vMFbase<T>::posterior(const vector<Matrix<T,Dynamic,Dynamic> >&x, const VectorXu& z, 
    uint32_t k)
{
};

template<typename T>
void vMFbase<T>::sample()
{
};

template<typename T>
T vMFbase<T>::logPdfUnderPrior() 
{
};

template<typename T>
T vMFbase<T>::logPdfUnderPriorMarginalized()
{
};

template<typename T>
T vMFbase<T>::logPdfUnderPriorMarginalizedMerged(const shared_ptr<NiwSampled<T> >& other) 
{
};


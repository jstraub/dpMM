/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>, Randi Cabezas <rcabezas@csail.mit.edu>                    
 * Licensed under the MIT license. See the license file LICENSE.
 */
 

#pragma once

#include <Eigen/Dense>

#include "basemeasure.hpp"
#include "vmfPriorFull.hpp"

/*
 * vmf base measure; uses monte carlo integration for p(x|hyperparams)
 * http://eprints.pascal-network.org/archive/00007206/01/iMMM.pdf
 */
template<typename T>
class vMFbase : public BaseMeasure<T>
{
public:
  vMFbase(const vMFpriorFull<T>& vmfPrior);
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
  virtual T logPdfUnderPriorMarginalized() const;

  virtual T logPdfUnderPriorMarginalized(const Matrix<T,Dynamic,1>& x);

//  virtual NiwSampled<T>* merge(const NiwSampled<T>& other);
//  void fromMerge(const NiwSampled<T>& niwA, const NiwSampled<T>& niwB);

  void print() const;
  virtual uint32_t getDim() const {return(uint32_t(vmf_.D_));}; 

//  const Matrix<T,Dynamic,Dynamic>& scatter() const {return niw0_.scatter();};
//  const Matrix<T,Dynamic,1>& mean() const {return niw0_.mean();};
//  T count() const {return niw0_.count();};
////  T& count() {return niw0_.count_;};
  const Matrix<T,Dynamic,1>& getMean() const {return vmf_.mu_;};
  const T tau() const {return vmf_.tau();};

  vMFpriorFull<T> vmfPrior_;
  vMF<T> vmf_;
private:

};

// ------------------------- impl -------------------------------------------

template<typename T>
vMFbase<T>::vMFbase(const vMFpriorFull<T>& vmfPrior)
  : vmfPrior_(vmfPrior), vmf_(vmfPrior_.sample())
{};

template<typename T>
vMFbase<T>::vMFbase(const vMFbase<T>& base)
  :  vmfPrior_(base.vmfPrior_), vmf_(base.vmf_) 
{};


template<typename T>
vMFbase<T>::~vMFbase()
{};

template<typename T>
BaseMeasure<T>* vMFbase<T>::copy()
{
  return new vMFbase<T>(*this);
};

template<typename T>
vMFbase<T>* vMFbase<T>::copyNative()
{
  return new vMFbase<T>(*this);
};

template<typename T>
T vMFbase<T>::logLikelihood(const Matrix<T,Dynamic,1>& x) const
{
  return vmf_.logPdf(x);
};

template<typename T>
void vMFbase<T>::posterior(const Matrix<T,Dynamic,Dynamic>& x, const VectorXu& z, 
    uint32_t k)
{ 
  vmfPrior_.getSufficientStatistics(x,z,k);
  // needs current vmf since it samples tau|mu_old and then mu|tau
  vmf_ = vmfPrior_.sampleFromPosterior(vmf_);
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
void vMFbase<T>::print() const
{
  vmf_.print();
};

template<typename T>
T vMFbase<T>::logPdfUnderPrior() const
{
  return 0.;
};

template<typename T>
T vMFbase<T>::logPdfUnderPriorMarginalized() const
{
  return 0.;
};

template<typename T>
T vMFbase<T>::logPdfUnderPriorMarginalized(const Matrix<T,Dynamic,1>& x) 
{
  // approximate the log pdf under the prior via monte carlo sampling
  T logPdfMarg = 0;
//#pragma omp parallel for reduction(+:logPdfMarg)
  for(uint32_t t=0; t<3; ++t)
  {
    vMF<T> vmf = vmfPrior_.sample();
    logPdfMarg = logPdfMarg + vmf.logPdf(x);
  }
  return logPdfMarg/3.;
};


/* Copyright (c) 2017, Julian Straub <jstraub@csail.mit.edu>, Randi Cabezas <rcabezas@csail.mit.edu>                    
 * Licensed under the MIT license. See the license file LICENSE.
 */
 

#pragma once

#include <Eigen/Dense>

#include <dpMM/basemeasure.hpp>
#include <dpMM/vmfPriorAOne.hpp>

/*
 * vmf base measure; uses closed form for marginal data density
 * (J. Straub, "Nonparamatric Directional Perception", 2017)
 */
template<typename T>
class vMFbase3D : public BaseMeasure<T>
{
public:
  vMFbase3D(const vMFprior<T>& vmfPrior);
  vMFbase3D(const vMFbase3D<T>& vmf);
  ~vMFbase3D();

  virtual BaseMeasure<T>* copy();
  virtual vMFbase3D<T>* copyNative();

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

  vMFprior<T> vmfPrior_;
  vMF<T> vmf_;
private:

};

// ------------------------- impl -------------------------------------------

template<typename T>
vMFbase3D<T>::vMFbase3D(const vMFprior<T>& vmfPrior)
  : vmfPrior_(vmfPrior), vmf_(vmfPrior_.sample())
{};

template<typename T>
vMFbase3D<T>::vMFbase3D(const vMFbase3D<T>& base)
  :  vmfPrior_(base.vmfPrior_), vmf_(base.vmf_) 
{};


template<typename T>
vMFbase3D<T>::~vMFbase3D()
{};

template<typename T>
BaseMeasure<T>* vMFbase3D<T>::copy()
{
  return new vMFbase3D<T>(*this);
};

template<typename T>
vMFbase3D<T>* vMFbase3D<T>::copyNative()
{
  return new vMFbase3D<T>(*this);
};

template<typename T>
T vMFbase3D<T>::logLikelihood(const Matrix<T,Dynamic,1>& x) const
{
//  cout<<vmf_.logPdf(x)<<endl;
  return vmf_.logPdf(x);
};

template<typename T>
void vMFbase3D<T>::posterior(const Matrix<T,Dynamic,Dynamic>& x, const VectorXu& z, 
    uint32_t k)
{ 
  vmfPrior_.getSufficientStatistics(x,z,k);
  std::cout << vmfPrior_.xSum_.transpose() << " " << vmfPrior_.count_ << std::endl;
  vmf_ = vmfPrior_.sampleFromPosterior();
};

template<typename T>
void vMFbase3D<T>::posterior(const vector<Matrix<T,Dynamic,Dynamic> >&x, const VectorXu& z, 
    uint32_t k)
{
};

template<typename T>
void vMFbase3D<T>::sample()
{
  vmf_ = vmfPrior_.sample();
};

template<typename T>
void vMFbase3D<T>::print() const
{
  vmf_.print();
};

template<typename T>
T vMFbase3D<T>::logPdfUnderPrior() const
{
  return 0.;
};

template<typename T>
T vMFbase3D<T>::logPdfUnderPriorMarginalized() const
{
  return vmfPrior_.logPdfMarginalized();
};

template<typename T>
T vMFbase3D<T>::logPdfUnderPriorMarginalized(const Matrix<T,Dynamic,1>& x) 
{
  return vmfPrior_.logMarginal(x);
};


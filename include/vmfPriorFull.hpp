/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>, Randi Cabezas <rcabezas@csail.mit.edu>                    
 * Licensed under the MIT license. See the license file LICENSE.                
 */

#pragma once

#include <Eigen/Dense>

#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/distributions/inverse_gamma.hpp>

#include <boost/random/chi_squared_distribution.hpp>
#include <boost/random/normal_distribution.hpp>

#include <boost/special_functions/bessel.hpp>

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
  vMFpriorFull(const Matrix<T,Dynamic,1>& m0, T t0, T a0, T b0, boost::mt19937
    *pRndGen);
  vMFpriorFull(const vMFpriorFull<T>& vmfPriorFull);
  ~vMFpriorFull();

  vMFpriorFull<T> posterior(const Matrix<T,Dynamic,Dynamic>& x, const
      VectorXu& z, uint32_t k); 
  vMFpriorFull<T> posterior() const;  

  void resetSufficientStatistics();
  void getSufficientStatistics(const Matrix<T,Dynamic,Dynamic> &x, 
    const VectorXu& z, uint32_t k);

  vMF<T> sample();
  vMF<T> sampleFromPosterior(const vMF<T>& vmf);

  T logPdf(const vMF<T>& vmf) const;
  T logPdfMarginalized() const; // log pdf of SS under NIW prior
//  T logPdfUnderPriorMarginalizedMerged(const NIW<T>& other) const;
  
  
  vMF<T> vmf0_; // prior on the mean
  T a0_;
  T b0_;

  // sufficient statistics
  Matrix<T,Dynamic,1> xSum_;
  T count_;

  boost::mt19937 *pRndGen_;

privat:
  boost::uniform_01<> unif_;

  T concentrationLogPdf(const T tau, const T a, const T b) const;
  T sampleConcentration(const T a, const T b) const ;
};
// -----------------------------------------------------------------------

template<typename T> 
vMFpriorFull<T>::vMFpriorFull(const Matrix<T,Dynamic,1>& m0, T t0, T a0, T
    b0, boost::mt19937 *pRndGen)
: vmf0_(m0,t0), a0(a0_), b0(b0_), xSum_(Matrix<T,Dynamic,1>::Zero(m0.rows())),
  count_(0.), pRndGen_(pRndGen)
{
};

template<typename T> 
vMFpriorFull<T>::vMFpriorFull(const vMFpriorFull<T>& vmfPriorFull)
: vmf0_(vmfPriorFull.vmf0_), a0_(vmfPriorFull.a0_), b0_(vmfPriorFull.b0_),
  xSum_(vmfPriorFull.xSum_), count_(vmfPriorFull.count_),
  pRndGen_(vmfPriorFull.pRndGen_)
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

template<typename T>
vMF<T> vMFpriorFull<T>::sample()
{
  tau = sampleConcentration(a0_,b0_);
  Matrix<T,Dynamic,1> mu = vmf0_.sample();
  return vMF<T>(mu, tau,pRndGen_);
};

template<typename T>
vMF<T> vMFpriorFull<T>::sampleFromPosterior(const vMF<T>& vmf)
{
  // posterior concentration prior parameters
  const T a = a0_ + count_;
  const T b = b0_ + vmf.mu().transpose()*xSum_;
  // sample concentration form this posterior
  const T tau = sampleConcentration(a,b);
  // posterior mean (m_N) and concentration (t_N) for vMF
  Matrix<T,Dynamic,1> m_k = vmf0_.tau()*vmf0_.mu() + tau*xSum_;
  T t_N = xi.norm();
  m_k /= t_N;
  // sample mean mu from posterior vMF
  Matrix<T,Dynamic,1> mu = vMF<T>(m_N,t_N, pRndGen_).sample();
  // return sampled posterior vMF
  return vMF<T>(mu, tau, pRndGen_);
};


template<typename T>
T vMFpriorFull<T>::concentrationLogPdf(const T tau, const T a, const T b)
{
  // modified bessel function of the first kind
  // http://www.boost.org/doc/libs/1_35_0/libs/math/doc/sf_and_dist/html/math_toolkit/special/bessel/mbessel.html
  const T D = D_;
  return a*((D/2. -1.)*log(tau) 
    - (D/2.)*log(2.*M_PI) 
    - log(cyl_bessel_i(D_/2. -1.,tau)))
    + tau*b;
};

template<typename T>
T vMFpriorFull<T>::sampleConcentration(const T a, const T b)
{
  // slice sampler for concentration paramter tau
  const T w = 0.1;  // width for expansions of search region
  T tau = 1.0;      // arbitrary starting point
  for(int32_t t=0; t<100; ++t)
  {
    const T yMax = concentrationLogPdf(tau,a,b);
    const T y = unif_(*pRndGen_)*yMax; 
    T tauMin = tau-w; 
    T tauMax = tau+w; 
    while (concentrationLogPdf(tauMin,a,b) >= y) tauMin -= w;
    while (concentrationLogPdf(tauMax,a,b) >= y) tauMax += w;
    while(42){
      T tauNew = unif_(*pRndGen_)*(tauMax-tauMin)+tauMin;
      if(concentrationLogPdf(tauNew,a,b) >= y)
      {
        tau = tauNew; break;
      }else{
        if (tauNew < tau) tauMin = tauNew; else tauMax = tauNew;
      }
    };
  }
  return tau;
};

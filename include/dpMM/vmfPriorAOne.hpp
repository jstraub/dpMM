/* Copyright (c) 2017, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */
#pragma once
#include <cmath>
#include <iostream>
#include <Eigen/Dense>

#include "vmf.hpp"

///
/// vMF prior for 3D where a=1 and 0<b<1 for the hyperparameters
/// This allows the marginal data distribution under the prior to be
/// computed in closed form.
/// 
/// Note that the posterior can have a != 0.
///
template<typename T>
class vMFprior : public Distribution<T> 
{
 public:
  vMFprior(const Eigen::Matrix<T,Eigen::Dynamic,1>& m0, T a, T b, boost::mt19937
    *pRndGen)
    : Distribution<T>(pRndGen), m0_(m0), b_(b), a_(a), pRndGen_(pRndGen)
  {}

  void resetSufficientStatistics();
  void getSufficientStatistics(const Matrix<T,Dynamic,Dynamic> &x, 
    const VectorXu& z, uint32_t k);

  /// Sample from the prior with some parameters a, b, m
  vMF<T> sample(T aN, T bN, const Eigen::Matrix<T,Eigen::Dynamic,1>& mN) {
//    std::cout << "sample from vMF prior " << a_ << " " << b_ 
//      << " " << m0_.transpose() << std::endl;
    Eigen::Matrix<T,Eigen::Dynamic,1> mu;
    T tau = 1.;
    vMF<T> vmf(mN, tau*b_, pRndGen_);
//    std::cout << "sampling from base" << std::endl;
    for (size_t it=0; it<10; ++it) {
      vmf.tau_ = tau*b_;
      mu = vmf.sample();
//      std::cout << "mu " << mu.transpose() << std::endl;
      const T dot = mu.dot(m0_); 
      tau = sampleConcentration(dot, 3, tau);
//      std::cout <<"@" << it << "tau " << tau << " mu " << mu.transpose() << std::endl;
    }
    return vMF<T>(mu, tau, pRndGen_);
  }

  /// Sample from the prior
  vMF<T> sample() {
    return sample(a_, b_, m0_);
  }

  /// Sample from the posterior
  vMF<T> sampleFromPosterior() {
    Eigen::Matrix<T, Eigen::Dynamic, 1> vartheta = b_*m0_ + xSum_;
    return sample(a_+count_, vartheta.norm() , vartheta.normalized());
  }

  vMFprior<T> posterior(const Eigen::Matrix<T,3,1>& xSum, const T count) const {
    T aN = a_+count;
    Eigen::Matrix<T,3,1> muN = xSum + b_*m0_;
    T bN = muN.norm();
    muN /= bN;
    return vMFprior<T>(muN, aN, bN);
  }
  vMFprior<T> posterior() const {
    return posterior(xSum_, count_);
  }
  vMFprior<T> posterior(const Matrix<T,Dynamic,Dynamic>& x, const
      VectorXu& z, uint32_t k) {
    getSufficientStatistics(x, z, k);
    return posterior();
  }

//  T logPdf(const vMF<T>& vmf) const;

  T logMarginal(const Eigen::Matrix<T,3,1>& x) const {
    const T bTilde = (x + b_*m0_).norm();
    const T bOverTan = b_ < 1e-9 ? 2./M_PI : b_/tan(M_PI*0.5*b_);  
    const T sinc = bTilde < 1e-9 ? 1. : sin(bTilde*M_PI)/(bTilde*M_PI);
    const T sinus = sin(bTilde*0.5*M_PI);
    return log(bOverTan*0.125) + log(1.-sinc) - 2*log(sinus);
  }
  T logPdfMarginalized() const {
    return logMarginal(xSum_);
  }
//  T logPdfUnderPriorMarginalizedMerged(const NIW<T>& other) const;

  vMF<T> MAP() {
    Eigen::Matrix<T,3,1> xSum=b_*m0_;
    float tau = MLEstimateTau<float,3>(xSum, m0_, a_);
    return vMF<T>(m0_, tau);
  }

  Eigen::Matrix<T,3,1> m0_;
  T b_;
  T a_;
  Matrix<T,3,1> xSum_;
  T count_;

  boost::mt19937 *pRndGen_;
private:
  boost::uniform_01<> unif_;

  T propToConcentrationLogPdf(const T tau, const T dot) const
  {
    if (tau < 1e-16) {
      return 0.; 
    } else {
      return a_*(log(tau) + LOG_2 - log(1.-exp(-2.*tau))) + tau*(b_*dot-a_); 
    }
  };

  T propToConcentrationLogPdfDeriv(const T tau, const T dot) const
  {
    // this is only for 3D case
    if (tau < 1e-16) {
      return b_*dot; 
    } else {
      return a_/tau - (2.*a_*exp(-2.*tau)/(1.-exp(-2.*tau))) + b_*dot -a_;
    }
  };
  T propToConcentrationLogPdfDerivDeriv(const T tau, const T dot) const
  {
    // this is only for 3D case
    if (tau < 1e-16) {
      return -a_/3.; 
    } else {
//      return -a_/(tau*tau) + (4.*a_*exp(2.*tau)/(1.-2.*exp(2.*tau)+exp(4.*tau)));
      return -a_/(tau*tau) + (4.*a_*exp(-2.*tau)/(1.-2.*exp(-2.*tau)+exp(-4.*tau)));
    }
  };

  T maximum(const T dot) {
    if (dot*b_ <= 0)
      return 0.;
    T tau = 1.;
    for (size_t it=0; it<100; ++it) {
      T f = propToConcentrationLogPdfDeriv(tau, dot);
      T df = propToConcentrationLogPdfDerivDeriv(tau, dot);
      tau -= f/df;
      if (fabs(f/df) < 1e-6)
        break;
    }
    return tau;
  };

  T intersect(const T c, const T dot, const T tau0) {
    T tau = tau0;
    for (size_t it=0; it<100; ++it) {
      T f = propToConcentrationLogPdf(tau, dot) - c;
      T df = propToConcentrationLogPdfDeriv(tau, dot);
      tau = std::max((T)0., tau-f/df);
//      std::cout << "   __ " <<it << ": " << f << " " << df 
//        << "\t f/df " << fabs(f/df) 
//        << "\t step to " << tau-f/df << ": "<< tau << std::endl;
      if (fabs(f/df) < 1e-6 || tau == 0.)
        break;
    }
    return tau;
  };

  /// slice sampler for concentration paramter tau
  T sampleConcentration(const T dot, size_t maxIt, T tau0 = 0.3)
  {
    T tauMax = maximum(dot);
    T tau = tau0;
//    std::cout << " ----- max " << tauMax << " " << tau0 
//      << " dot " << dot << std::endl;
    T tauL = 0.;
    T tauR = tauMax;
    for(size_t t=0; t<maxIt; ++t)
    {
      const T f = propToConcentrationLogPdf(tau,dot);
      const T u = log(unif_(*pRndGen_)) + f; 
      
      if (tauMax > 0.) {
        tauL = intersect(u, dot, tauMax*0.001);
        tauR = intersect(u, dot, tauMax*1.5);
      } else {
        tauL = 0.;
        tauR = intersect(u, dot, 0.5);
      }
      tau = unif_(*pRndGen_)*(tauR-tauL)+tauL;
//      std::cout << tauL << " - " << tauMax << " - " << tauR 
//        << " tau= " << tau
//        << " : u " << u << " f(tau) " << f 
//        << " f(tau^star) " << propToConcentrationLogPdf(tauMax,dot)
//        << std::endl;
    }
    return tau;
  };

};

template<typename T>
void vMFprior<T>::resetSufficientStatistics()
{
  xSum_.setZero();
  count_ = 0;
};


template<typename T>
void vMFprior<T>::getSufficientStatistics(const
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
//  cout<<"SS: "<<count_<<" "<<xSum_.transpose()<<endl;
};


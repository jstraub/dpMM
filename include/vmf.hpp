/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */

#pragma once

#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <algorithm>

#include <iostream>

#include <boost/special_functions/bessel.hpp>
#include "distribution.hpp"
#include "normal.hpp"

using namespace Eigen;
using std::cout;
using std::endl;
using std::min;

/* von-Mises-Fisher distribution
 */
template<typename T>
class vMF : public Distribution<T>
{
public:
  uint32_t  D_;

  vMF(const Matrix<T,Dynamic,1>& mu, T tau, boost::mt19937 *pRndGen);
  vMF(const vMF<T>& vmf);
  ~vMF();

  T logPdf(const Matrix<T,Dynamic,Dynamic>& x) const;

  Matrix<T,Dynamic,1> sample();

  void print() const;

  const Matrix<T,Dynamic,1>& mu() const {return mu_;};
  void mu(const Matrix<T,Dynamic,1>& mu) const {mu_ = mu;};

  T tau() const {return tau_;};
  void tau(const T tau) {tau_ = tau;};

private 
  T tau_;
  Matrix<T,Dynamic,1> mu_;
  
// Gaussian as a proposal distribution
  Normal<T> q_;
  boost::mt19937 *pRndGen_;
  boost::uniform_01<> unif_;
};

typedef vMF<double> vMFd;
typedef vMF<float> vMFf;

template<class T>
vMF<T>::vMF(const Matrix<T,Dynamic,1>& mu, T tau, boost::mt19937 *pRndGen)
  : D_(mu_.rows()), mu_(mu), tau_(tau), 
  q_(Matrix<T,Dynamic,1>::Zero(D_),Matrix<T,Dynamic,Dynamic>::Identity(D_),pRndGen),
  pRndGen_(pRndGen)
{};

template<class T>
vMF<T>::vMF(const vMF<T>& vmf)
  : D_(vmf.D_), mu_(vmf.mu()), tau_(vmf.tau()), 
  q_(Matrix<T,Dynamic,1>::Zero(D_),Matrix<T,Dynamic,Dynamic>::Identity(D_),vmf.pRndGen_),
  pRndGen_(vmf.pRndGen_)
{};

template<class T>
vMF<T>::~vMF()
{};

template<class T>
T vMF<T>::logPdf(const Matrix<T,Dynamic,Dynamic>& x)
{
  // modified bessel function of the first kind
  // http://www.boost.org/doc/libs/1_35_0/libs/math/doc/sf_and_dist/html/math_toolkit/special/bessel/mbessel.html
  // 
  const T D = D_;
  return (D/2. -1.)*log(tau_) 
    - (D/2.)*log(2.*M_PI) 
    - log(cyl_bessel_i(D_/2. -1.,tau_)) 
    + tau_*(mu_.transpose()*x).sum();
};

template<class T>
Matrix<T,Dynamic,1> vMF<T>::sample()
{
  // implemented using rejection sampling and proposals from a gaussian
  Matrix<T,Dynamic,1> x;
  while(42)
  {
    x = q_.sample(); // propose value
    x /= x.norm();   // make it lie on sphere
    // rejection sampling (in log domain)
    T u = log(unif_(*pRndGen_));
    T pdf = this->logPdf(x);
    // bound via maximum over vMF at mu
    T M = this->logPdf(mu_); // np.exp(tau)*tau/(4*np.pi*np.sinh(tau))
    // normalizer (surface area of hyper-sphere)
    T U = LOG_2+0.5*D_*LOG_PI - boost::math::lgamma(0.5*D_);
    if(u < pdf-(M+U)) break;
  };
  return x;
};


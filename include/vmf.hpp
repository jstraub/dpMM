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
  Matrix<T,Dynamic,1> mu_;

  vMF(const Matrix<T,Dynamic,1>& mu, T tau, boost::mt19937 *pRndGen);
  ~vMF();

  T logPdf(const Matrix<T,Dynamic,Dynamic>& x) const;

  Matrix<T,Dynamic,1> sample();

  void print() const;

  const Matrix<T,Dynamic,Dynamic>& tau() const {return tau_;};
  void tau(T tau) {tau_ = tau;};

private 
  T tau_;
  
// as a proposal distribution
//  normal_distribution<> gauss_;
};

typedef vMF<double> vMFd;
typedef vMF<float> vMFf;

template<class T>
vMF<T>::vMF(const Matrix<T,Dynamic,1>& mu, T tau, boost::mt19937 *pRndGen)
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
  // TODO: implement using rejection sampling and proposals from a gaussian
  assert(false);
  return mu_;
};



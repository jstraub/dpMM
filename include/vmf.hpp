/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */

#pragma once

#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <algorithm>
#include <iostream>

#include <boost/random/normal_distribution.hpp>
#include <boost/math/special_functions/bessel.hpp>

#include <distribution.hpp>
#include <sphere.hpp>

using namespace Eigen;
using std::cout;
using std::endl;
using std::min;

template<typename T> 
inline T logBesselI(T nu, T x)
{
  // for large values of x besselI \approx exp(x)/sqrt(2 PI x)
  if(x>100.)  return x - 0.5*log(2.*M_PI*x);
  return log(boost::math::cyl_bessel_i(nu,x));
};

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

  T logPdf(const Matrix<T,Dynamic,1>& x) const;

  Matrix<T,Dynamic,1> sample();

  void print() const;

  const Matrix<T,Dynamic,1>& mu() const {return mu_;};
  void mu(const Matrix<T,Dynamic,1>& mu) const {mu_ = mu;};

  T tau() const {return tau_;};
  void tau(const T tau) {tau_ = tau;};

private:
  Matrix<T,Dynamic,1> mu_;
  T tau_;
  
// Gaussian as a proposal distribution
  boost::mt19937 *pRndGen_;
  boost::uniform_01<> unif_;
  normal_distribution<> gauss_;

  Sphere<T> S_;
};

typedef vMF<double> vMFd;
typedef vMF<float> vMFf;

template<typename T>
vMF<T>::vMF(const Matrix<T,Dynamic,1>& mu, T tau, boost::mt19937 *pRndGen)
  : Distribution<T>(pRndGen), D_(mu.rows()), mu_(mu), tau_(tau),
  pRndGen_(pRndGen), S_(D_)
{};

template<typename T>
vMF<T>::vMF(const vMF<T>& vmf)
  : Distribution<T>(vmf.pRndGen_), D_(vmf.D_), mu_(vmf.mu()), tau_(vmf.tau()),
    pRndGen_(vmf.pRndGen_), S_(D_)
{};

template<typename T>
vMF<T>::~vMF()
{};

template<typename T>
T vMF<T>::logPdf(const Matrix<T,Dynamic,1>& x) const 
{
  // modified bessel function of the first kind
  // http://www.boost.org/doc/libs/1_35_0/libs/math/doc/sf_and_dist/html/math_toolkit/special/bessel/mbessel.html
  // 
  const T D = static_cast<T>(D_);
//  cout<<"vMF: bessel: D="<<D<<" "<<(D/2.-1.)<<" tau="<<tau_<<endl;
  return (D/2. -1.)*log(tau_) 
    - (D/2.)*log(2.*M_PI) 
    - logBesselI<T>(D_/2. -1.,tau_) 
    + tau_*(mu_.transpose()*x)(0);
};

template<typename T>
Matrix<T,Dynamic,1> vMF<T>::sample()
{
  // implemented using rejection sampling and proposals from a gaussian
  Matrix<T,Dynamic,1> x(D_);
  Matrix<T,Dynamic,1> xtNorth(D_-1);

  uint32_t t=0;
  while(42)
  {
    // sample from zero mean Gaussian in tangent space at north pole
    for (uint32_t d=0; d<D_-1; d++)
      xtNorth[d] = gauss_(*this->pRndGen_)*sqrt(1./tau_); //gsl_ran_gaussian(r,1);
    // rotate sample to mu and map it down to sphere
    x = S_.Exp_p_single(mu_,S_.rotate_north2p(mu_,xtNorth));
    // rejection sampling (in log domain)
    T u = log(unif_(*pRndGen_));
    T pdf_f = this->logPdf(x);
//    cout<<endl<<"xtNorth "<<xtNorth.transpose()<<" tau="<<tau_<<endl;
    T pdf_g = -0.5*(log(2.*M_PI)*D_ -log(tau_)+tau_*(xtNorth.transpose()*xtNorth)(0) );
//    cout<<endl<<"xtNorth "<<xtNorth.transpose()<<" tau="<<tau_<<" pdf_g="<<pdf_g<<" "<<tau_*(xtNorth.transpose()*xtNorth)(0)<<endl;

    // bound via maximum over vMF at mu
//    T M = this->logPdf(mu_) + 0.5*(log(2.*M_PI)*D_ - log(tau_));
//  conservative bound
    T M = this->logPdf(mu_) + log(10) + 0.5*(log(2.*M_PI)*D_ - log(tau_));
//    cout<< u<<" "<< (pdf_f-(M+pdf_g))<< " "<<pdf_f<<" "<<M<<" "<<pdf_g<<endl;
    if(u < pdf_f-(M+pdf_g)) break;

    ++t;
  };
//  cout<<"vMF<T>::sample: T="<<t<<endl;
  return x;
};

template<typename T>
void vMF<T>::print() const
{
  cout<<"mu = "<<mu_.transpose()<<" tau = "<<tau_<<endl;
};

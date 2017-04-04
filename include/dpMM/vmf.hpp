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

#include <dpMM/distribution.hpp>
#include <dpMM/sphere.hpp>

#define LOG_2 0.69314718055994529
#define LOG_PI 1.1447298858494002
#define LOG_2PI 1.8378770664093453
#define LOG_4PI 2.5310242469692907

using namespace Eigen;
using std::cout;
using std::endl;
using std::min;

template<typename T> 
inline T logBesselI(T nu, T x)
{
  // for large values of x besselI \approx exp(x)/sqrt(2 PI x)
  if(x>100.)  return x - 0.5*LOG_2PI -0.5*log(x);
  return log(boost::math::cyl_bessel_i(nu,x));

};

template<typename T> 
inline T logxOverSinhX(T x) {
  if (fabs(x) < 1e-9) 
    return 0.;
  else
    return log(x)-log(sinh(x));
}
template<typename T> 
inline T xOverSinhX(T x) {
  if (fabs(x) < 1e-9) 
    return 1.;
  else
    return x/sinh(x);
}
template<typename T> 
inline T xOverTanPiHalfX(T x) {
  if (fabs(x) < 1e-9) 
    return 2./M_PI;
  else
    return x/tan(x*M_PI*0.5);
}

template <typename T>
inline T MLEstimateTau(const Eigen::Matrix<T,3,1>& xSum, const
    Eigen::Matrix<T,3,1>& mu, T count) {
  // Need double precision to achive convergence; single is not enough.
  double tau = 1.0;
  double prevTau = 0.;
  double eps = 1e-8;
  double R = xSum.norm()/count;
  while (fabs(tau - prevTau) > eps) {
//    std::cout << "tau " << tau << " R " << R << std::endl;
    double inv_tanh_tau = 1./tanh(tau);
    double inv_tau = 1./tau;
    double f = -inv_tau + inv_tanh_tau - R;
    double df = inv_tau*inv_tau - inv_tanh_tau*inv_tanh_tau + 1.;
    prevTau = tau;
    tau -= f/df;
  }
  return tau;
};

/* von-Mises-Fisher distribution in D=3 dimensions
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

  Matrix<T,Dynamic,1> mu_;
  T tau_;
private:
  
// Gaussian as a proposal distribution
  boost::mt19937 *pRndGen_;
  boost::uniform_01<> unif_;
  normal_distribution<> gauss_;
};

typedef vMF<double> vMFd;
typedef vMF<float> vMFf;

template<typename T>
vMF<T>::vMF(const Matrix<T,Dynamic,1>& mu, T tau, boost::mt19937 *pRndGen)
  : Distribution<T>(pRndGen), D_(mu.rows()), mu_(mu), tau_(tau),
  pRndGen_(pRndGen)
{};

template<typename T>
vMF<T>::vMF(const vMF<T>& vmf)
  : Distribution<T>(vmf.pRndGen_), D_(vmf.D_), mu_(vmf.mu()), tau_(vmf.tau()),
    pRndGen_(vmf.pRndGen_)
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
  const T d = static_cast<T>(D_);
  if (tau_ < 1e-9) {
    // TODO insert general formula here (this currently works only
    // for D=3
    assert(D_==3);
    return -LOG_4PI;
  } else {
    if (D_ == 3) {
      return -LOG_2PI + log(tau_) + tau_*(mu_.dot(x)-1.) -
        log(1.-exp(-2.*tau_));
    }else {
      return (d/2. -1.)*log(tau_) - (d/2.)*LOG_2PI 
        - logBesselI<T>(d/2. -1.,tau_) + tau_*mu_.dot(x);
    }
  }
};

template<typename T>
Matrix<T,Dynamic,1> vMF<T>::sample()
{
  assert(D_==3);
  if (tau_ < 1e-10) {
//    Eigen::VectorXf x;
//    x << gauss_(*pRndGen_), gauss_(*pRndGen_), gauss_(*pRndGen_);
//    return x.normalized();
    return Eigen::Matrix<T,3,1>(gauss_(*pRndGen_), gauss_(*pRndGen_), gauss_(*pRndGen_)).normalized();
  }
  // https://www.mitsuba-renderer.org/~wenzel/files/vmf.pdf
  // sample around (0,0,1)
  Eigen::Matrix<T,2,1> v(gauss_(*pRndGen_), gauss_(*pRndGen_));
  v.normalize();
  const T u = unif_(*pRndGen_);
  const T w = 1. + log(u+(1.-u)*exp(-2.*tau_))/tau_;
  const T a = sqrtf(1.-w*w);
  Eigen::Matrix<T,3,1> x(a*v(0), a*v(1), w);

  // rotate to mu
  Eigen::Matrix<T,3,1> axis = Eigen::Matrix<T,3,1>(0,0,1).cross(Eigen::Matrix<T,3,1>(mu_(0),mu_(1),mu_(2)));
  T angle = acos(mu_[2]);

  if (fabs(angle) <1e-9) 
    return x;

  Eigen::Quaternion<T> q(cos(angle*0.5), 
      sin(angle*0.5)*axis(0)/axis.norm(),
      sin(angle*0.5)*axis(1)/axis.norm(),
      sin(angle*0.5)*axis(2)/axis.norm());
  Eigen::Matrix<T,Eigen::Dynamic,1> xx = q._transformVector(x);
  return xx;
};

template<typename T>
void vMF<T>::print() const
{
  cout<<"mu = "<<mu_.transpose()<<"\t tau = "<<tau_<<endl;
};

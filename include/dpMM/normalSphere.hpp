/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>                    
 * Licensed under the MIT license. See the license file LICENSE.                
 */
#pragma once

#include <Eigen/Dense>
#include "global.hpp"
#include "normal.hpp"
#include "iw.hpp"
#include "sphere.hpp"

#define LOG_2PI 1.8378770664093453

template<typename T>
class NormalSphere : public Distribution<T>
{
public:
  uint32_t D_; // dimension of the space

  NormalSphere(const Matrix<T,Dynamic,1>& mu, 
      const Matrix<T,Dynamic,Dynamic>& Sigma, boost::mt19937* pRndGen);
  NormalSphere(const Matrix<T,Dynamic,1>& mu, 
      const Normal<T>& normal, boost::mt19937* pRndGen);
  NormalSphere(const NormalSphere& other);
  ~NormalSphere();

  /* for any point on sphere - maps into T_muS and rotates north before logPdf */
  T logPdf(const Matrix<T,Dynamic,1>& q_i) const;
  /* assumes x_i is already in T_northS */
  T logPdfNorth(const Matrix<T,Dynamic,1>& x_i) const;
  T logPdfNorth(const Matrix<T,Dynamic,Dynamic>& scatter, 
      const Matrix<T,Dynamic,1>& mean, T count) const;
//  T logPdfNorth(const Matrix<T,Dynamic,Dynamic>& scatter, T count) const;

  Matrix<T,Dynamic,1> sample();

  const Matrix<T,Dynamic,Dynamic>& Sigma() const {return normal_.Sigma();};
  void setSigma(const Matrix<T,Dynamic,Dynamic>& Sigma)
  {return normal_.setSigma(Sigma);};
  T logDetSigma() const {return normal_.logDetSigma();};
  T logNormalizer() const {return -0.5*(normal_.logDetSigma()+D_*LOG_2PI);};

  /* mean on sphere */
  void setMean( const Matrix<T,Dynamic,1>& mu); 
  const Matrix<T,Dynamic,1>& getMean() const {return mu_;}; 

  /* mean in tangent plane */
  void setMuInTpS( const Matrix<T,Dynamic,1>& mu)
  {normal_.mu_ = mu;};

  void setNormal(const Normal<T>& normal) {normal_ = normal;};
  const Normal<T>& normal() const {return normal_;};

private:
  Normal<T> normal_; // zero-mean Gaussian in Tangent plane (dim: D-1)
  Sphere<T> S_;
  Matrix<T,Dynamic,1> mu_; // mean pointing to location in sphere
  Matrix<T,Dynamic,Dynamic> northR_; 
};

template<class T>
inline Matrix<T,Dynamic,Dynamic> sampleClustersOnSphere(
    Matrix<T,Dynamic,Dynamic>& x, uint32_t K);

template<class T>
inline Matrix<T,Dynamic,Dynamic> sampleClustersOnSphere(
    Matrix<T,Dynamic,Dynamic>& x, VectorXu& z, uint32_t K);

template<class T>
inline Matrix<T,Dynamic,Dynamic> sampleClustersOnSphere(
    const Matrix<T,Dynamic,Dynamic>& Delta, T nu,
    Matrix<T,Dynamic,Dynamic>& x, VectorXu& z, uint32_t K);

// ---------------------------------------------------------------------------
template<typename T>
NormalSphere<T>::NormalSphere(const Matrix<T,Dynamic,1>& mu,
    const Matrix<T,Dynamic,Dynamic>& Sigma, boost::mt19937* pRndGen)
  : Distribution<T>(pRndGen),  D_(mu.size()), 
    normal_(Sigma,pRndGen), S_(D_) 
{
  setMean(mu);
};

template<typename T>
NormalSphere<T>::NormalSphere(const Matrix<T,Dynamic,1>& mu, 
      const Normal<T>& normal, boost::mt19937* pRndGen)
  : Distribution<T>(pRndGen),  D_(mu.size()), 
    normal_(normal), S_(D_) 
{
  setMean(mu);
};

template<typename T>
NormalSphere<T>::NormalSphere(const NormalSphere& other)
  : Distribution<T>(other.pRndGen_), D_(other.mu_.size()), 
    normal_(other.normal_), S_(other.D_)
{
  setMean(other.mu_);
};

template<typename T>
NormalSphere<T>::~NormalSphere()
{};

template<typename T>
void NormalSphere<T>::setMean( const Matrix<T,Dynamic,1>& mu)
{
  assert(mu.rows() == D_);
  mu_ = mu;
  northR_ = S_.north_R_TpS2(mu_);
}


template<typename T>
T NormalSphere<T>::logPdf(const Matrix<T,Dynamic,1>& q_i) const
{
//  cout<<q_i.transpose()<<endl;
  Matrix<T,Dynamic,1> x_i = S_.Log_p_single(mu_,q_i);
//  cout<<x_i.transpose()<<endl;
//  cout<<northR_<<endl;

#ifndef NDEBUG
  ASSERT(fabs(x_i.transpose()*mu_)<1e-6, x_i.transpose()*mu_);

  Matrix<T,Dynamic,1> xNorth = (northR_*x_i);
  ASSERT(fabs( xNorth(D_-1)) < 1e-6, 
      xNorth.transpose() << endl
      << " northR_ "<<endl<<northR_<<endl
      << " recomputed"<<endl<<S_.north_R_TpS2(mu_)<<endl);
  return normal_.logPdf(xNorth.topRows(D_-1));
#else
  return normal_.logPdf((northR_*x_i).topRows(D_-1));
#endif
//  return normal_.logPdf( S_.Log_p_north(mu_,q_i) );
};

template<typename T>
T NormalSphere<T>::logPdfNorth(const Matrix<T,Dynamic,1>& x_i) const
{
  assert(x_i.rows() == D_-1);
#ifndef NDEBUG
  Matrix<T,Dynamic,1> x(D_);
  x.topRows(D_-1) = x_i;
  x(D_-1) = 1.0;
//  cout<<(x.transpose()*S_.north())<<endl;
  assert(fabs((x.transpose()*S_.north()).norm() -1.) < 1.e-5);
#endif
  return normal_.logPdf(x_i);
};

template<typename T>
T NormalSphere<T>::logPdfNorth(const Matrix<T,Dynamic,Dynamic>& scatter, 
    const Matrix<T,Dynamic,1>& mean, T count) const
{
  return normal_.logPdf(scatter,mean,count); 
}

//template<typename T>
//T NormalSphere<T>::logPdfNorth(const Matrix<T,Dynamic,Dynamic>& scatter, 
//    T count) const
//{
//  return normal_.logPdf(scatter,count); 
//}

template<typename T>
Matrix<T,Dynamic,1> NormalSphere<T>::sample()
{
  Matrix<T,Dynamic,1> xNorth(D_-1);
  xNorth = normal_.sample();
  // if outside radius of PI wrap around
  // TODO
  while(xNorth.norm() > PI)
  {
    cout<<"wrapping around! ---------------------------------------------"<<endl;
    xNorth -= (T(2*PI))*(xNorth/xNorth.norm());
  }
//  cout<<"xNorth = "<<xNorth.transpose()<<endl;
//  cout<<"mu = "<<mu_.transpose()<<endl;
  Matrix<T,Dynamic,1> x = S_.rotate_north2p(mu_,xNorth);
//  cout<<"x = "<<x.transpose()<<endl;
  return S_.Exp_p(mu_,S_.rotate_north2p(mu_,xNorth));
};

template<class T>
inline Matrix<T,Dynamic,Dynamic> sampleClustersOnSphere(
    const Matrix<T,Dynamic,Dynamic>& Delta, T nu,
    Matrix<T,Dynamic,Dynamic>& x, VectorXu& z, uint32_t K,
    T minAngle = static_cast<T>(6.))
{
  uint32_t N = x.cols();
  uint32_t D = x.rows();
  Sphere<T> S_(D);
  boost::mt19937 rndGen(9119);

  IW<T> iw(Delta,nu,&rndGen);
  Matrix<T,Dynamic,Dynamic> mus(D,K);
  for(uint32_t k=0; k<K; ++k)
  {
    Matrix<T,Dynamic,Dynamic> Sigma = iw.sample();
//    cout<<Sigma<<endl;
//    cout<<"nu "<<nu<<endl;
//    cout<<Delta<<endl;
    Matrix<T,Dynamic,1> mu = S_.sampleUnif(&rndGen);
    if(k>0) 
    {
      bool done = false; 
      while(!done)
      {
        mu = S_.sampleUnif(&rndGen);
        done = true;
        for(uint32_t j=0; j<k; ++j)
          done = done & (mu.transpose()*mus.col(j) < cos(minAngle*M_PI/180.0));
      }
    }
    cout<<"sampling data for k="<<k<<" around mu="<<mu.transpose()<<" Sigma:"<<endl;
    cout<<Sigma*(180.0/M_PI)*(180.0/M_PI)<<endl;
    NormalSphere<T> gauss_k(mu,Sigma,&rndGen);
    mus.col(k) = gauss_k.getMean();
    for (uint32_t i=k*(N/K); i<min(N,(k+1)*(N/K)+N%K); ++i) 
    {
//      cout<<"--"<<endl;
//      cout<<mus.col(k).transpose()<<endl;
      do{
        x.col(i) = gauss_k.sample();
      }while(fabs(x.col(i).norm()-1.0) > 1e-3); 
      if(fabs(x.col(i).norm()-1.0) > 1e-2)
        cout<<x.col(i).norm()<<endl;
      z(i) = k;
//        x.col(i) /= x.col(i).norm();
//      cout<<x.col(i).transpose()<<endl;
    }
  }
  return mus;
};


template<class T>
inline Matrix<T,Dynamic,Dynamic> sampleClustersOnSphere(
    Matrix<T,Dynamic,Dynamic>& x, VectorXu& z, uint32_t K)
{
  uint32_t N = x.cols();
  uint32_t D = x.rows();
  Sphere<T> S_(D);
  boost::mt19937 rndGen(9119);

  Matrix<T,Dynamic,Dynamic> Sigma = Matrix<T,Dynamic,Dynamic>::Identity(D-1,D-1);
  Sigma *= 0.05;
  Matrix<T,Dynamic,Dynamic> mus(D,K);
  for(uint32_t k=0; k<K; ++k)
  {
    NormalSphere<T> gauss_k(S_.sampleUnif(&rndGen),Sigma,&rndGen);
    mus.col(k) = gauss_k.getMean();
    for (uint32_t i=k*(N/K); i<min(N,(k+1)*(N/K)+N%K); ++i) 
    {
      do{
        x.col(i) = gauss_k.sample();
      }while(fabs(x.col(i).norm()-1.0) > 1e-3); 
      if(fabs(x.col(i).norm()-1.0) > 1e-2)
        cout<<x.col(i).norm()<<endl;
      z(i) = k;
//        x.col(i) /= x.col(i).norm();
//        cout<<x.col(i).transpose()<<endl;
    }
  }
  return mus;
};

template<class T>
inline Matrix<T,Dynamic,Dynamic> sampleClustersOnSphere(
    Matrix<T,Dynamic,Dynamic>& x, uint32_t K)
{
  VectorXu z(x.cols());
  return sampleClustersOnSphere<T>(x,z,K);
};

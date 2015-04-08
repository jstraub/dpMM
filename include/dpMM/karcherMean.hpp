/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */
 

#pragma once

#include <iostream>
#include <dpMM/sphere.hpp>

using std::cout;
using std::endl;

template <typename T>
Matrix<T,Dynamic,1> karcherMeanWeighted(const Matrix<T,Dynamic,1>& p0,
  const Matrix<T,Dynamic,Dynamic>& q, const Matrix<T,Dynamic,1>& w, 
  uint32_t maxIter)
{
  uint32_t D = p0.size();
  Matrix<T,Dynamic,1> p = p0;
  p /= p.norm();

  Sphere<T> M(D);
  T W = w.sum();
  if (W == 0.0) return p;
  for(uint32_t i=0; i< maxIter; ++i)
  {
    Matrix<T,Dynamic,Dynamic> x = M.Log_p(p,q,w);
//    cout<<q.rows()<<" "<<q.cols()<<endl;
//    cout<<x.rows()<<" "<<x.cols()<<endl;
    Matrix<T,Dynamic,1> x_mean = (x*w)/W; // weighted mean in TpS
//    cout<<"w "<<w.transpose()<<endl;
//    cout<<"W "<<W<<endl;
//    cout<<"x_mean "<<x_mean.transpose()<<endl;
    p = M.Exp_p(p,x_mean);
    if(x_mean.norm() <1.e-8)
    {
#ifndef NDEBUG
      cout<<"converged after "<<i<<" residual="<<x_mean.norm()<<endl;
#endif
      break;
    }
//      cout<<"@"<<i<<" residual="<<x_mean.norm()<<endl;
  }
  return p;
}

template <typename T>
Matrix<T,Dynamic,1> karcherMean(const Matrix<T,Dynamic,1>& p0,
  const Matrix<T,Dynamic,Dynamic>& q, Matrix<T,Dynamic,Dynamic>& x, 
  const VectorXu& z, uint32_t k, uint32_t maxIter, uint32_t zDivider=1)
{
  uint32_t D = p0.size();
  uint32_t N = q.cols();
  Matrix<T,Dynamic,1> p = p0;
  p /= p.norm();

  Sphere<T> M(D);
  for(uint32_t t=0; t< maxIter; ++t)
  {
    M.Log_p(p,q,z,k,x,zDivider);
//    cout<<q.rows()<<" "<<q.cols()<<endl;
//    cout<<x.rows()<<" "<<x.cols()<<endl;
    Matrix<T,Dynamic,1> x_mean(D); x_mean.setZero();
    for(uint32_t i=0; i<N; ++i)
      if(z(i)/zDivider == k)
        x_mean += x.col(i);
    x_mean /= N;
//    cout<<"w "<<w.transpose()<<endl;
//    cout<<"W "<<W<<endl;
//    cout<<"x_mean "<<x_mean.transpose()<<endl;
    p = M.Exp_p(p,x_mean);
    if(x_mean.norm() <1.e-8)
    {
#ifndef NDEBUG
      cout<<"converged after "<<t<<" residual="<<x_mean.norm()<<endl;
#endif
      break;
    }
//      cout<<"@"<<t<<" residual="<<x_mean.norm()<<endl;
  }
  return p;
}

template <typename T>
Matrix<T,Dynamic,1> karcherMean(const Matrix<T,Dynamic,1>& p0,
  const Matrix<T,Dynamic,Dynamic>& q, uint32_t maxIter)
{
  uint32_t D = p0.size();
  Matrix<T,Dynamic,1> p = p0;
  p /= p.norm();

  Sphere<T> M(D);
  T W = q.cols();
  for(uint32_t i=0; i< maxIter; ++i)
  {
    Matrix<T,Dynamic,Dynamic> x = M.Log_p(p,q);
//    cout<<q.rows()<<" "<<q.cols()<<endl;
//    cout<<x.rows()<<" "<<x.cols()<<endl;
    Matrix<T,Dynamic,1> x_mean = x.rowwise().sum()/W; // weighted mean in TpS
//    cout<<"w "<<w.transpose()<<endl;
//    cout<<"W "<<W<<endl;
//    cout<<"x_mean "<<x_mean.transpose()<<endl;
    p = M.Exp_p(p,x_mean);
    if(x_mean.norm() <1.e-8)
    {
#ifndef NDEBUG
      cout<<"converged after "<<i<<" residual="<<x_mean.norm()<<endl;
#endif
      break;
    }
//      cout<<"@"<<i<<" residual="<<x_mean.norm()<<endl;
  }
  return p;
}

/* 
 * compute the karcher means of several clusters indicated by labelx z
 * and return not only the means but also the points in the tangent planes
 */
template <typename T>
Matrix<T,Dynamic,Dynamic> karcherMeanMultiple(const Matrix<T,Dynamic,Dynamic>& p0s,
  const Matrix<T,Dynamic,Dynamic>& q, Matrix<T,Dynamic,Dynamic>& x, 
  const VectorXu& z, uint32_t K, uint32_t maxIter)
{
  assert(p0s.cols() == K);
  uint32_t D = p0s.rows();
  uint32_t N = q.cols();

  Matrix<T,Dynamic,Dynamic> x_mean(D,K);
  Matrix<T,Dynamic,1> Ns(K);
  Matrix<T,Dynamic,1> residuals(K);
  Matrix<T,Dynamic,Dynamic> ps = p0s;
  for(uint32_t k=0; k<K; ++k)
    ps.col(k) /= ps.col(k).norm();

  Sphere<T> M(D);
  for(uint32_t t=0; t< maxIter; ++t)
  {
//    cout<<x.transpose()<<endl;
    M.Log_ps(ps,q,z,x);
//    cout<<q.rows()<<" "<<q.cols()<<endl;
//    cout<<x.rows()<<" "<<x.cols()<<endl;
//    cout<<x.transpose()<<endl;
    //TODO: parallelize this?
    x_mean.setZero();
    Ns.setZero();
    for(uint32_t i=0; i<N; ++i)
    {
      x_mean.col(z(i)) += x.col(i);
      Ns(z(i)) ++;
    }
#pragma omp parallel for
    for(uint32_t k=0; k<K; ++k)
      if(Ns(k) > 0)
    {
      x_mean.col(k) /= Ns(k);
//      cout<<"karcherMeanMultiple: "<<endl<<x_mean<<endl;
      ps.col(k) = M.Exp_p(ps.col(k),x_mean.col(k));
      residuals(k) = x_mean.col(k).norm();
    }else
      residuals(k) = 0.0;
//    Matrix<T,Dynamic,1> x_mean = x.rowwise().sum()/W; // weighted mean in TpS
//    cout<<"w "<<w.transpose()<<endl;
//    cout<<"W "<<W<<endl;
//    cout<<"x_mean "<<x_mean.transpose()<<endl;
    if((residuals.array() <1.e-8).all())
    {
#ifndef NDEBUG
     cout<<"converged after "<<t<<" residuals = "<<residuals.transpose()<<endl;
#endif
      break;
    }
//      cout<<"@"<<t<<" residual="<<residuals.transpose()<<endl;
  }
  return ps;
}


//VectorXd karcherMeanWeighted(const VectorXd& p0,
//  const MatrixXd& q, const VectorXd& w, uint32_t maxIter)
//{
//  VectorXd p = p0;
//  Sphere<> M;
//  double W = w.sum();
//  for(uint32_t i=0; i< maxIter; ++i)
//  {
//    MatrixXd x = M.Log_p(p,q);
//    VectorXd x_mean = (x*w)/W;
//    p = M.Exp_p(p,x_mean);
//    if(x_mean.norm() <1e-6*6.)
//    {
//      cout<<"converged after "<<i<<" residual="<<x_mean.norm()<<endl;
//      break;
//    }
//  }
//  return p;
//}


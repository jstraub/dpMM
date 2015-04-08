/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>, Randi Cabezas <rcabezas@csail.mit.edu>                    
 * Licensed under the MIT license. See the license file LICENSE.                
 */

#pragma once

#include <Eigen/Dense>

#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/distributions/inverse_gamma.hpp>

#include <boost/random/chi_squared_distribution.hpp>
#include <boost/random/normal_distribution.hpp>

#include "distribution.hpp"
#include "normal.hpp"

using namespace Eigen;
using std::endl;
using std::cout;

template<typename T>
class IW : public Distribution<T>
{
public:
  Matrix<T,Dynamic,Dynamic> Delta_;
  T nu_;
  uint32_t D_;

  IW(const Matrix<T,Dynamic,Dynamic>& Delta, T nu, boost::mt19937 *pRndGen);
  IW(const Matrix<T,Dynamic,Dynamic>& Delta, T nu, const Matrix<T,Dynamic,Dynamic>& scatter, 
	 const Matrix<T,Dynamic,1>& mean, T counts, boost::mt19937 *pRndGen);
  IW(const IW& iw);
  ~IW();

  IW<T>* copy();

  IW<T> posterior(const Matrix<T,Dynamic,Dynamic>& x, const VectorXu& z,
      uint32_t k, uint32_t zDivider=1);
  IW<T> posterior(const Matrix<T,Dynamic,Dynamic>& outer, T count) const;
  IW<T> posterior() const ;
  void resetSufficientStatistics();

  Matrix<T,Dynamic,Dynamic> sample();
//  Normal<T> sample();
  T logPdf(const Matrix<T,Dynamic,Dynamic>& Sigma) const;
  T logPdf(const Normal<T>& normal) const;
  T logLikelihoodMarginalized() const; // log pdf of SS under NIW prior
  T logLikelihoodMarginalized(const Matrix<T,Dynamic,Dynamic>& Scatter, 
    T count) const;

  Matrix<T,Dynamic,Dynamic> mode() const;

  const Matrix<T,Dynamic,Dynamic>& scatter() const {return scatter_;};
  Matrix<T,Dynamic,Dynamic>& scatter() {return scatter_;};
  const Matrix<T,Dynamic,1>& mean() const {return mean_;};
  Matrix<T,Dynamic,1>& mean() {return mean_;};
  T count() const {return count_;};
  T& count() {return count_;};

private:
  boost::random::normal_distribution<> gauss_;

  // sufficient statistics
  Matrix<T,Dynamic,Dynamic> scatter_;
  Matrix<T,Dynamic,1> mean_;
  T count_;
};

typedef IW<double> IWd;
typedef IW<float> IWf;

/* prior for spherical covariances (i.e. \Sigma = \sigma^2 I) */
template<typename T>
class IW_spherical : public Distribution<T>
{
public:
  T delta_;
  T nu_;
  uint32_t D_;

  IW_spherical(T delta, T nu, uint32_t D, boost::mt19937 *pRndGen);
  IW_spherical(const IW_spherical& iw);
  ~IW_spherical();

  IW_spherical<T>* copy();

  IW_spherical<T> posterior(const Matrix<T,Dynamic,Dynamic>& x, 
      const VectorXu& z, uint32_t k, uint32_t zDivider=1);
//  T sample();
//  T logPdf(T sigma);
////  T logPdf(const Normal& normal);

private:
  boost::random::normal_distribution<> gauss_;
};

// ----------------------------------------------------------------------------

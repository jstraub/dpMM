#pragma once

#include <Eigen/Dense>

#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/distributions/inverse_gamma.hpp>

#include <boost/random/chi_squared_distribution.hpp>
#include <boost/random/normal_distribution.hpp>

#include "distribution.hpp"
#include "normal.hpp"

#define PI 3.141592653589793

using namespace Eigen;
using namespace std;

template<typename T>
class NIW : public Distribution<T>
{
public:
  Matrix<T,Dynamic,Dynamic> Delta_;
  Matrix<T,Dynamic,1> theta_;
  T nu_,kappa_;
  uint32_t D_;

  NIW(const Matrix<T,Dynamic,Dynamic>& Delta, 
    const Matrix<T,Dynamic,Dynamic>& theta, T nu,  T kappa, 
    boost::mt19937 *pRndGen);
  NIW(const NIW& niw);
  ~NIW();

  NIW<T>* copy();

  NIW<T> posterior(const Matrix<T,Dynamic,Dynamic>& x, const VectorXu& z, 
    uint32_t k);
  NIW<T> posterior() const;
  void getSufficientStatistics(const Matrix<T,Dynamic,Dynamic>& x, 
    const VectorXu& z, uint32_t k);
  T logProb(const Matrix<T,Dynamic,Dynamic>& x_i) const;
  T logPosteriorProb(const Matrix<T,Dynamic,Dynamic>& x, VectorXu& z, uint32_t k, 
    uint32_t i);

  Normal<T> sample();
  Normal<T> sampleFromPosterior();

  T logPdf(const Normal<T>& normal) const;
  T logPdfMarginalized() const; // log pdf of SS under NIW prior
  T logPdfUnderPriorMarginalizedMerged(const NIW<T>& other) const;

  T logLikelihoodMarginalized(const Matrix<T,Dynamic,Dynamic>& Scatter, 
      const Matrix<T,Dynamic,1>& mean, T count) const;
  void print() const;

  virtual NIW<T>* merge(const NIW<T>& other);
  void fromMerge(const NIW<T>& niwA, const NIW<T>& niwB);

  const Matrix<T,Dynamic,Dynamic>& scatter() const {return scatter_;};
  Matrix<T,Dynamic,Dynamic>& scatter() {return scatter_;};
  const Matrix<T,Dynamic,1>& mean() const {return mean_;};
  Matrix<T,Dynamic,1>& mean() {return mean_;};
  T count() const {return count_;};
  T& count() {return count_;};

  void computeMergedSS( const NIW<T>& niwA, 
    const NIW<T>& niwB, Matrix<T,Dynamic,Dynamic>& scatterM, 
    Matrix<T,Dynamic,1>& muM, T& countM) const;

private:
  boost::random::normal_distribution<> gauss_;

  // sufficient statistics
  Matrix<T,Dynamic,Dynamic> scatter_;
  Matrix<T,Dynamic,1> mean_;
  T count_;

};

typedef NIW<double> NIWd;
typedef NIW<float> NIWf;


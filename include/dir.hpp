#pragma once

#include <Eigen/Dense>
#include <time.h>
#include <vector>

#include <boost/random/gamma_distribution.hpp> // for gamma_distribution.
#include <boost/math/special_functions/gamma.hpp>

#include "distribution.hpp"
#include "cat.hpp"

using namespace Eigen;
using namespace std;

template<typename T>
class Dir : public Distribution<T>
{
public:
  uint32_t K_;
  Matrix<T,Dynamic,1> alpha_;

  Dir(const Matrix<T,Dynamic,1>& alpha, boost::mt19937 *pRndGen);
  Dir(const Dir& other);
  ~Dir();

  Cat<T> sample();
  Dir<T> posterior(const VectorXu& z);
  Dir<T> posteriorFromCounts(const Matrix<T,Dynamic,1>& counts);
  Dir<T> posteriorFromCounts(const VectorXu& counts);

  T logPdf(const Cat<T>& cat);

  uint32_t K(){return K_;}

private:
  vector<boost::random::gamma_distribution<> > gammas_;

  Matrix<T,Dynamic,1> samplePdf();
};

typedef Dir<double> Dird;
typedef Dir<float> Dirf;


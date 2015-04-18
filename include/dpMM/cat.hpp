/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>                    
 * Licensed under the MIT license. See the license file LICENSE.                
 */

#pragma once

#include <Eigen/Dense>
#include <time.h>
#include <memory>

//#include <boost/random/uniform_int_distribution.hpp>
//#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_01.hpp>

#include <dpMM/distribution.hpp>
#include <dpMM/sampler.hpp>

using namespace Eigen;
using std::cout;
using std::endl;

template<typename T>
class Cat : public Distribution<T>
{
public:
  uint32_t K_;
  Matrix<T,Dynamic,1> pdf_;
  Matrix<T,Dynamic,1> cdf_;

  /* constructor from pdf */
  Cat(const Matrix<T,Dynamic,1>& pdf, boost::mt19937 *pRndGen);
  /* constructor from indicators - estimates from counts */
  Cat(const VectorXu& z, boost::mt19937 *pRndGen);
  /* copy constructor */
  Cat(const Cat& other);
  virtual ~Cat();

  uint32_t sample();
  void sample(VectorXu& z);

  T logPdf(const Matrix<T,Dynamic,1>& x) const 
  {
    //assuming x is all zeros except one element
    assert(x.rows()==K_); //data dimmension should agree with # categories
    for(uint32_t d=0; d<K_; ++d) {
      if(x(d)==1) {
        return(log(pdf_(d))); 
      }
    }
    assert(false); //invalid data (at least one element must be one) [this should never happen] 
    return(-1); 
  };

  T logPdf(uint32_t x) const 
  {
    assert(x < K_);
    return(log(pdf_(x))); 
  };

  T logPdfOfSS(const Matrix<T,Dynamic,1>& x) const 
  {
    
    assert(x.rows()==K_); //data dimmension should agree with # categories
    T logPdf = 0;
    for(uint32_t d=0; d<K_; ++d) {
      logPdf += x(d) * log(pdf_(d)); 
    }
    return logPdf; 
  };

  const Matrix<T,Dynamic,1>& pdf() const {return pdf_;};
  void pdf(const Matrix<T,Dynamic,1>& pdf){
    pdf_ = pdf;
    updateCdf();
  };
  const Matrix<T,Dynamic,1>& cdf() const {return cdf_;};

  void print() const;

private:
  boost::uniform_01<T> unif_;
  void updateCdf();
};

typedef Cat<float> Catf;
typedef Cat<double> Catd;

template<typename T>
inline T logSumExpRow(const Matrix<T,Dynamic,Dynamic>& pdf, uint32_t i)
{
  T max = pdf.row(i).maxCoeff();
  return log((pdf.row(i).array()-max).exp().matrix().sum()) + max;
}

template<typename T>
inline T logSumExpCol(const Matrix<T,Dynamic,Dynamic>& pdf, uint32_t i)
{
  T max = pdf.col(i).maxCoeff();
  return log((pdf.col(i).array()-max).exp().matrix().sum()) + max;
}


template<typename T>
inline T logSumExp(const Matrix<T,Dynamic,1>& pdf)
{
  T max = pdf.maxCoeff();
  return log((pdf.array()-max).exp().matrix().sum()) + max;
}

//template<typename T>
//inline T logSumExp(const Matrix<T,1,Dynamic>& pdf)
//{
//  T max = pdf.maxCoeff();
//  return log((pdf.array()-max).exp().matrix().sum()) + max;
//}


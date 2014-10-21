#pragma once

#include <iostream>
#include <stdint.h>
#include <Eigen/Dense>

#include "global.hpp"
#include "clData.hpp"

using namespace Eigen;
using std::cout;
using std::endl;

template<typename T>
class BaseMeasure
{
public:
  BaseMeasure()
  {};
  virtual ~BaseMeasure()
  {};

  virtual BaseMeasure<T>* copy() = 0;

  virtual T logLikelihood(const Matrix<T,Dynamic,1>& x) const =0;
  virtual T logLikelihood(const Matrix<T,Dynamic,Dynamic>& x, uint32_t i) const 
    {return logLikelihood(x.col(i));};

  virtual T logLikelihood(const vector<Matrix<T,Dynamic,1> >&x)
	  {assert(false); return 0.0;};
  virtual T logLikelihood(const vector<Matrix<T,Dynamic,Dynamic> >&x, uint32_t i) const 
    {return logLikelihood(x[i]);};

  virtual void posterior(const Matrix<T,Dynamic,Dynamic>&x, const VectorXu& z, 
      uint32_t k) =0;
  virtual void posterior(const vector<Matrix<T,Dynamic,Dynamic> >&x, const VectorXu& z, 
      uint32_t k)
  {assert(false);};
  virtual void posterior(const boost::shared_ptr<ClData<T> >& cldp, uint32_t k) 
  {assert(false);};

  virtual T logPdfUnderPrior() const =0;
  virtual void print() const =0;

//  //TODO UGLY
//  virtual Matrix<T,Dynamic,Dynamic>& scatter()
//    {assert(false); return Matrix<T,Dynamic,Dynamic>::Zero(1,1);};
//  //TODO UGLY
//  virtual T& count()
//    {assert(false); return 0;};
};



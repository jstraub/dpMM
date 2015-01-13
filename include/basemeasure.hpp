/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>                    
 * Licensed under the MIT license. See the license file LICENSE.                
 */
#pragma once

#include <iostream>
#include <stdint.h>
#include <Eigen/Dense>

#include "global.hpp"
#include "clData.hpp"

using namespace Eigen;
using std::cout;
using std::endl;

//TODO add to remaining classes. I added them to the ones I found.. RC 10/29/14.
enum baseMeasureType {BASE, NIW_SAMPLED, NIW_MARGINALIZED, 
					  DIR_SAMPLED, NIW_TANGENT, NIW_SPHERE, 
					  NIW_SPHERE_FULL, UNIF_SPHERE}; 


template<typename T>
class BaseMeasure
{
public:
  BaseMeasure()
  {};
  virtual ~BaseMeasure()
  {};

  virtual baseMeasureType getBaseMeasureType() const {return(BASE); }

  virtual BaseMeasure<T>* copy() = 0;

  virtual T logLikelihood(const Matrix<T,Dynamic,1>& x) const =0;
  virtual T logLikelihood(const Matrix<T,Dynamic,Dynamic>& x, uint32_t i) const 
    {return logLikelihood(x.col(i));};

  virtual void posterior(const Matrix<T,Dynamic,Dynamic>&x, const VectorXu& z, 
      uint32_t k) =0;
  virtual void posterior(const vector<Matrix<T,Dynamic,Dynamic> >&x, const VectorXu& z, 
      uint32_t k)
  {assert(false);};
  virtual void posterior(const boost::shared_ptr<ClData<T> >& cldp, uint32_t k) 
  {assert(false);};

  virtual T logPdfUnderPrior() const =0;
  virtual void print() const =0;
  virtual uint32_t getDim() const =0; // {return(0);};

//  //TODO UGLY
//  virtual Matrix<T,Dynamic,Dynamic>& scatter()
//    {assert(false); return Matrix<T,Dynamic,Dynamic>::Zero(1,1);};
//  //TODO UGLY
//  virtual T& count()
//    {assert(false); return 0;};
};



/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>, Randi Cabezas <rcabezas@csail.mit.edu>                    
 * Licensed under the MIT license. See the license file LICENSE.                
 */
#pragma once

#include <iostream>
#include <stdint.h>
#include <Eigen/Dense>

#include "global.hpp"
#include "clGMMData.hpp"

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

  // log likelihood under the model sampled from this base measure
  virtual T logLikelihood(const Matrix<T,Dynamic,1>& x) const =0;
  virtual T logLikelihood(const Matrix<T,Dynamic,Dynamic>& x, uint32_t i) const 
    {return logLikelihood(x.col(i));};

  //adding this here to let derived classes to work with SS rather than data
  virtual T logLikelihoodFromSS(const Matrix<T,Dynamic,1>& x) const
	  {assert(false); return(1e20);}
  virtual void posteriorFromSS(const vector<Matrix<T,Dynamic,1> >&x, const VectorXu& z, uint32_t k)
	{assert(false);}
  virtual void posteriorFromSS(const Matrix<T,Dynamic,1> &x)
	{assert(false);}


  // sample new model from the possterior under this base measure
  virtual void posterior(const Matrix<T,Dynamic,Dynamic>&x, const VectorXu& z, 
      uint32_t k) =0;
  virtual void posterior(const vector<Matrix<T,Dynamic,Dynamic> >&x, const VectorXu& z, 
      uint32_t k)
  {assert(false);};
  virtual void posterior(const boost::shared_ptr<ClGMMData<T> >& cldp, uint32_t k) 
  {assert(false);};

  // log pdf value of the current model under the base measure
  virtual T logPdfUnderPrior() const =0;

  // log pdf value of the data point under the base measure (integrated over
  // model parameters - could use Monte carlo integration if not analytic)
  virtual T logPdfUnderPriorMarginalized(const Matrix<T,Dynamic,1>& x) {return 0.;};
  //  log pdf value of the data point under the base measure using the
  //  sufficient statistics stored with the base measure
  virtual T logPdfUnderPriorMarginalized() const {return 0;};

  virtual void print() const =0;
  virtual uint32_t getDim() const =0; // {return(0);};

//  //TODO UGLY
//  virtual Matrix<T,Dynamic,Dynamic>& scatter()
//    {assert(false); return Matrix<T,Dynamic,Dynamic>::Zero(1,1);};
//  //TODO UGLY
//  virtual T& count()
//    {assert(false); return 0;};
};



/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>                    
 * Licensed under the MIT license. See the license file LICENSE.                
 */
#pragma once

#include <Eigen/Dense>
#include <boost/shared_ptr.hpp>
#include <fstream>

#include <dpMM/global.hpp>
#include <dpMM/clGMMData.hpp>

using namespace Eigen;
using boost::shared_ptr;

template<class T>
class DpMM
{
public:
  DpMM()
  {};
  virtual ~DpMM()
  {};

  virtual void initialize(const shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx)
    {assert(false);};
  virtual void initialize(const Matrix<T,Dynamic,Dynamic>& x)
  {spx_=shared_ptr<Matrix<T,Dynamic,Dynamic> >(new Matrix<T,Dynamic,Dynamic>(x));
  initialize(spx_);};
  virtual void initialize(const shared_ptr<ClGMMData<T> >& cld) 
    {assert(false);};
  virtual void sampleLabels() = 0;
  virtual void sampleParameters() = 0;
  virtual void proposeSplits() {};
  virtual void proposeMerges() {};
  // call this right before sampleParameters()
  virtual void resampleFromBase(uint32_t Kmax) {};
  virtual const VectorXu & getLabels() = 0;
  virtual Matrix<T,Dynamic,1> getCounts() = 0;
  virtual uint32_t getK() const = 0;
  virtual double logJoint() { return 0.0;}; 

  virtual double evalLogLikelihood(const Matrix<T,Dynamic,1>& x){
    return 0;}

//  virtual MatrixXu mostLikelyInds(uint32_t n) 
//{ return MatrixXu::Zero(n,1);};
  virtual MatrixXu mostLikelyInds(uint32_t n, Matrix<T,Dynamic,Dynamic>& logLikes)
  { return MatrixXu::Zero(n,1);};

  virtual void dump(std::ofstream& fOutMeans, std::ofstream& fOutCovs)
  {};

  shared_ptr<Matrix<T,Dynamic,Dynamic> > spx_;
private:
};


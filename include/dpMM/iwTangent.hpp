/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>                    
 * Licensed under the MIT license. See the license file LICENSE.                
 */
#pragma once

#include <Eigen/Dense>

#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/distributions/inverse_gamma.hpp>

#include <boost/random/chi_squared_distribution.hpp>
#include <boost/random/normal_distribution.hpp>

#include <dpMM/basemeasure.hpp>
#include <dpMM/distribution.hpp>
#include <dpMM/normalSphere.hpp>
#include <dpMM/iw.hpp>
#include <dpMM/sphere.hpp>
#include <dpMM/karcherMean.hpp>

using namespace Eigen;
using std::endl;
using std::cout;

/* Actually just places an IW prior on covariances in the tangent
 * plane. 
 *
 * The point of tangentcy has to be set "manually". This makes the
 * clase usefull for models that externaly update the point of tangency
 * such as the Manhattan Frame 
 */
template<typename T>
class IwTangent : public BaseMeasure<T>
{
public:

  IW<T> iw0_; // IW prior on the covariance of the normal in T_\muS^D
  Sphere<T> S_;
  NormalSphere<T> normalS_; // normal on sphere

  IwTangent(const IW<T>& iw, boost::mt19937* pRndGen);
  ~IwTangent();

  virtual BaseMeasure<T>* copy();
  virtual IwTangent<T>* copyNative();

  virtual baseMeasureType getBaseMeasureType() const {return(NIW_TANGENT); }

  /* for any point on the sphere; maps into T_muS and rotates north first */
  virtual T logLikelihood(const Matrix<T,Dynamic,1>& x) const ;
  virtual T logLikelihood(const Matrix<T,Dynamic,Dynamic>& x, uint32_t i) const 
    {return logLikelihood(x.col(i));};
  // right now this does not support actual scatter!  it supports
  // weighted directional data though.  count is the weight
  // [counts,karcherMean,scatter around KarcherMean] 
  virtual T logLikelihoodFromSS(const Matrix<T,Dynamic,1>& x) const;
  /* assumes x is already in T_northS */
  virtual T logLikelihoodNorth(const Matrix<T,Dynamic,1>& x) const ;

  void posterior(const Matrix<T,Dynamic,Dynamic>& x, const VectorXu& z, 
    uint32_t k);
  void posterior(const shared_ptr<ClGMMData<T> >& cldp, uint32_t k);
  /* assumes the x are already in T_northS correctly */
  void posteriorFromPtsInTpS(const Matrix<T,Dynamic,Dynamic>& x, 
    const VectorXu& z, uint32_t k);
  // right now this does not support actual scatter!  it supports
  // weighted directional data though.  count is the weight
  // [counts,karcherMean,scatter around KarcherMean] 
  void posteriorFromSS(const vector<Matrix<T,Dynamic,1> >&x, const
      VectorXu& z, uint32_t k);
  void posteriorFromSS(const Matrix<T,Dynamic,1> &x);

  void sample();

  T logPdfUnderPrior() const;
  T logPdfUnderPriorMarginalized() const;
//  T logPdfUnderPriorMarginalizedMerged(const shared_ptr<IwTangent<T> >& other) const;

  void print() const;
  virtual uint32_t getDim() const {return(uint32_t(normalS_.D_));}; 

//  virtual IwTangent<T>* merge(const IwTangent<T>& other);
//  void fromMerge(const IwTangent<T>& niwA, const IwTangent<T>& niwB);

  const Matrix<T,Dynamic,Dynamic>& scatter() const {return iw0_.scatter();};
  Matrix<T,Dynamic,Dynamic>& scatter() {return iw0_.scatter();};
  T count() const {return iw0_.count();};
  T& count() {return iw0_.count();};

  const Matrix<T,Dynamic,Dynamic>& Sigma() const {return normalS_.Sigma();};

  const Matrix<T,Dynamic,1>& getMean() const {return normalS_.getMean();};
  void setMean(const Matrix<T,Dynamic,1>& mean) {return normalS_.setMean(mean);};

private:

//  void computeMergedSS( const IwTangent<T>& niwA, 
//    const IwTangent<T>& niwB, Matrix<T,Dynamic,Dynamic>& scatterM, 
//    Matrix<T,Dynamic,1>& muM, T& countM) const;


};

typedef IwTangent<double> IwTangentd;
typedef IwTangent<float> IwTangentf;


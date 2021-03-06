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
#include <dpMM/clGMMData.hpp>

using namespace Eigen;
using std::endl;
using std::cout;

/* Actually just places an IW prior on covariances in the tangent plane */
template<typename T>
class NiwSphere : public BaseMeasure<T>
{
public:

  IW<T> iw0_; // IW prior on the covariance of the normal in T_\muS^D
  Sphere<T> S_;
  NormalSphere<T> normalS_; // normal on sphere

  NiwSphere(const IW<T>& iw, boost::mt19937* pRndGen);
  ~NiwSphere();

  virtual BaseMeasure<T>* copy();
  virtual NiwSphere<T>* copyNative();

  virtual baseMeasureType getBaseMeasureType() const {return(NIW_SPHERE); }

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
  T logPdfUnderPriorMarginalizedMerged(const shared_ptr<NiwSphere<T> >& other) const;

  void print() const;
  virtual uint32_t getDim() const {return(uint32_t(normalS_.D_));}; 

  virtual NiwSphere<T>* merge(const NiwSphere<T>& other);
  void fromMerge(const NiwSphere<T>& niwA, const NiwSphere<T>& niwB);

  const Matrix<T,Dynamic,Dynamic>& scatter() const {return iw0_.scatter();};
  Matrix<T,Dynamic,Dynamic>& scatter() {return iw0_.scatter();};
  T count() const {return iw0_.count();};
  T& count() {return iw0_.count();};

  const Matrix<T,Dynamic,Dynamic>& Sigma() const {return normalS_.Sigma();};

  const Matrix<T,Dynamic,1>& getMean() const {return normalS_.getMean();};
  void setMean(const Matrix<T,Dynamic,1>& mean) {return normalS_.setMean(mean);};

private:

  void computeMergedSS( const NiwSphere<T>& niwA, 
    const NiwSphere<T>& niwB, Matrix<T,Dynamic,Dynamic>& scatterM, 
    Matrix<T,Dynamic,1>& muM, T& countM) const;


};

typedef NiwSphere<double> NiwSphered;
typedef NiwSphere<float> NiwSpheref;


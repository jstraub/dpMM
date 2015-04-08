/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>                    
 * Licensed under the MIT license. See the license file LICENSE.                
 */
#pragma once

#include <Eigen/Dense>

#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/distributions/inverse_gamma.hpp>

#include <boost/random/chi_squared_distribution.hpp>
#include <boost/random/normal_distribution.hpp>

#include "basemeasure.hpp"
#include "distribution.hpp"
#include "normalSphere.hpp"
#include "iw.hpp"
#include "sphere.hpp"
#include "karcherMean.hpp"
#include "clGMMData.hpp"

using namespace Eigen;
using std::endl;
using std::cout;


template<typename T>
class NiwSphereFull : public BaseMeasure<T>
{
public:

  uint32_t D_;
  IW<T> iw0_; // IW prior on the covariance of the normal in T_\muS^D
  Sphere<T> S_;
  NormalSphere<T> normalS_; // sampled normal on sphere - distribution of data

  NiwSphereFull(const IW<T>& iw, boost::mt19937* pRndGen);
  ~NiwSphereFull();

  virtual baseMeasureType getBaseMeasureType() const {return(NIW_SPHERE_FULL); }

  virtual BaseMeasure<T>* copy();
  virtual NiwSphereFull<T>* copyNative();

  /* for any point on the sphere; maps into T_muS and rotates north first */
  virtual T logLikelihood(const Matrix<T,Dynamic,1>& x) const ;
  virtual T logLikelihood(const Matrix<T,Dynamic,Dynamic>& x, uint32_t i) const 
    {return logLikelihood(x.col(i));};
  /* assumes x is already in T_northS */
  virtual T logLikelihoodNorth(const Matrix<T,Dynamic,1>& x) const ;

  void posterior(const Matrix<T,Dynamic,Dynamic>& x, const VectorXu& z, 
    uint32_t k);
  void posterior(const shared_ptr<ClGMMData<T> >& cldp, uint32_t k);
  /* assumes the x are already in T_northS correctly */
  void posteriorFromPtsInTpS(const Matrix<T,Dynamic,Dynamic>& x, 
    const VectorXu& z, uint32_t k, uint32_t zDivider=1);

  void sampleMergedParams();
  /* samples Cov and then proposes means (covs are always accepted!) */
  void sample();
  /* proposes means and covariances jointly (and rejects them jointly as well)*/
  void sample_2();

  T logPdfUnderPrior() const;
  T logPdfUnderPriorMarginalized() const;
  T logPdfUnderPriorMarginalizedMerged(const shared_ptr<NiwSphereFull<T> >& other) const;
  T qRandomMuProposal() const;

  void print() const;

  virtual NiwSphereFull<T>* merge(const NiwSphereFull<T>& other);
  void fromMerge(const NiwSphereFull<T>& niwA, const NiwSphereFull<T>& niwB);

  const Matrix<T,Dynamic,Dynamic>& scatter() const {return iw0_.scatter();};
  Matrix<T,Dynamic,Dynamic>& scatter() {return iw0_.scatter();};
  const Matrix<T,Dynamic,1>& mean() const {return iw0_.mean();};
  Matrix<T,Dynamic,1>& mean() {return iw0_.mean();};
  T count() const {return iw0_.count();};
  T& count() {return iw0_.count();};

  const Matrix<T,Dynamic,Dynamic>& Sigma() const {return normalS_.Sigma();};

  const Matrix<T,Dynamic,1>& getMean() const {return normalS_.getMean();};
  const Matrix<T,Dynamic,1>& getMeanKarch() const {return meanKarch_;};
  // this is the sample mean
  void setMeanKarch(const Matrix<T,Dynamic,1>& mean) {
    meanKarch_ = mean; 
  };
  void setMean(const Matrix<T,Dynamic,1>& mean) {normalS_.setMean(mean);};
  virtual uint32_t getDim() const {return(D_);};
private:

  void computeMergedSS( const NiwSphereFull<T>& niwA, 
    const NiwSphereFull<T>& niwB, Matrix<T,Dynamic,Dynamic>& scatterM, 
    Matrix<T,Dynamic,1>& meanM,
    Matrix<T,Dynamic,1>& muM, T& countM) const;

  Matrix<T,Dynamic,1> meanKarch_;

};

typedef NiwSphereFull<double> NiwSphereFulld;
typedef NiwSphereFull<float> NiwSphereFullf;


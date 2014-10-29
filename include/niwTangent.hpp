#pragma once

#include <Eigen/Dense>

#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/distributions/inverse_gamma.hpp>

#include <boost/random/chi_squared_distribution.hpp>
#include <boost/random/normal_distribution.hpp>

#include "basemeasure.hpp"
#include "distribution.hpp"
#include "normalSphere.hpp"
#include "niw.hpp"
#include "sphere.hpp"
#include "karcherMean.hpp"
#include "clData.hpp"

using namespace Eigen;
using namespace std;


template<typename T>
class NiwTangent : public BaseMeasure<T>
{
public:

  NIW<T> niw0_; // IW prior on the covariance of the normal in T_\muS^D
  Sphere<T> S_;
  NormalSphere<T> normalS_; // normal on sphere

  NiwTangent(const NIW<T>& niw, boost::mt19937* pRndGen);
  ~NiwTangent();

  virtual baseMeasureType getBaseMeasureType() const {return(NIW_TANGENT); }

  virtual BaseMeasure<T>* copy();
  virtual NiwTangent<T>* copyNative();

  /* for any point on the sphere; maps into T_muS and rotates north first */
  virtual T logLikelihood(const Matrix<T,Dynamic,1>& x) const ;
  virtual T logLikelihood(const Matrix<T,Dynamic,Dynamic>& x, uint32_t i) const 
    {return logLikelihood(x.col(i));};
  /* assumes x is already in T_northS */
  virtual T logLikelihoodNorth(const Matrix<T,Dynamic,1>& x) const ;

  void posterior(const Matrix<T,Dynamic,Dynamic>& x, const VectorXu& z, 
    uint32_t k);
  void posterior(const boost::shared_ptr<ClData<T> >& cldp, uint32_t k);
  /* assumes the x are already in T_northS correctly */
  void posteriorFromPtsInTpS(const Matrix<T,Dynamic,Dynamic>& x, 
    const VectorXu& z, uint32_t k);

  void sample();

  T logPdfUnderPrior() const;
  T logPdfUnderPriorMarginalized() const;
  T logPdfUnderPriorMarginalizedMerged(const boost::shared_ptr<NiwTangent<T> >& other) const;

  void print() const;

  virtual NiwTangent<T>* merge(const NiwTangent<T>& other);
  void fromMerge(const NiwTangent<T>& niwA, const NiwTangent<T>& niwB);

  const Matrix<T,Dynamic,Dynamic>& scatter() const {return niw0_.scatter();};
  Matrix<T,Dynamic,Dynamic>& scatter() {return niw0_.scatter();};
  T count() const {return niw0_.count();};
  T& count() {return niw0_.count();};

  const Matrix<T,Dynamic,Dynamic>& Sigma() const {return normalS_.Sigma();};

  const Matrix<T,Dynamic,1>& getMean() const {return normalS_.getMean();};
  void setMean(const Matrix<T,Dynamic,1>& mean) {return normalS_.setMean(mean);};

  // extension
  Matrix<T,Dynamic,Dynamic> qqTSum_; // \sum q_i q_i^T
  Matrix<T,Dynamic,Dynamic> qqTAngSum_; // \sum q_i q_i^T * theta_{q_i \mu}
  Matrix<T,Dynamic,Dynamic> qqTAngSqSum_; // \sum q_i q_i^T * theta_{q_i \mu}^2
  Matrix<T,Dynamic,1> qSum_; // \sum q_i
  Matrix<T,Dynamic,1> qSumAngle_; // \sum q_i* theta_{q_i \mu}
  Matrix<T,Dynamic,1> qSumAngleSq_; // \sum q_i* theta_{q_i \mu}^2
  T sumAngle_; // \sum theta_{q_i \mu}
  T sumAngleSq_; // \sum theta_{q_i \mu}^2

  Matrix<T,Dynamic,Dynamic> scatterCorrection1(const Matrix<T,Dynamic,1>& p) const;

private:

  void computeMergedSS( const NiwTangent<T>& niwA, 
    const NiwTangent<T>& niwB, Matrix<T,Dynamic,Dynamic>& scatterM, 
    Matrix<T,Dynamic,1>& meanM, Matrix<T,Dynamic,1>& pM, T& countM) const;

  // sufficient statistics
  Matrix<T,Dynamic,Dynamic> scatter_;
  T count_;

};

typedef NiwTangent<double> NiwTangentd;
typedef NiwTangent<float> NiwTangentf;


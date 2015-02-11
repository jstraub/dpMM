/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>                    
 * Licensed under the MIT license. See the license file LICENSE.                
 */

#pragma once 

#include <Eigen/Dense>
#include <iostream>

#include <boost/random/gamma_distribution.hpp> // for gamma_distribution.

#include "basemeasure.hpp"
#include "cat.hpp"
#include "niwSphere.hpp"
#include "niwSphereFull.hpp"
//#include "niwTangent.hpp"

using namespace Eigen;
using std::cout;
using std::endl;

template<class B, typename T>
class LrCluster // : public BaseMeasure<T>
{
protected:
  boost::mt19937 *pRndGen_;
  T alpha_; 
  T logLikeSubClSplit_;
  T avgLogLikeData_; // data log likelihood in subclusters
  vector<T> avgLogLikeDataHist_; // data log likelihood in subclusters
  bool splittable_;
  int32_t nDecrease_;

  Matrix<T,Dynamic,1> kernel_;
//  Matrix<T,Dynamic,1> logLikeWindow;

public:
  LrCluster(const shared_ptr<B>& theta, T alpha, boost::mt19937 *pRndGen);
  virtual ~LrCluster();

  virtual LrCluster<B,T>* copy();
  virtual T logLikelihood(const Matrix<T,Dynamic,1>& x) const;

  void posterior(const Matrix<T,Dynamic,Dynamic>& x, const VectorXu& z, uint32_t k);

  Matrix<T,1,2> getSubclusterLogPdf(const Matrix<T,Dynamic,1>& x);
  T dataLogLikelihoodMarginalized();
//  void dataLogLikelihoodMarginalizedMergedWith(
  virtual T logPdfUnderPrior() const;
  virtual T logPdfUnderPriorMarginalized();
  virtual T dataLogLikelihoodMarginalizedMerged(const shared_ptr<LrCluster<B,T> >& other);
  virtual void print() const;

  virtual T& logLikelihoodOfSubclusterSplit(){ return logLikeSubClSplit_;};
  virtual const T& logLikelihoodData(){ return avgLogLikeData_;};

  virtual T qRandomParamProposal() { return 0.0; };
//  virtual T qRandomSplitProposal() { return 0.0; };

//  virtual void sampleUpper();
//  virtual void sampleLR();
  void sample();

  LrCluster<B,T>* merge(const shared_ptr<LrCluster<B,T> >& other);

  bool splittable();
  void setSplittable() { splittable_ = true;};
  void resetSplittable() { 
    splittable_ = false; 
    nDecrease_ = 0; 
    avgLogLikeData_ = -999999999.0; 
    avgLogLikeDataHist_.clear(); };
  void updateAvgLogLikeData(T avgLogLikeData);
  void updateAvgLogLikeData();

  shared_ptr<B>& getUpper(){ return theta_;};
  shared_ptr<B>& getR(){ return thetaR_;};
  shared_ptr<B>& getL(){ return thetaL_;};

  T count() const {return theta_->count();};
  T countR(){ return thetaR_->count();};
  T countL(){ return thetaL_->count();};

  T avgLogLikeData(){ return avgLogLikeData_;};

  void posteriorUpper();
  void posteriorLR(const Matrix<T,Dynamic,Dynamic>& x, 
    const VectorXu& z, uint32_t k);

  T& alpha(){return alpha_;};
  T& stickL(){return stickL_;};
  T& stickR(){return stickR_;};
  
  T stickL_, stickR_;
  shared_ptr<B> theta0_; // prior
  shared_ptr<B> theta_; // full cluster
  shared_ptr<B> thetaL_; // left cluster
  shared_ptr<B> thetaR_; // right cluster
private:

  void posteriorSticks();
};

// ---------------------------------- impl -----------------------------------

template<class B, typename T>
LrCluster<B,T>::LrCluster(const shared_ptr<B>& theta, T alpha,
  boost::mt19937 *pRndGen)
  : pRndGen_(pRndGen), alpha_(alpha), avgLogLikeData_(-9999999.), splittable_(false), 
  nDecrease_(0),
  stickL_(log(0.5)), stickR_(log(0.5)),
  theta0_(theta), 
  theta_(theta->copyNative()), 
  thetaL_(theta->copyNative()), 
  thetaR_(theta->copyNative())
{
  kernel_.setOnes(8);
  kernel_ /= 8.;
};

template<class B, typename T>
LrCluster<B,T>::~LrCluster()
{};

template<class B, typename T>
LrCluster<B,T>* LrCluster<B,T>::copy()
{
  LrCluster<B,T>* lrCl = new LrCluster<B,T>(theta0_,alpha_, pRndGen_);
  lrCl->stickL_ = stickL_;
  lrCl->stickR_ = stickR_;
  lrCl->theta_.reset(theta_->copyNative());
  lrCl->thetaL_.reset(thetaL_->copyNative());
  lrCl->thetaR_.reset(thetaR_->copyNative());
  return lrCl;
}

template<class B, typename T>
LrCluster<B,T>* LrCluster<B,T>::merge(const shared_ptr<LrCluster<B,T> >& other)
{
  LrCluster<B,T>* merged = new LrCluster<B,T>(
    shared_ptr<B>(theta_->merge(*other->getUpper())),
    alpha_,pRndGen_);
  merged->getL().reset(theta_->copyNative());
  merged->getR().reset(other->getUpper()->copyNative());
  return merged;
};

template<class B, typename T>
T LrCluster<B,T>::logPdfUnderPrior() const
{
//TODO might not need that one
  assert(false);
  return theta_->logPdfUnderPrior();
}

template<class B, typename T>
T LrCluster<B,T>::logPdfUnderPriorMarginalized()
{
//  this->posteriorUpper(); 
  return theta_->logPdfUnderPriorMarginalized();
}

template<class B, typename T>
T LrCluster<B,T>::dataLogLikelihoodMarginalizedMerged(
    const shared_ptr<LrCluster<B,T> >& other) 
{
//  LrCluster<B,T>* merged = new LrCluster<B,T>(
//    shared_ptr<B>(theta_->merge(*other->getUpper())),
//    alpha_,pRndGen_);
//  merged->getL().reset(theta_->copyNative());
//  merged->getR().reset(other->getUpper()->copyNative());
//  return merged;

  return theta_->logPdfUnderPriorMarginalizedMerged(other->getUpper());
};

template<class B, typename T>
void LrCluster<B,T>::sample()
{
  thetaL_->sample();
  thetaR_->sample();
  theta_->sample();
}

template<class B, typename T>
void LrCluster<B,T>::posterior(const Matrix<T,Dynamic,Dynamic>& x, 
  const VectorXu& z, uint32_t k)
{
  thetaL_->posterior(x,z,k);
  thetaR_->posterior(x,z,k+1);
  theta_->fromMerge(*thetaL_,*thetaR_);

//  thetaL_->sample();
//  thetaR_->sample();
//  theta_->sample();
  posteriorSticks();
}

//TODO: only to test the scatter correction!
template<>
void LrCluster<NiwSphere<double> ,double>::posterior(
    const Matrix<double,Dynamic,Dynamic>& x, const VectorXu& z, uint32_t k)
{
//  cout<<"::posterior posteriorFromPtsInTpS"<<endl;
  /* assumes that x is already correctly in T_northS */
  thetaL_->posteriorFromPtsInTpS(x,z,k);
  thetaR_->posteriorFromPtsInTpS(x,z,k+1);
  theta_->fromMerge(*thetaL_,*thetaR_);

//  thetaL_->sample();
//  thetaR_->sample();
//  theta_->sample();
  posteriorSticks();
}

template<>
void LrCluster<NiwSphereFull<double> ,double>::posterior(
    const Matrix<double,Dynamic,Dynamic>& x, const VectorXu& z, uint32_t k)
{
//  cout<<"::posterior posteriorFromPtsInTpS"<<endl;
  /* assumes that x is already correctly in T_northS */
  thetaL_->posteriorFromPtsInTpS(x,z,k);
  thetaR_->posteriorFromPtsInTpS(x,z,k+1);
  theta_->fromMerge(*thetaL_,*thetaR_);
//  theta_->posteriorFromPtsInTpS(x,z,k/2,2);

//  thetaL_->sample();
//  thetaR_->sample();
//  theta_->sample();
  posteriorSticks();
}

//template<>
//void LrCluster<NiwTangent<double> ,double>::posterior(
//    const Matrix<double,Dynamic,Dynamic>& x, const VectorXu& z, uint32_t k)
//{
////  cout<<"::posterior posteriorFromPtsInTpS"<<endl;
//  /* assumes that x is already correctly in T_northS */
//  thetaL_->posteriorFromPtsInTpS(x,z,k);
//  thetaR_->posteriorFromPtsInTpS(x,z,k+1);
//  // TODO: make sure that from merge works well if both L and R have exactly 
//  // the same linearization point
//  theta_->fromMerge(*thetaL_,*thetaR_);
//
////  thetaL_->sample();
////  thetaR_->sample();
////  theta_->sample();
//  posteriorSticks();
//}

template<>
void LrCluster<NiwSphere<float> , float>::posterior(
    const Matrix<float,Dynamic,Dynamic>& x, const VectorXu& z, uint32_t k)
{
  /* assumes that x is already correctly in T_northS */
  thetaL_->posteriorFromPtsInTpS(x,z,k);
  thetaR_->posteriorFromPtsInTpS(x,z,k+1);
  theta_->fromMerge(*thetaL_,*thetaR_);

//  thetaL_->sample();
//  thetaR_->sample();
//  theta_->sample();
  posteriorSticks();
}


template<class B, typename T>
void LrCluster<B,T>::posteriorSticks()
{
  boost::random::gamma_distribution<> gammaL((thetaL_->count()+alpha_)*0.5);
  boost::random::gamma_distribution<> gammaR((thetaR_->count()+alpha_)*0.5);
  stickL_ = gammaL(*pRndGen_);
  stickR_ = gammaR(*pRndGen_);
  T normalizer = log(stickL_ + stickR_);
  stickL_ = log(stickL_) - normalizer;
  stickR_ = log(stickR_) - normalizer;

#ifndef NDEBUG
  cout<<"sticks "<<stickL_<<", "<<stickR_
    <<" exp: "<<exp(stickL_)<<", "<<exp(stickR_)
    <<" log(sum) = "<<logsumexp(stickL_,stickR_)<<endl;
#endif
  assert(fabs(logsumexp(stickL_,stickR_)) <1e-6);
};

template<class B, typename T>
Matrix<T,1,2> LrCluster<B,T>::getSubclusterLogPdf(const Matrix<T,Dynamic,1>& x)
{ 
  assert(thetaL_);
  assert(thetaR_);
  Matrix<T,1,2> logPdf;
  logPdf(0) = thetaL_->logLikelihood(x) + stickL_;
  logPdf(1) = thetaR_->logLikelihood(x) + stickR_;
//  cout<<logPdf<<endl;
  return logPdf; //(logPdf.array()-logsumexp(logPdf(0),logPdf(1))).exp().matrix();
}

//template<>
//Matrix<double,1,2> LrCluster<NiwSphered,double>::getSubclusterLogPdf(
//    const Matrix<double,Dynamic,1>& x)
//{ 
////  cout<<"LrCluster<NiwSphered,double>::getSubclusterLogPdf"<<endl;
//  assert(thetaL_);
//  assert(thetaR_);
//  Matrix<double,1,2> logPdf;
//  logPdf(0) = thetaL_->logLikelihoodNorth(x) + stickL_;
//  logPdf(1) = thetaR_->logLikelihoodNorth(x) + stickR_;
////  cout<<logPdf<<endl;
//  return logPdf; //(logPdf.array()-logsumexp(logPdf(0),logPdf(1))).exp().matrix();
//}
//
//template<>
//Matrix<float,1,2> LrCluster<NiwSpheref,float>::getSubclusterLogPdf(
//    const Matrix<float,Dynamic,1>& x)
//{ 
//  assert(thetaL_);
//  assert(thetaR_);
//  Matrix<float,1,2> logPdf;
//  logPdf(0) = thetaL_->logLikelihoodNorth(x) + stickL_;
//  logPdf(1) = thetaR_->logLikelihoodNorth(x) + stickR_;
////  cout<<logPdf<<endl;
//  return logPdf; //(logPdf.array()-logsumexp(logPdf(0),logPdf(1))).exp().matrix();
//}

template<>
double LrCluster<NiwSphereFulld,double>::qRandomParamProposal() 
{ 
  return theta_->qRandomMuProposal() ;
}

//template<>
//double LrCluster<NiwSphereFulld,double>::qRandomSplitProposal() 
//{ 
//  return thetaL_->qRandomMuProposal() 
//    + thetaR_->qRandomMuProposal() ;
//}

template<class B, typename T>
T LrCluster<B,T>::dataLogLikelihoodMarginalized()
{
  return theta_->logPdfUnderPriorMarginalized();
};

template<class B, typename T>
bool LrCluster<B,T>::splittable()
{
  return splittable_;
};

template<class B, typename T>
void LrCluster<B,T>::updateAvgLogLikeData()
{
  if(thetaL_->count() == 0 || thetaR_->count() == 0)
  {
    splittable_ = true; // to allow reseting or merging this cluster
    updateAvgLogLikeData(99999.0);
  }else{
    T logJoint = 2.*log(alpha_) + boost::math::lgamma(alpha_) 
      - boost::math::lgamma(alpha_+ theta_->count());
    logJoint += boost::math::lgamma(thetaL_->count());
    logJoint += boost::math::lgamma(thetaR_->count());
    logJoint += thetaL_->logPdfUnderPriorMarginalized();
    logJoint += thetaR_->logPdfUnderPriorMarginalized();
    updateAvgLogLikeData(logJoint/count());
  }
}

template<class B, typename T>
void LrCluster<B,T>::updateAvgLogLikeData(T avgLogLikeData)
{
#ifndef NDEBUG
  cout<<"\x1b[35m LrCluster<B,T>::updateAvgLogLikeData ";
#endif
  int32_t W = static_cast<int32_t>(kernel_.size());
  avgLogLikeDataHist_.push_back(avgLogLikeData);
  if(static_cast<int32_t>(avgLogLikeDataHist_.size()) >= W)
  {
    // compute convolution to smooth wiggeling
    int32_t N = avgLogLikeDataHist_.size();
    T avgLogLikeDataNow =0.;
    for(int32_t i=0; i< W; ++i)
      avgLogLikeDataNow += kernel_(i)*avgLogLikeDataHist_[N+i-kernel_.size()];

    if(avgLogLikeDataNow - avgLogLikeData_ <= 1e-3)
    {
      // if same or decreased for more thant 2 times we are splittable
      if(nDecrease_ >= 2){
        //TODO
        splittable_ = true;
      }else
        ++ nDecrease_;
    }else{
      //    splittable_ = false;
    }
#ifndef NDEBUG
    cout << (avgLogLikeDataNow-avgLogLikeData_);
#endif
    avgLogLikeData_ = avgLogLikeDataNow;
  }else{
  } 
//  avgLogLikeData_ = avgLogLikeData;
#ifndef NDEBUG
    cout<< " -> "<< splittable_<<" history: "<<endl;
  for(int32_t i=0; i<  avgLogLikeDataHist_.size(); ++i)
    cout<<avgLogLikeDataHist_[i]<<" ";
  cout<<"\x1b[0m"<<endl;
#endif
};


//template<class B, typename T>
//void LrCluster<B,T>::dataLogLikelihoodMarginalizedMergedWith(
//    const shared_ptr<B >& other)
//{
//  return 
//};

//template<class B, typename T>
//void sampleParamUpper()
//{
//  theta_->
//};

//template<class B, typename T>
//void LrCluster<B,T>::sampleUpper()
//{
//  theta_->sample();
//};
//
//template<class B, typename T>
//void LrCluster<B,T>::sampleLR()
//{
//  thetaL_->sample();
//  thetaR_->sample();
//};

template<class B, typename T>
void LrCluster<B,T>::posteriorUpper()
{
//TODO: hopefully this is a good enough approximation
  theta_->fromMerge(*thetaL_,*thetaR_);
//  theta_->sample();
}

//template<class B, typename T>
//void LrCluster<B,T>::posteriorUpper(const Matrix<T,Dynamic,Dynamic>& x, 
//    const VectorXu& z, uint32_t k)
//{
//  //TODO need a way to tell that it should use all labels k and k+1
//  theta_->posterior(x,z,k);
//  assert(false);
//};

template<class B, typename T>
void LrCluster<B,T>::posteriorLR(const Matrix<T,Dynamic,Dynamic>& x, 
    const VectorXu& z, uint32_t k)
{
  thetaL_->posterior(x,z,k);
  thetaR_->posterior(x,z,k+1);
//  thetaL_->sample();
//  thetaR_->sample();
};

template<class B, typename T>
T LrCluster<B,T>::logLikelihood(const Matrix<T,Dynamic,1>& x) const
{
//  assert(false);// TODO may not be necessary
  return theta_->logLikelihood(x);
}

/* do not want that, since we only have linearization for a single cluster
template<>
double LrCluster<NiwSphered,double>::logLikelihood(
    const Matrix<double,Dynamic,1>& x) const
{
//  cout<<"LrCluster<NiwSphered,double>::logLikelihood"<<endl;
//  assert(false);// TODO may not be necessary
  return theta_->logLikelihoodNorth(x);
}

template<>
float LrCluster<NiwSpheref,float>::logLikelihood(
    const Matrix<float,Dynamic,1>& x) const
{
//  assert(false);// TODO may not be necessary
  return theta_->logLikelihoodNorth(x);
}
*/

template<class B, typename T>
void LrCluster<B,T>::print() const
{
  cout<<" ---------------- LrCluster ----------------- "<<endl;
  cout<<" upper "<<endl;
  theta_->print();
  cout<<" left "<<endl;
  thetaL_->print();
  cout<<" right "<<endl;
  thetaR_->print();
}

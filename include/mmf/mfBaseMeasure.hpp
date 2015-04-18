/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license.  See the license file LICENSE.
 */
 
#pragma once

#include <Eigen/Dense>

#include <dpMM/basemeasure.hpp>
#include <dpMM/niwSphereFull.hpp>
#include <mmf/mfPrior.hpp>
#include <mmf/mf.hpp>

template<typename T>
class MfBase : public BaseMeasure<T>
{
public:
  MfBase(const MfPrior<T>& mfPrior);
  MfBase(const MfBase<T>& mfBase);
  ~MfBase();

  virtual baseMeasureType getBaseMeasureType() const {return(MF_T); }

  virtual BaseMeasure<T>* copy();
  virtual MfBase<T>* copyNative();

  T logLikelihood(const Matrix<T,Dynamic,1>& x) const;
  T logLikelihood(const Matrix<T,Dynamic,Dynamic>& x, uint32_t i) const
    {return logLikelihood(x.col(i));};
  // assumes vector [N, sum(x), flatten(sum(outer(x,x)))]
  T logLikelihoodFromSS(const Matrix<T,Dynamic,1>& x) const;
  void posterior(const Matrix<T,Dynamic,Dynamic>& x, const VectorXu& z,
    uint32_t k);
  void posterior(const vector<Matrix<T,Dynamic,Dynamic> >&x, const
      VectorXu& z, uint32_t k);
  void posteriorFromSS(const vector<Matrix<T,Dynamic,1> >&x, const
      VectorXu& z, uint32_t k);
  void posteriorFromSS(const Matrix<T,Dynamic,1> &x);
  void sample();

  T logPdfUnderPrior() const;
  T logPdfUnderPriorMarginalized() const;
  T logPdfUnderPriorMarginalizedMerged(const shared_ptr<MfBase<T> >& other) const;

  T logPdfUnderPriorMarginalized(const Matrix<T,Dynamic,1>& x) {return 0.;};

  void print() const;
  virtual uint32_t getDim() const {return(uint32_t(normal_.D_));};

//  const Matrix<T,Dynamic,Dynamic>& scatter() const {return niw0_.scatter();};
//  const Matrix<T,Dynamic,1>& mean() const {return niw0_.mean();};
//  T count() const {return niw0_.count();};
////  T& count() {return niw0_.count_;};
//  const Matrix<T,Dynamic,1>& getMean() const {return normal_.mu_;};
//
//  const Matrix<T,Dynamic,Dynamic>& Sigma() const {return normal_.Sigma();};
private:
  MF<T> mf_;
  MfPrior<T> mf0_;

};

typedef MfBase<double> MfBased;
typedef MfBase<float> MfBasef;

// ---------------------------------------------------------


template<typename T>
MfBase<T>::MfBase(const MF<T>& mf)
  : mf_(mf)
{};

template<typename T>
MfBase<T>::~MfBase()
{};

template<typename T>
BaseMeasure<T>* MfBase<T>::copy()
{
  MfBase<T>* mfBase = new MfBase<T>(mf_);
//  niwSampled->normal_ = normal_;
  return mfBase;
};

template<typename T>
MfBase<T>* MfBase<T>::copyNative()
{
  MfBase<T>* mfBase = new MfBase<T>(mf_);
//  niwSampled->normal_ = normal_;
  return mfBase;
};

template<typename T>
T MfBase<T>::logLikelihood(const Matrix<T,Dynamic,1>& x) const
{
//  normal_.print();
  T logLike = mf_.logPdf(x);
//  cout<<x.transpose()<<" -> " <<logLike<<endl;
//  cout<<x.transpose()<<" -> " <<normal_.logPdfSlower(x)<<endl;
  return logLike;
};

// assumes vector [N, sum(x), flatten(sum(outer(x,x)))]
template<typename T>
T MfBase<T>::logLikelihoodFromSS(const Matrix<T,Dynamic,1>& x) const
{
//  normal_.print();
  uint32_t D = niw0_.D_;
  T count = x(0);
  Matrix<T,Dynamic,1> mean;
  if(count>0)
	  mean = x.middleRows(1,D)/count;
  else
	  mean = Matrix<T,Dynamic,1>::Zero(D); //this should not matter since everything gets multiplied by 0 counts

  //NOTE: Eigen::Map does not like const data, so this cast is needed to strip const data from input
  //alternatively the input could be changed to non-const, but this is cleaner from the outside
  T* datPtr = const_cast<T*>(&(x.data()[(D+1)]));
  Matrix<T,Dynamic,Dynamic> scatter =  Map<Matrix<T,Dynamic,Dynamic> >(datPtr,D,D);
  scatter -= (mean*mean.transpose())*count;

  T logLike = normal_.logPdf(scatter,mean,count);
  //  cout<<x.transpose()<<" -> " <<logLike<<endl;
  //  cout<<x.transpose()<<" -> " <<normal_.logPdfSlower(x)<<endl;
  return logLike;
};

template<typename T>
void MfBase<T>::posterior(const Matrix<T,Dynamic,Dynamic>& x,
    const VectorXu& z, uint32_t k)
{
  mf_ = mf0_.posterior(x,z,k).sample();
};

template<typename T>
void MfBase<T>::posterior(const vector<Matrix<T,Dynamic,Dynamic> >&x,
	const VectorXu& z, uint32_t k)
{
	mf_ = mf0_.posterior(x,z,k).sample();
}

template<typename T>
void MfBase<T>::posteriorFromSS(const vector<Matrix<T,Dynamic,1> > &x, const VectorXu& z, uint32_t k)
{
	normal_ = niw0_.posteriorFromSS(x,z,k).sample();
}

template<typename T>
void MfBase<T>::posteriorFromSS(const Matrix<T,Dynamic,1> &x)
{
	normal_ = niw0_.posteriorFromSS(x).sample();
}

template<typename T>
T MfBase<T>::logPdfUnderPrior() const
{
  return niw0_.logPdf(normal_);
};

template<typename T>
T MfBase<T>::logPdfUnderPriorMarginalized() const
{
  // evaluates log pdf of sufficient statistics stored within niw0_
//  niw0_.print();
  return niw0_.logPdfMarginalized();
};

template<typename T>
T MfBase<T>::logPdfUnderPriorMarginalizedMerged(
    const shared_ptr<MfBase<T> >& other) const
{
  return niw0_.logPdfUnderPriorMarginalizedMerged(other->niw0_);
};

template<typename T>
void MfBase<T>::sample()
{
  normal_ = niw0_.posterior().sample();
};

template<typename T>
void MfBase<T>::print() const
{
//  niw0_.posterior().print();
//  normal_.print();
};


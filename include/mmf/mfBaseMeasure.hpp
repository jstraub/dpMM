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
  MfBase(const MfPrior<T>& mf0);
  MfBase(const MfPrior<T>& mf0, const MF<T> mf);
  MfBase(const MfBase<T>& mfBase);
  ~MfBase();

  virtual baseMeasureType getBaseMeasureType() const {return(MF_T); }

  virtual BaseMeasure<T>* copy(); // TODO
  virtual MfBase<T>* copyNative(); // TODO

  T logLikelihood(const Matrix<T,Dynamic,1>& x) const;
  T logLikelihood(const Matrix<T,Dynamic,Dynamic>& x, uint32_t i) const
    {return logLikelihood(x.col(i));};
// TODO
// assumes vector [N, sum(x), flatten(sum(outer(x,x)))]
  T logLikelihoodFromSS(const Matrix<T,Dynamic,1>& x) const;

  void posterior(const Matrix<T,Dynamic,Dynamic>& x, const VectorXu& z,
    uint32_t k);
// TODO
//  void posterior(const vector<Matrix<T,Dynamic,Dynamic> >&x, const
//      VectorXu& z, uint32_t k);
// TODO
  void posteriorFromSS(const vector<Matrix<T,Dynamic,1> >&x, const
      VectorXu& z, uint32_t k);
//  void posteriorFromSS(const Matrix<T,Dynamic,1> &x);
//  void sample();

// TODO
  T logPdfUnderPrior() const;
//  T logPdfUnderPriorMarginalized() const;
//  T logPdfUnderPriorMarginalizedMerged(const shared_ptr<MfBase<T> >& other) const;
//
//  T logPdfUnderPriorMarginalized(const Matrix<T,Dynamic,1>& x) {return 0.;};
// TODO
  void print() const;
  virtual uint32_t getDim() const {return 3;};

//  const Matrix<T,Dynamic,Dynamic>& scatter() const {return mf0_.scatter();};
//  const Matrix<T,Dynamic,1>& mean() const {return mf0_.mean();};
//  T count() const {return mf0_.count();};
////  T& count() {return mf0_.count_;};
//  const Matrix<T,Dynamic,1>& getMean() const {return normal_.mu_;};
//
//  const Matrix<T,Dynamic,Dynamic>& Sigma() const {return normal_.Sigma();};
  MfPrior<T> mf0_;
  MF<T> mf_;

private:
};

typedef MfBase<double> MfBased;
typedef MfBase<float> MfBasef;

// ---------------------------------------------------------


template<typename T>
MfBase<T>::MfBase(const MfPrior<T>& mf0)
  :  mf0_(mf0), mf_(mf0.sample())
{
  mf0_.R_ = mf_.R();
};

template<typename T>
MfBase<T>::MfBase(const MfPrior<T>& mf0, const MF<T> mf)
  : mf0_(mf0), mf_(mf)
{};

template<typename T>
MfBase<T>::MfBase(const MfBase<T>& mfB)
  : mf0_(mfB.mf0_), mf_(mfB.mf_)
{};

template<typename T>
MfBase<T>::~MfBase()
{};

template<typename T>
BaseMeasure<T>* MfBase<T>::copy()
{
  MfBase<T>* mfBase = new MfBase<T>(mf0_);
  mfBase->mf_ = mf_;
//  niwSampled->normal_ = normal_;
  return mfBase;
};

template<typename T>
MfBase<T>* MfBase<T>::copyNative()
{
  MfBase<T>* mfBase = new MfBase<T>(mf0_);
  mfBase->mf_ = mf_;
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


template<typename T>
T MfBase<T>::logLikelihoodFromSS(const Matrix<T,Dynamic,1>& x) const
{
  uint32_t D = 3;
  T count = x(0);
  Matrix<T,Dynamic,1> mean = x.middleRows(1,D);
  Matrix<T,Dynamic,Dynamic> scatter = Matrix<T,Dynamic,Dynamic>::Map(&(x.data()[(D+1)]),D-1,D-1);
  // right now this does not support actual scatter!  it supports
  // weighted directional data though.  count is the weight
  assert(scatter(0,0)==0.);
  assert(scatter(1,0)==0.);
  assert(scatter(0,1)==0.);
  assert(scatter(1,1)==0.);
  return count*mf_.logPdf(mean);
};

template<typename T>
void MfBase<T>::posterior(const Matrix<T,Dynamic,Dynamic>& x,
    const VectorXu& z, uint32_t k)
{
  mf_ = mf0_.posteriorSample(x,z,k);
};

//template<typename T>
//void MfBase<T>::posterior(const vector<Matrix<T,Dynamic,Dynamic> >&x,
//	const VectorXu& z, uint32_t k)
//{
//  mf_ = mf0_.posteriorFromSSsample(x,z,k);
//}

template<typename T>
void MfBase<T>::posteriorFromSS(const vector<Matrix<T,Dynamic,1> > &x, const VectorXu& z, uint32_t k)
{
  mf_ = mf0_.posteriorFromSSsample(x,z,k);
  //TODO
//  uint32_t D = normalS_.D_;
//  Matrix<T,Dynamic,1> w(z.size()); 
//  w.setZero(z.size());
//  uint32_t N = 0;
//  for (int32_t i=0; i<z.size(); ++i)
//    if(z[i] == k)
//    {
//      w[i]=x[i](0); // counts
//      ++N;
//    }
//  if(N > 0)
//  {
//    Matrix<T,Dynamic,Dynamic> q(D,N); 
//    uint32_t j=0;
//    for (int32_t i=0; i<z.size(); ++i)
//      if(z[i] == k)
//      {
//        q.col(j++) = x[i].middleRows(1,D);
//      }
//    normalS_.setMean(karcherMeanWeighted<T>(normalS_.getMean(), q, w, 100));
//    //TODO: slight permutation here for mu to allow proper sampling
//    // TODO: wastefull since it computes stuff for normals that are not used in the 
//    // later computations
//    Matrix<T,Dynamic,Dynamic> x_mu = S_.Log_p_north(normalS_.getMean(),q);
//    Matrix<T,Dynamic,Dynamic> outer =  Matrix<T,Dynamic,Dynamic>::Zero(D-1,D-1);
//    j=0;
//    for (int32_t i=0; i<z.size(); ++i)
//      if(z[i] == k)
//      {
//        outer += w(i)* x_mu.col(j) * x_mu.col(j).transpose();
//		j++; 
//      }
//    normalS_.setSigma(iw0_.posterior(outer, w.sum()).sample());
//  
//  }else{
//    normalS_.setMean(S_.sampleUnif(normalS_.pRndGen_));
//    iw0_.resetSufficientStatistics();
//    normalS_.setSigma(iw0_.sample());
//  }
////  sample();
//  //  cout<<"Delta: \n"<<iw0_.posterior(x_mu,z,k).Delta_<<endl;
//  //  cout<<"Sigma: \n"<<normalS_.Sigma()<<endl;
////  cout<<"Sigma Eigs:"<<normalS_.Sigma().eigenvalues()<<endl;
//#ifndef NDEBUG
//  cout<<"NiwSphere<T>::posterior"<<endl
//    <<normalS_.getMean().transpose()<<endl
//    <<normalS_.Sigma()<<endl;
//#endif
//
//  // old
//	normal_ = mf0_.posteriorFromSS(x,z,k).sample();
}

//template<typename T>
//void MfBase<T>::posteriorFromSS(const Matrix<T,Dynamic,1> &x)
//{
//	normal_ = mf0_.posteriorFromSS(x).sample();
//}

template<typename T>
T MfBase<T>::logPdfUnderPrior() const
{
  return mf0_.logPdf(mf_);
};

//template<typename T>
//T MfBase<T>::logPdfUnderPriorMarginalized() const
//{
//  // evaluates log pdf of sufficient statistics stored within mf0_
////  mf0_.print();
//  return mf0_.logPdfMarginalized();
//};
//
//template<typename T>
//T MfBase<T>::logPdfUnderPriorMarginalizedMerged(
//    const shared_ptr<MfBase<T> >& other) const
//{
//  return mf0_.logPdfUnderPriorMarginalizedMerged(other->mf0_);
//};
//
//template<typename T>
//void MfBase<T>::sample()
//{
//  normal_ = mf0_.posterior().sample();
//};

template<typename T>
void MfBase<T>::print() const
{
  mf_.print();
//  mf0_.posterior().print();
//  normal_.print();
};


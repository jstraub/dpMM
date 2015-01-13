/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <iostream>
#include <fstream>

#include <stdint.h>
#include <vector>
#include <Eigen/Dense>

#include <boost/shared_ptr.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/gamma_distribution.hpp> // for gamma_distribution.
#include <boost/math/special_functions/gamma.hpp>

#include "cat.hpp"
#include "dpMM.hpp"
#include "lrCluster.hpp"
#include "dir.hpp"
#include "dirBaseMeasure.hpp"


using namespace Eigen;
using std::endl;
using std::cout;
using boost::shared_ptr;
//using boost::math::lgamma;

#undef CUDA

template <class B, typename T>
class DpSubclusterMM : public DpMM<T>
{
public:
  DpSubclusterMM(const T alpha, const shared_ptr<LrCluster<B,T> >& theta,
  virtual ~DpSubclusterMM();

  virtual void initialize(const shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx);

  void relinearize();
  void relinearize(uint32_t k);

  virtual void sampleLabels();
  virtual void sampleParameters_(); // only parameters
  virtual void sampleParameters() // also does splits and merges 
  { sampleParameters_(); 
//    cout<<"  counts= "<<counts(z_,2*K_).transpose()<<endl;
//    cout<<" ----------------------- proposing merges ----------------------"<<endl;
//    proposeMerges(); 
//    cout<<"  counts= "<<counts(z_,2*K_).transpose()<<endl;
//    cout<<" ----------------------- proposing splits ----------------------"<<endl;
//    proposeSplits();
//    cout<<"  counts= "<<counts(z_,2*K_).transpose()<<endl;

//    for(uint32_t k=0; k< getK(); ++k)
//      get(k)->print();
  };
  void proposeSplits();
  void proposeMerges();

  void proposeRandomSplits();

  virtual double logJoint();

  virtual const VectorXu & getLabels(){ return z_;};
  virtual uint32_t getK() const { return K_;};

  shared_ptr<LrCluster<B,T> > get(uint32_t k) { return thetas_[k];};

  MatrixXu mostLikelyInds(uint32_t n, Matrix<T,Dynamic,Dynamic>& logLikes);

  Matrix<T,Dynamic,1> getCounts();

  void dump(std::ofstream& fOutMeans, std::ofstream& fOutCovs);

protected:

  bool checkLabelMap(const VectorXi & labelMap, uint32_t Knew);

//  void sampleSuperClusters();

  uint32_t K_;
  uint32_t N_;
  uint32_t D_;
  T alpha_;
  shared_ptr<LrCluster<B,T> > theta0_;

  vector<shared_ptr<LrCluster<B,T> > > thetas_;

  shared_ptr<Matrix<T,Dynamic,Dynamic> > spq_; // points on the sphere
  shared_ptr<Matrix<T,Dynamic,Dynamic> > spxMu_; // points in T_muS
  shared_ptr<Matrix<T,Dynamic,Dynamic> > spx_; // points in T_northS
  VectorXu z_;
  VectorXu lr_;

  Matrix<T,Dynamic,1> sticks_;

  boost::mt19937 *pRndGen_;
  
#ifdef CUDA
  SamplerGpu<T>* sampler_;
#else 
  Sampler<T>* sampler_;
#endif
  Matrix<T,Dynamic,Dynamic> pdfsUpper_;
  Matrix<T,Dynamic,2> pdfsLR_;

  Matrix<T,Dynamic,Dynamic> logLikeZcls_; // K*K matrix with the probabilities of sampling a merge between two clusters

};


// --------------------------------- impl -------------------------------------
template <class B, typename T>
DpSubclusterMM<B,T>::DpSubclusterMM(const T alpha, 
    const shared_ptr<LrCluster<B,T> >& theta, uint32_t K0, boost::mt19937 *pRndGen)
  : K_(K0), alpha_(alpha), theta0_(theta), sticks_(K0), pRndGen_(pRndGen)
{
};

template <class B, typename T>
DpSubclusterMM<B,T>::~DpSubclusterMM()
{
  if (sampler_ != NULL) delete sampler_;
};

template <class B, typename T>
Matrix<T,Dynamic,1> DpSubclusterMM<B,T>::getCounts()
{
  Matrix<T,Dynamic,1> counts(K_);
  for(uint32_t k=0; k<K_; ++k)
    counts(k) = thetas_[k]->count();
  return counts;
};

template <class B, typename T>
void DpSubclusterMM<B,T>::initialize(
  const shared_ptr<Matrix<T,Dynamic,Dynamic> >& spq)
{
//  assert(K_ == 1);
  spq_ = spq;
  N_ = spq_->cols();
  D_ = spq_->rows();

#ifdef CUDA
  sampler_ = new SamplerGpu<T>(D_,K_,pRndGen_);
  assert(false); //TODO: need to support changing K on GPU sampler!
#else 
  sampler_ = new Sampler<T>(pRndGen_);
#endif
  z_.setZero(N_);
  lr_.setZero(N_);

  Matrix<T,Dynamic,1> alpha(2); alpha.setOnes(2);
//  alpha *= alpha_;
  Dir<Cat<T>, T> dirLr(alpha,pRndGen_);
  Cat<T> piLr = dirLr.sample(); 
  piLr.sample(lr_);
  if(K_ > 1)
  {
    alpha.setOnes(K_);
//    alpha *= alpha_;
    Dir<Cat<T>, T> dir(alpha,pRndGen_);
    Cat<T> pi = dir.sample(); 
    cout<<"init pi="<<pi.pdf().transpose()<<endl;
    pi.sample(z_);
  }
#pragma omp parallel for
  for(uint32_t i=0; i<z_.size(); ++i)
    z_(i) = z_(i)*2 + lr_(i);

//  cout<<z_.transpose()<<" #labels="<<z_.size()<<endl;
  cout<<"counts = "<<counts<T,uint32_t>(z_,2*K_).transpose()<<endl;

  for(uint32_t k=0; k<K_; ++k)
  {
    thetas_.push_back(shared_ptr<LrCluster<B,T> >(theta0_->copy()));
//    thetas_[k]->posterior(*spx_,z_,2*k);
  }
  pdfsLR_.resize(N_,2);

  uint32_t totalCount=0;
  for(uint32_t k=0; k<K_; ++k)
  {
    assert(thetas_[k] != NULL);
    relinearize(k);
    // compute posteriors and sample parameters
    thetas_[k]->posterior(*spx_,z_,2*k);
    cout<<thetas_[k]->countL()<<" + "<<thetas_[k]->countR()<<" = "<<thetas_[k]->count()<<endl;
    assert(thetas_[k]->countL()+thetas_[k]->countR() == thetas_[k]->count());
    totalCount += thetas_[k]->count();
  }
  cout<<totalCount<<endl;
  assert(totalCount == N_);
//  sticks_.setOnes(K_);
//  sticks_ *= 1./K_;
//  sampleParameters_();
//  dump(); 
};

template <class B, typename T>
void DpSubclusterMM<B,T>::dump(std::ofstream& fOutMeans, std::ofstream& fOutCovs)
{
  const uint32_t Kmax = 500;
  Matrix<T,Dynamic,Dynamic> means(D_,Kmax*3); means.setZero();
  for(uint32_t k=0; k<K_; ++k)
  {
    means.col(3*k) = thetas_[k]->getL()->getMean();
    means.col(3*k+1) = thetas_[k]->getR()->getMean();
    means.col(3*k+2) = thetas_[k]->getUpper()->getMean();
  }
  fOutMeans << means<<endl;

  uint32_t D = thetas_[0]->getL()->Sigma().cols();
  Matrix<T,Dynamic,Dynamic> covs(D,Kmax*3*D); covs.setZero();
  for(uint32_t k=0; k<K_; ++k)
  {
    covs.middleCols(3*k*D,D) = thetas_[k]->getL()->Sigma();
    covs.middleCols((3*k+1)*D,D) = thetas_[k]->getR()->Sigma();
    covs.middleCols((3*k+2)*D,D) = thetas_[k]->getUpper()->Sigma();
  }
  fOutCovs << covs<<endl;
}

template <>
void DpSubclusterMM<DirMultSampledd,double>::dump(std::ofstream& fOutMeans, std::ofstream& fOutCovs)
{
//  const uint32_t Kmax = 500;
//  Matrix<T,Dynamic,Dynamic> means(D_,Kmax*3); means.setZero();
//  for(uint32_t k=0; k<K_; ++k)
//  {
//    means.col(3*k) = thetas_[k]->getL()->getMean();
//    means.col(3*k+1) = thetas_[k]->getR()->getMean();
//    means.col(3*k+2) = thetas_[k]->getUpper()->getMean();
//  }
//  fOutMeans << means<<endl;
//
  for(uint32_t k=0; k<K_; ++k)
  {
    fOutMeans << thetas_[k]->getL()->pdf().transpose()<<endl;
    fOutMeans << thetas_[k]->getR()->pdf().transpose()<<endl;
    fOutMeans << thetas_[k]->getUpper()->pdf().transpose()<<endl;
  }
}

template <>
void DpSubclusterMM<DirMultSampledf,float>::dump(std::ofstream& fOutMeans, std::ofstream& fOutCovs)
{
  for(uint32_t k=0; k<K_; ++k)
  {
    fOutMeans << thetas_[k]->getL()->pdf().transpose()<<endl;
    fOutMeans << thetas_[k]->getR()->pdf().transpose()<<endl;
    fOutMeans << thetas_[k]->getUpper()->pdf().transpose()<<endl;
  }
}

template <class B, typename T>
void DpSubclusterMM<B,T>::relinearize()
{
  // just use spq as spq - no relinearization is performed
  if(spx_.get() != spq_.get())
    spx_ = spq_;  
}

template <class B, typename T>
void DpSubclusterMM<B,T>::relinearize(uint32_t k)
{
  // just use spq as spq - no relinearization is performed
  if(spx_.get() != spq_.get())
    spx_ = spq_;  
}

/* specialization for spherical data */
/* TODO: just for testing */
template <>
void DpSubclusterMM<NiwSphered, double>::relinearize(uint32_t k)
{
  if(spx_.get() == NULL)
  {
    spxMu_ = shared_ptr<Matrix<double,Dynamic,Dynamic> >(
        new Matrix<double,Dynamic,Dynamic>(D_,N_));
    spx_ = shared_ptr<Matrix<double,Dynamic,Dynamic> >(
        new Matrix<double,Dynamic,Dynamic>(D_-1,N_));
  }
  assert(spx_.get() != NULL);
  // find karcher means  for left and right
  Matrix<double,Dynamic,1> pL = thetas_[k]->getL()->getMean();
  pL = karcherMean<double>(pL,*spq_, *spxMu_, z_, 2*k,100); 
  Matrix<double,Dynamic,1> pR = thetas_[k]->getR()->getMean();
  pR = karcherMean<double>(pR,*spq_, *spxMu_, z_, 2*k+1,100);

  // set all means to the same value
  thetas_[k]->getL()->setMean(pL);
  thetas_[k]->getR()->setMean(pR);
  // linearize around the mean and rotate to tangent plane
  Sphere<double> S(D_);
  S.rotate_p2north(pL,*spxMu_,*spx_,z_,2*k);
  S.rotate_p2north(pR,*spxMu_,*spx_,z_,2*k+1);
}

template <>
void DpSubclusterMM<NiwSphereFulld, double>::relinearize(uint32_t k)
{
  if(spx_.get() == NULL)
  {
    spxMu_ = shared_ptr<Matrix<double,Dynamic,Dynamic> >(
        new Matrix<double,Dynamic,Dynamic>(D_,N_));
    spx_ = shared_ptr<Matrix<double,Dynamic,Dynamic> >(
        new Matrix<double,Dynamic,Dynamic>(D_-1,N_));
  }
  assert(spx_.get() != NULL);
  // find karcher means  for left and right
  Matrix<double,Dynamic,1> pL = thetas_[k]->getL()->getMean();
  pL = karcherMean<double>(pL,*spq_, *spxMu_, z_, 2*k,100); 
  Matrix<double,Dynamic,1> pR = thetas_[k]->getR()->getMean();
  pR = karcherMean<double>(pR,*spq_, *spxMu_, z_, 2*k+1,100);
  Matrix<double,Dynamic,1> p = thetas_[k]->getUpper()->getMean();
  p = karcherMean<double>(p,*spq_, *spxMu_, z_, k,100,2);
  
#ifndef NDEBUG
  cout<<"\x1b[34m ne linearization point for "<<k<<endl
    <<"U: "<<p.transpose()<<endl
    <<"L: "<<pL.transpose()<<endl
    <<"R: "<<pR.transpose()<<"\x1b[0m"<<endl;
#endif

//  Matrix<double,Dynamic,1> p = rotationFromAtoB<T>(pL,pR,
//      thetas_[k]->countR()/(thetas_[k]->))

  // set all means 
  thetas_[k]->getL()->setMean(pL);
  thetas_[k]->getR()->setMean(pR);
  thetas_[k]->getUpper()->setMean(p);

  thetas_[k]->getL()->setMeanKarch(pL); // pL
  thetas_[k]->getR()->setMeanKarch(pR); // pR
  thetas_[k]->getUpper()->setMeanKarch(p);
  // linearize around the mean and rotate to tangent plane
  Sphere<double> S(D_);
  S.Log_p(pL,*spq_,z_,2*k,*spxMu_);
  S.Log_p(pR,*spq_,z_,2*k+1,*spxMu_);
  S.rotate_p2north(pL,*spxMu_,*spx_,z_,2*k);
  S.rotate_p2north(pR,*spxMu_,*spx_,z_,2*k+1);
}

//template <>
//void DpSubclusterMM<NiwTangentd, double>::relinearize(uint32_t k)
//{
//  if(spx_.get() == NULL)
//  {
//    spxMu_ = shared_ptr<Matrix<double,Dynamic,Dynamic> >(
//        new Matrix<double,Dynamic,Dynamic>(D_,N_));
//    spx_ = shared_ptr<Matrix<double,Dynamic,Dynamic> >(
//        new Matrix<double,Dynamic,Dynamic>(D_-1,N_));
//  }
//  assert(spx_.get() != NULL);
//  //TODO: find the karcher mean of both L and R at the same time (from union of 
//  // point sets).
//  // find karcher means  for left and right
//  Matrix<double,Dynamic,1> p = thetas_[k]->getUpper()->getMean();
//  p = karcherMean<double>(p,*spq_, *spxMu_, z_, k,100,2); 
//  cout<<"p upper "<<p.transpose()<<endl;
//  // set all means to the same value
//  thetas_[k]->getL()->setMean(p);
//  thetas_[k]->getR()->setMean(p);
//  thetas_[k]->getUpper()->setMean(p);
//  // linearize around the mean and rotate to tangent plane
//  Sphere<double> S(D_);
//  S.Log_p(p,*spq_,z_,2*k,*spxMu_);
//  S.Log_p(p,*spq_,z_,2*k+1,*spxMu_);
//  S.rotate_p2north(p,*spxMu_,*spx_,z_,2*k);
//  S.rotate_p2north(p,*spxMu_,*spx_,z_,2*k+1);
//}

template <>
void DpSubclusterMM<NiwSphered, double>::relinearize()
{
  cout<<" ------ relinearizing ----------"<<endl;
  Sphere<double> S(D_);
  if(spx_.get() == NULL)
  {
    spxMu_ = shared_ptr<Matrix<double,Dynamic,Dynamic> >(
        new Matrix<double,Dynamic,Dynamic>(D_,N_));
    spx_ = shared_ptr<Matrix<double,Dynamic,Dynamic> >(
        new Matrix<double,Dynamic,Dynamic>(D_-1,N_));
  }
  Matrix<double,Dynamic,Dynamic> p0s(D_,2*K_);
  Matrix<double,Dynamic,Dynamic> ps(D_,2*K_);

  for(uint32_t k=0; k<K_; ++k)
  {
    p0s.col(2*k)   = thetas_[k]->getL()->getMean();
    p0s.col(2*k+1) = thetas_[k]->getR()->getMean();
  }
#ifndef NDEBUG
  cout<< p0s<<endl;
#endif
  ps = karcherMeanMultiple(p0s, *spq_, *spxMu_, z_, 2*K_, 100);
#ifndef NDEBUG
  for(uint32_t i=0; i<N_; ++i)
  {
    assert((ps.col(z_(i)).transpose() * spxMu_->col(i)).sum()<1e6);
  }
  cout<< ps<<endl;
#endif
  S.rotate_p2north(ps, *spxMu_, *spx_, z_, 2*K_);
  for(uint32_t k=0; k<K_; ++k)
  {
    thetas_[k]->getL()->setMean(ps.col(2*k));
    thetas_[k]->getR()->setMean(ps.col(2*k+1));
  }
}

//template <>
//void DpSubclusterMM<NiwTangentd, double>::relinearize()
//{
//  cout<<" ------ linearizing ----------"<<endl;
//  Sphere<double> S(D_);
//  if(spx_.get() == NULL)
//  {
//    spxMu_ = shared_ptr<Matrix<double,Dynamic,Dynamic> >(
//        new Matrix<double,Dynamic,Dynamic>(D_,N_));
//    spx_ = shared_ptr<Matrix<double,Dynamic,Dynamic> >(
//        new Matrix<double,Dynamic,Dynamic>(D_-1,N_));
//  }
////  Matrix<double,Dynamic,Dynamic> p0s(D_,2*K_);
//  Matrix<double,Dynamic,Dynamic> ps(D_,2*K_);
//
//  for(uint32_t k=0; k<K_; ++k)
//  {
//    ps.col(2*k)   = thetas_[k]->getL()->getMean();
//    ps.col(2*k+1) = thetas_[k]->getR()->getMean();
//  }
//#ifndef NDEBUG
//  cout<< ps<<endl;
//#endif
////  ps = karcherMeanMultiple(p0s, *spq_, *spxMu_, z_, 2*K_, 100);
////#ifndef NDEBUG
////  for(uint32_t i=0; i<N_; ++i)
////  {
////    assert((ps.col(z_(i)).transpose() * spxMu_->col(i)).sum()<1e6);
////  }
////  cout<< ps<<endl;
////#endif
//  S.rotate_p2north(ps, *spxMu_, *spx_, z_, 2*K_);
//  // only bring points to that 
////  for(uint32_t k=0; k<K_; ++k)
////  {
////    thetas_[k]->getL()->setMean(ps.col(2*k));
////    thetas_[k]->getR()->setMean(ps.col(2*k+1));
////  }
//}

template <>
void DpSubclusterMM<NiwSpheref, float>::relinearize()
{
  cout<<" ------ relinearizing ----------"<<endl;
  Sphere<float> S(D_);
  if(spx_.get() == NULL)
  {
    spxMu_ = shared_ptr<Matrix<float,Dynamic,Dynamic> >(
        new Matrix<float,Dynamic,Dynamic>(D_,N_));
    spx_ = shared_ptr<Matrix<float,Dynamic,Dynamic> >(
        new Matrix<float,Dynamic,Dynamic>(D_-1,N_));
 }

  Matrix<float,Dynamic,Dynamic> p0s(D_,2*K_);
  Matrix<float,Dynamic,Dynamic> ps(D_,2*K_);

  for(uint32_t k=0; k<K_; ++k)
  {
    p0s.col(2*k)   = thetas_[k]->getL()->getMean();
    p0s.col(2*k+1) = thetas_[k]->getR()->getMean();
  }
  ps = karcherMeanMultiple(p0s, *spq_, *spxMu_, z_, 2*K_, 100);
  S.rotate_p2north(ps,*spxMu_, *spx_, z_, 2*K_);
  for(uint32_t k=0; k<K_; ++k)
  {
    thetas_[k]->getL()->setMean(ps.col(2*k));
    thetas_[k]->getR()->setMean(ps.col(2*k+1));
  }
}

template <>
void DpSubclusterMM<NiwSphereFulld, double>::relinearize()
{
  cout<<" ------ relinearizing ----------"<<endl;
  Sphere<double> S(D_);
  if(spx_.get() == NULL)
  {
    spxMu_ = shared_ptr<Matrix<double,Dynamic,Dynamic> >(
        new Matrix<double,Dynamic,Dynamic>(D_,N_));
    spx_ = shared_ptr<Matrix<double,Dynamic,Dynamic> >(
        new Matrix<double,Dynamic,Dynamic>(D_-1,N_));
  }
  Matrix<double,Dynamic,Dynamic> p0s(D_,2*K_);
  Matrix<double,Dynamic,Dynamic> ps(D_,2*K_);

  for(uint32_t k=0; k<K_; ++k)
  {
    p0s.col(2*k)   = thetas_[k]->getL()->getMean();
    p0s.col(2*k+1) = thetas_[k]->getR()->getMean();
  }
  ps = karcherMeanMultiple(p0s, *spq_, *spxMu_, z_, 2*K_, 100);
#ifndef NDEBUG
//  cout<< p0s<<endl;
  for(uint32_t i=0; i<N_; ++i)
  {
    assert((ps.col(z_(i)).transpose() * spxMu_->col(i)).sum()<1e6);
  }
  cout<< ps<<endl;
#endif
//  S.rotate_p2north(ps, *spxMu_, *spx_, z_, 2*K_);
  for(uint32_t k=0; k<K_; ++k)
  {
    thetas_[k]->getL()->setMeanKarch(ps.col(2*k));
    thetas_[k]->getR()->setMeanKarch(ps.col(2*k+1));
  }
  // linearize around the sampled means!
//  S.Log_ps(p0s,*spq_,  z_, *spxMu_);
//  S.rotate_p2north(p0s, *spxMu_, *spx_, z_, 2*K_);
//  //TODO
  S.Log_ps(ps, *spq_,  z_, *spxMu_);
  S.rotate_p2north(ps, *spxMu_, *spx_, z_, 2*K_);

#ifndef NDEBUG
  cout<<"\x1b[34m linearizing around "<<endl
    <<p0s<<"\x1b[0m"<<endl;
#endif
}

template <class B, typename T>
void DpSubclusterMM<B,T>::sampleParameters_()
{
//  relinearize();

  uint32_t totalCount=0;
//#pragma omp parallel for
  for(uint32_t k=0; k<K_; ++k)
  {
    cout<<"count "<<k<<"/"<<K_<<": "<<thetas_[k]->count()<<endl;
    // compute posteriors and sample parameters
//    thetas_[k]->posterior(*spx_,z_,2*k);
    thetas_[k]->sample();
    // use counts to sample stick breaks
    if(thetas_[k]->count() > 0){
      boost::random::gamma_distribution<> gamma(thetas_[k]->count());
      sticks_(k) = gamma(*pRndGen_); 
    }else{
      sticks_(k) = 0.0;
    }
    totalCount += thetas_[k]->count();
  }
  assert(totalCount == N_);

//  for(uint32_t k=0; k<K_; ++k)
//    // to decide internally about splitability
//    thetas_[k]->updateAvgLogLikeData(); 

  T total = sticks_.sum();
  sticks_ = (sticks_.array().log() - log(total)).matrix();

  assert(fabs(logSumExp(sticks_)) <1e-6);

//  dump();
};

template <class B, typename T>
void DpSubclusterMM<B,T>::sampleLabels()
{
#ifndef NDEBUG
  cout<<"sticks = "<<sticks_.array().exp().matrix().transpose()
    <<" || "<<sticks_.array().exp().matrix().sum()<<endl;
#endif
  assert(fabs(logSumExp(sticks_)) <1e-6);

  // sample association to top upper clusters 
  // TODO: will have to take care of left right clusters ie left is even 
  // and right is odd cluster
  pdfsUpper_.resize(N_,K_);
#pragma omp parallel for
  for(uint32_t i=0; i<z_.size(); ++i)
  {
//    VectorXd logPdf_z = sticks_;
    pdfsUpper_.row(i) = sticks_;
    for(uint32_t k=0; k<K_; ++k)
      if(thetas_[k]->count() > 0)
      {
        // ! need to use q here and to relinearize since we are going through all
        // clusters and we only have linearization for a single one !
        pdfsUpper_(i,k) += thetas_[k]->logLikelihood(spq_->col(i));
//        if(pdfsUpper_(i,k) != pdfsUpper_(i,k)) // check for nan
//        {
//          cout<<pdfsUpper_(i,k)<<endl;
//          cout<<"sticks "<<sticks_.transpose()<<endl;
////          cout<<thetas_[k]->logLikelihood(spx_->col(i))<<endl;
//          thetas_[k]->getUpper()->print();
//          thetas_[k]->getL()->print();
//          thetas_[k]->getR()->print();
//          cout<<"---------------"<<endl;
//          thetas_[k].reset(); 
//        }
        assert(pdfsUpper_(i,k) == pdfsUpper_(i,k));
      }
    // make pdf sum to 1. and exponentiate
//    pdfsUpper_.row(i) =(logPdf_z.array()-logSumExp<T>(logPdf_z)).exp().matrix().transpose();
  }
  // sample all z_i
  sampler_->sampleDiscLogPdfUnNormalized(pdfsUpper_,z_);

//  MatrixXd A(N_,K_+1);
//  A<< pdfsUpper_, z_.cast<T>();
//  cout<<A<<endl;

  // aggregate the probability of sampling a merge between any two clusters
  logLikeZcls_.setZero(K_,K_);
  VectorXi toDelete(K_); toDelete.setOnes(K_); 
#pragma omp parallel for
  for(uint32_t i=0; i<z_.size(); ++i)
  {
    Matrix<T,1,Dynamic> logLikeZcls_i(K_);
    for(uint32_t k=0; k<K_; ++k)
      if(k!=z_(i))
        logLikeZcls_i(k) = logsumexp(pdfsUpper_(i,z_(i)),pdfsUpper_(i,k));
      else
        logLikeZcls_i(k) = pdfsUpper_(i,z_(i)); // on the diagonal
//    cout<<"------"<<endl;
//    cout<<pdfsUpper_.row(i)<<endl;
//    cout<<logLikeZcls_i<<endl;
    // to sample association to subclusters
    pdfsLR_.row(i) = thetas_[z_(i)]->getSubclusterLogPdf(spq_->col(i));
#pragma omp critical
    { 
      logLikeZcls_.row(z_(i)) += logLikeZcls_i;
      // compute counts to determine empty clusters
      --toDelete(z_(i));
    }
  }

  // sample all z_i
  sampler_->sampleDiscLogPdfUnNormalized(pdfsLR_,lr_);

//  cout<<" ----- 0 ------ "<<endl;
//  thetas_[0]->getL()->print();
//  cout<<" ----- 1 ------ "<<endl;
//  thetas_[0]->getR()->print();
//
//  MatrixXd C(N_,3);
//  C<< pdfsLR_, lr_.cast<T>();
//  cout<<C<<endl;
 
//TODO do this in GPU -> should be simple
  VectorXd logLikeZ(K_); // agregate probability of this subcluster sample
  logLikeZ.setZero();
  VectorXd logLikeData(K_); logLikeData.setZero();
//  VectorXd Nk(K_);  Nk.setZero();
//#pragma omp parallel for
  for(uint32_t i=0; i<z_.size(); ++i)
  {
//#pragma omp critical
    {
      logLikeZ(z_(i)) += pdfsLR_(i,lr_(i)) - logsumexp(pdfsLR_(i,0),pdfsLR_(i,1));
//      cout<<exp(pdfsLR_(i,lr_(i)) - logsumexp(pdfsLR_(i,0),pdfsLR_(i,1)))<<" "
//        <<exp(pdfsLR_(i,(lr_(i)+1)%2) - logsumexp(pdfsLR_(i,0),pdfsLR_(i,1)))
//        <<endl;
      logLikeData(z_(i)) += pdfsLR_(i,lr_(i));
//      ++ Nk(z_(i));
    }
  }
  for(uint32_t k=0; k<K_; ++k)
    thetas_[k]->logLikelihoodOfSubclusterSplit() = logLikeZ(k);
//  // to decide internally about splitability
//  for(uint32_t k=0; k<K_; ++k)
//    thetas_[k]->updateAvgLogLikeData(logLikeData(k)/Nk(k)); 
  // compute labelMap to remove empty clusters
#ifndef NDEBUG
  cout<<"toDelete: "<<toDelete.transpose()<<endl;
#endif
  VectorXi labelMap(2*K_); 
  for(uint32_t k=0; k<2*K_; ++k)
    labelMap(k) = k;
  int32_t nDelete = toDelete(0) == 1?1:0;
  for(uint32_t k=1; k<K_; ++k)
  {
    labelMap(2*k) = 2*(k-nDelete);
    labelMap(2*k+1) = 2*(k-nDelete) +1;
    if(toDelete(k) == 1)
    {
      labelMap(2*k) -= 2;
      labelMap(2*k+1) -= 2;
      nDelete ++;
    }
  }
  
  // delete clusters
  for(int32_t k=K_-1; k>=0; --k)
    if(toDelete(k) == 1)
      thetas_.erase(thetas_.begin() + k);
  // delete respective cols and rows from logLikeZcls_ 
  Matrix<T,Dynamic,Dynamic> logLikeZclsAfter(K_-nDelete, K_-nDelete);
  logLikeZclsAfter.setZero();
#ifndef NDEBUG
//  Matrix<T,Dynamic,Dynamic> logLikeZcls(K_, K_);
//  for(int32_t k=0; k<K_; ++k)
//    for(int32_t j=0; j<K_; ++j)
//    { 
//      logLikeZcls(k,j)= j+K_*k;
//    }
  cout<<"before"<<endl<<logLikeZcls_<<endl;
#endif
  int32_t lk=0;
  int32_t lj=0;
  for(int32_t k=0; k<static_cast<int32_t>(K_); ++k)
    if(toDelete(k) <1)
    {
      lj = 0;
      for(int32_t j=0; j<static_cast<int32_t>(K_); ++j)
        if(toDelete(j) <1)
          logLikeZclsAfter(j-lj,k-lk) = logLikeZcls_(j,k);  
        else
          lj ++;
//      cout<<"@"<<k<<endl<<logLikeZclsAfter<<endl;
    }else 
      lk ++;
  logLikeZcls_ = logLikeZclsAfter;
#ifndef NDEBUG
  cout<<"after"<<endl<<logLikeZcls_<<endl;
  cout<<" deleted : "<<nDelete<<" -> Knew = "<<K_ - nDelete<< " from "<<K_<<endl;
  cout<<"labelMap: "<<labelMap.transpose()<<endl;
  cout<<"counts before:   "<<counts<T,uint32_t>(z_,K_).transpose()<<endl;
  cout<<"sticks before: "<<sticks_.transpose()<<endl;
#endif
  Matrix<T,Dynamic,1> sticksNew(K_ - nDelete);
  for(uint32_t k=0; k<K_; ++k)
    sticksNew(labelMap(2*k)/2) = sticks_(k);
  K_ = K_ - nDelete;
  sticks_.resize(K_);
  sticks_ = sticksNew;
  // subtract of LR assignments since we resampled them
  for(uint32_t k=0; k<labelMap.size(); ++k)
    labelMap(k) -= labelMap(k)%2; 
#ifndef NDEBUG
  cout<<"sticks after:  "<<sticks_.transpose()<<endl;
  cout<<"Knew = "<<K_<<" #thetas = "<<thetas_.size()
    <<" sticks = "<<sticks_.transpose()<<endl;
//  assert(checkLabelMap(labelMap,2*K_));
  cout<<"labelMap after: "<<labelMap.transpose()<<endl;
//  cout<<"z:  "<<z_.transpose()<<endl;
//  cout<<"lr: "<<lr_.transpose()<<endl;
#endif

#pragma omp parallel for
  for(uint32_t i=0; i<z_.size(); ++i)
    z_(i) = labelMap(2*z_(i)) + lr_(i);
//  cout<<"z:  "<<z_.transpose()<<endl;
#ifndef NDEBUG
  cout<<"counts after:    "<<counts<T,uint32_t>(z_,2*K_).transpose()<<endl;
#endif

  // relinearize and compute posteriors
  relinearize();

//#pragma omp parallel for
//  for(uint32_t i=0; i<z_.size(); ++i)
//  {
//    
//  }

//#pragma omp parallel for
  for(uint32_t k=0; k<K_; ++k)
  {
    // compute posteriors and sample parameters
    thetas_[k]->posterior(*spx_,z_,2*k);
    // to decide internally about splitability
    thetas_[k]->updateAvgLogLikeData(); 
  }
};

template <class B, typename T>
void DpSubclusterMM<B,T>::proposeMerges()
{
  uint32_t nMerges = 0;
//  vector<int32_t> mergeWidth(K_,-1);
  VectorXi toDelete(K_); toDelete.setZero();
  VectorXi labelMap(2*K_);
  for(uint32_t jj=0; jj<2*K_; ++jj)
    labelMap(jj) = jj;
#ifndef NDEBUG
  cout<<"labelMap before: "<<labelMap.transpose()<<endl;
#endif
  for(uint32_t k=0; k<K_; ++k) if(thetas_[k]->splittable()) cout<<" cluster "<<k<<" splittable"<<endl; else cout<<" cluster "<<k<<" NOT splittable"<<endl;
  for(uint32_t k=0; k<K_; ++k) if(thetas_[k]->splittable())
    for(uint32_t j=k+1; j<K_; ++j) if(thetas_[j]->splittable())
      if(labelMap(2*k) != labelMap(2*k+1) && labelMap(2*j) != labelMap(2*j+1))
      {
//        cout<<k<<" "<<j<<endl;
        uint32_t Nk = thetas_[k]->count();
        uint32_t Nj = thetas_[j]->count();

        // construct the merge here and sample it then decide to accept or 
        // reject it
// TODO we sample inside the merge method since it may be a specific proposal 
        shared_ptr<LrCluster<B,T> > merged(thetas_[k]->merge(thetas_[j]));
//        merged->sample();
        // Hastings Ratio
        //TODO deterministic merge
        T logJointAfter = 
            merged->dataLogLikelihoodMarginalized() 
          + boost::math::lgamma(alpha_) + boost::math::lgamma(Nk+Nj)
          + boost::math::lgamma(0.5*alpha_+Nk)
          + boost::math::lgamma(0.5*alpha_+Nj);
        T logJointBefore = 
            thetas_[k]->dataLogLikelihoodMarginalized()
          + thetas_[j]->dataLogLikelihoodMarginalized()
          + log(alpha_) + 2*boost::math::lgamma(0.5*alpha_)
          + boost::math::lgamma(alpha_+Nk+Nj)
          + boost::math::lgamma(Nk)
          + boost::math::lgamma(Nj);
        T qBefore = thetas_[k]->qRandomParamProposal()
          + thetas_[j]->qRandomParamProposal();
        T qAfter = merged->qRandomParamProposal();
// ------------- old 
//        double HR =  boost::math::lgamma(alpha_) + boost::math::lgamma(Nk+Nj);
//        HR += -log(alpha_)-2*boost::math::lgamma(0.5*alpha_)
//          -boost::math::lgamma(alpha_+Nk+Nj);
//        HR += merged->dataLogLikelihoodMarginalized();
////        HR += thetas_[k]->dataLogLikelihoodMarginalizedMerged(thetas_[j]);
//        HR -= thetas_[k]->dataLogLikelihoodMarginalized();
//        HR -= thetas_[j]->dataLogLikelihoodMarginalized();
// -----------------
        //TODO inverse of local subcluster split
//       T logJointAfter = merged->dataLogLikelihoodMarginalized()
//          + boost::math::lgamma(Nk+Nj);
//       T logJointBefore = thetas_[k]->dataLogLikelihoodMarginalized()
//         + thetas_[j]->dataLogLikelihoodMarginalized()
//         + log(alpha_) + boost::math::lgamma(Nk)
//         + boost::math::lgamma(Nj);
//       T qBefore = logLikeZcls_(k,k) + logLikeZcls_(j,j)
//         + thetas_[k]->qRandomParamProposal()
//         + thetas_[j]->qRandomParamProposal();
//       T qAfter =  logLikeZcls_(j,k) + logLikeZcls_(k,j)
//         + merged->qRandomParamProposal();
// --------------------- old
//        double HR = -log(alpha_) - boost::math::lgamma(Nk);
//        HR += -boost::math::lgamma(Nj) + boost::math::lgamma(Nk+Nj);
//        HR += merged->dataLogLikelihoodMarginalized();
////        HR += thetas_[k]->dataLogLikelihoodMarginalizedMerged(thetas_[j]);
//        HR -= thetas_[k]->dataLogLikelihoodMarginalized();
//        HR -= thetas_[j]->dataLogLikelihoodMarginalized();
//        HR += logLikeZcls_(k,k) + logLikeZcls_(j,j);
//        HR -= logLikeZcls_(j,k) + logLikeZcls_(k,j);
// ------------------------

      T HR = logJointAfter - logJointBefore + qBefore - qAfter;
#ifndef NDEBUG
    cout<<"\x1b[32m p after  = "<< logJointAfter <<"\x1b[0m"<<endl;
    cout<<"\x1b[32m p before = "<< logJointBefore <<"\x1b[0m"<<endl;
    cout<<"\x1b[32m q after  = "<< qAfter <<"\x1b[0m"<<endl;
    cout<<"\x1b[32m q before = "<< qBefore <<"\x1b[0m"<<endl;
//      << "( "
//      << logLikeZcls_(k,k)<<" + "<<logLikeZcls_(j,j) << " - "
//      << logLikeZcls_(j,k)<<" - "<<logLikeZcls_(k,j)<<") "
//      <<"\x1b[0m"<<endl;

      cout<<"\x1b[32m log(alpha) = "<<log(alpha_)<<"\x1b[0m"<<endl;
      cout<<"\x1b[32m log(Gamma(Nk)) = "<<boost::math::lgamma(Nk)<<"\x1b[0m"<<endl;
      cout<<"\x1b[32m log(Gamma(Nj)) = "<<boost::math::lgamma(Nj)<<"\x1b[0m"<<endl;
      cout<<"\x1b[32m log(Gamma(N)) = "<<boost::math::lgamma(Nk+Nj)<<"\x1b[0m"<<endl;
      cout<<"\x1b[32m logLike k:  "
        <<thetas_[k]->logPdfUnderPriorMarginalized()<<"\x1b[0m"<<endl;
      cout<<"\x1b[32m logLike j: "
        <<thetas_[j]->logPdfUnderPriorMarginalized()<<"\x1b[0m"<<endl;
      cout<<"\x1b[32m logLike joint: "
        <<merged->dataLogLikelihoodMarginalized()<<"\x1b[0m"<<endl;
//        <<thetas_[k]->dataLogLikelihoodMarginalizedMerged(thetas_[j])<<"\x1b[0m"<<endl;
#endif

//        cout<<k<<" "<<j<<endl;
        boost::uniform_01<> unif_;
        if( HR > 0 || unif_(*pRndGen_) < exp(HR))
        { // accept merge
          labelMap(2*k) = 2*(k) + (labelMap(2*k)-2*k); // becomes left cluster
          labelMap(2*k+1) = 2*(k) + (labelMap(2*k)-2*k); // becomes left cluster
          labelMap(2*j) = 2*(k)+1 + (labelMap(2*k)-2*k); // becomes right cluster
          labelMap(2*j+1) = 2*(k)+1 + (labelMap(2*k)-2*k); // becomes right cluster
          toDelete(j) = 1;
          nMerges ++;
          for(uint32_t jj=j+1; jj<K_; ++jj)
            if((labelMap(2*jj) != labelMap(2*jj+1))
              || ((labelMap(2*jj) == labelMap(2*jj+1))
                  &&(labelMap(2*jj)/2 > static_cast<int32_t>(k))))
            {
              labelMap(2*jj) -= 2;
              labelMap(2*jj+1) -= 2;
            }
          
          // TODO may need to relinearize the merged one?
//          shared_ptr<LrCluster<B,T> > merged(thetas_[k]->merge(thetas_[j]));
          thetas_[k] = merged; // replace k with the merged one
          thetas_[k]->stickL() = sticks_(k);
          thetas_[k]->stickR() = sticks_(j);
          thetas_[k]->resetSplittable();
          sticks_(k) = logsumexp(sticks_(k),sticks_(j));
          sticks_(j) = sticks_(k); // so that it gets updated correctly below
          //thetas_[k]->sampleUpper(); // already done when merging
          cout<<"\x1b[31m merging "<<k<<" and "<<j<<" HR="<<HR<<"\x1b[0m"<<endl;
#ifndef NDEBUG
          cout<<" k:"<<k<<endl;
          thetas_[k]->getUpper()->print();
          cout<<" j:"<<j<<endl;
          thetas_[j]->getUpper()->print();
          cout<<" new left"<<endl;
          merged->getL()->print();
          cout<<" new right"<<endl;
          merged->getR()->print();
#endif
          break;
        }else{
          cout<<"NOT merging "<<k<<" and "<<j<<" HR="<<HR<<endl;
        }
      }
  if(nMerges > 0)
  {
    uint32_t Knew = K_ - nMerges;
#ifndef NDEBUG
    cout<<"labelMap before: "<<labelMap.transpose()<<endl;
    cout<<"K_before = "<<K_<<endl;
    cout<<"K_new = "<<Knew<<endl;
    assert(checkLabelMap(labelMap,2*Knew));
#endif
    // remove thetas, that are no needed anymore
    for(int32_t k=K_-1; k>=0; --k)
      if(toDelete(k)==1)
      { // second condition ensures that only the "j" cluster gets erased
#ifndef NDEBUG
        cout<<"Erasing "<< k<<endl;
#endif
        thetas_.erase(thetas_.begin() + k);
      }

    Matrix<T,Dynamic,1> sticksNew(Knew);
    for(uint32_t k=0; k<K_; ++k)
      sticksNew(labelMap(2*k)/2) = sticks_(k);
    sticks_.resize(Knew);
    //TODO: resize sticks and renormalize
    sticks_ = sticksNew;
    // map the labels to the new labels
#pragma omp parallel for
    for(uint32_t i=0; i<z_.size(); ++i)
      z_(i) = labelMap(z_(i)); //+ z_(i)%2;
    K_ = Knew;
  }

#ifndef NDEBUG
  cout<<"after merges sticks = "<<sticks_.array().exp().matrix().transpose()
    <<" ||="<<sticks_.array().exp().matrix().sum()<<endl;
#endif
};


template <class B, typename T>
void DpSubclusterMM<B,T>::proposeRandomSplits()
{
  // sample new LR labels for all datapoints completely at random
  Matrix<T,Dynamic,1> alpha(2); alpha.setOnes(2);
  alpha *= alpha_/2.;
  Dir<Cat<T>, T> dirLr(alpha,pRndGen_);
  Cat<T> piLr = dirLr.sample(); 
  VectorXu lr(N_);
  piLr.sample(lr);
  //
  for(uint32_t k=0; k<K_; ++k)
  {
    LrCluster<B,T>* thetaNew(theta0_->copy());
    // TODO!
  }
};

template <class B, typename T>
void DpSubclusterMM<B,T>::proposeSplits()
{

  for(uint32_t k=0; k<K_; ++k) if(thetas_[k]->splittable()) cout<<" cluster "<<k<<" splittable"<<endl; else cout<<" cluster "<<k<<" NOT splittable"<<endl;

  VectorXi doReset(K_); doReset.setOnes(K_); doReset *= -1;
  VectorXi doSplit(K_); doSplit.setOnes(K_); doSplit *= -1;
  for(uint32_t k=0; k<K_; ++k)
    if(thetas_[k]->splittable())
    {
      cout<<" testing split for "<<k<<endl;

      uint32_t Nk = thetas_[k]->count();
      uint32_t Nk_l = thetas_[k]->countL();
      uint32_t Nk_r = thetas_[k]->countR();
      assert(Nk == Nk_l+Nk_r);
      if(Nk_l == 0 || Nk_r == 0)
      {
        doReset(k) = 1;
        continue;
      }
//      double HR = log(alpha_) + boost::math::lgamma(Nk_l);
//      HR += boost::math::lgamma(Nk_r) - boost::math::lgamma(Nk);
//      HR += thetas_[k]->getL()->logPdfUnderPriorMarginalized();
//      HR += thetas_[k]->getR()->logPdfUnderPriorMarginalized();
//      HR -= thetas_[k]->getUpper()->logPdfUnderPriorMarginalized();
//      //TODO
////      HR -= thetas_[k]->logLikelihoodOfSubclusterSplit();
      
      T logJointAfter = log(alpha_) + boost::math::lgamma(Nk_l)
        + boost::math::lgamma(Nk_r) 
        + thetas_[k]->getL()->logPdfUnderPriorMarginalized()
        + thetas_[k]->getR()->logPdfUnderPriorMarginalized();
      T logJointBefore = boost::math::lgamma(Nk)
        + thetas_[k]->getUpper()->logPdfUnderPriorMarginalized();
      T qBefore = thetas_[k]->qRandomParamProposal();  //TODO
      T qAfter = 0.0;
// local split/merges
//      T qBefore = 0.0; 
//      T qAfter = thetas_[k]->logLikelihoodOfSubclusterSplit();
      

      T HR = logJointAfter - logJointBefore + qBefore - qAfter;
#ifndef NDEBUG
      cout<<"\x1b[32m p after  = "<< logJointAfter <<"\x1b[0m"<<endl;
      cout<<"\x1b[32m p before = "<< logJointBefore <<"\x1b[0m"<<endl;
      cout<<"\x1b[32m q before = "<< qBefore <<"\x1b[0m"<<endl;
      cout<<"\x1b[32m q after  = "<< qAfter <<"\x1b[0m"<<endl;

      cout<<"\x1b[32m log(alpha) = "<<log(alpha_)<<"\x1b[0m"<<endl;
      cout<<"\x1b[32m log(Gamma(Nkl)) = "<<boost::math::lgamma(Nk_l)<<"\x1b[0m"<<endl;
      cout<<"\x1b[32m log(Gamma(Nkr)) = "<<boost::math::lgamma(Nk_r)<<"\x1b[0m"<<endl;
      cout<<"\x1b[32m log(Gamma(Nk)) = "<<boost::math::lgamma(Nk)<<"\x1b[0m"<<endl;
      cout<<"\x1b[32m logLike Left:  "
        <<thetas_[k]->getL()->logPdfUnderPriorMarginalized()<<"\x1b[0m"<<endl;
      cout<<"\x1b[32m logLike Right: "
        <<thetas_[k]->getR()->logPdfUnderPriorMarginalized()<<"\x1b[0m"<<endl;
      cout<<"\x1b[32m logLike Upper: "
        <<thetas_[k]->getUpper()->logPdfUnderPriorMarginalized()<<"\x1b[0m"<<endl;
#endif

      boost::uniform_01<> unif_;
      if( HR > 0 || unif_(*pRndGen_) < exp(HR))
      { // accept split
        doSplit(k) = 1;
        cout<<"\x1b[31m splitting "<<k<<" HR = "<<HR<<"\x1b[0m"<<endl;
#ifndef NDEBUG
        thetas_[k]->getUpper()->print();
        thetas_[k]->getL()->print();
        thetas_[k]->getR()->print();
#endif
      }else{
        cout<<"NOT splitting "<<k<<" HR = "<<HR<<endl;
#ifndef NDEBUG
        thetas_[k]->getUpper()->print();
        thetas_[k]->getL()->print();
        thetas_[k]->getR()->print();
#endif
      }
    }
  if(doReset.sum() + K_ > 0)
  {
    // update labels so we can compute posteriors
#pragma omp parallel for
    for(uint32_t i=0; i<z_.size(); ++i)
      if(doReset(z_(i)/2) >= 0) 
      {
        boost::uniform_01<> unif;
        //TODO: should i assign the sublcusters randomly??
        z_(i) += -(z_(i)%2) + (unif(*pRndGen_)<0.5?0:1); //+ z_(i)%2;
        //cout<<(-(z_(i)%2))<<" "<< z_(i)<<endl;
      }
    //TODO: idealy only relinearize the clusters that need to be relinearized
//    relinearize();
    // update posteriors
    for(uint32_t k=0; k<K_; ++k)
      if(doReset(k) > 0)
      {
        cout<<"\x1b[31m resetting "<<k<<"\x1b[0m"<<endl; 
        // TODO: posterior also samples the stick lengths whereas posteriorLR and posterior Upper do not do that - what is right?
        // TODO: I dont think I need to compute posteriors here, since this is
        // next anyways after splits and merges
//        thetas_[k]->posterior(*spx_,z_,2*k);
//        thetas_[k]->posteriorLR(*spx_,z_,2*k);
//        thetas_[k]->posteriorUpper();
        relinearize(k);
        // compute posteriors and sample parameters
        thetas_[k]->posterior(*spx_,z_,2*k);
        thetas_[k]->resetSplittable();// dont consider cluster until splittable
        // to decide internally about splitability
//        thetas_[k]->updateAvgLogLikeData(); 

#ifndef NDEBUG
        cout<<"reseting "<<k<<endl;
        thetas_[k]->getUpper()->print();
        thetas_[k]->getL()->print();
        thetas_[k]->getR()->print();
#endif
      }
  }
  if(doSplit.sum() + K_ > 0)
  {
    uint32_t Knew = K_ + int32_t(doSplit.sum() + K_)/2;
#ifndef NDEBUG
    cout<<"doSplit: "<<doSplit.transpose()<<endl;
    cout<<"Knew:    "<<Knew<<endl;
    cout<<"before splits sticks = "<<sticks_.array().exp().matrix().transpose()
      <<" ||="<<sticks_.array().exp().matrix().sum()<<endl;
#endif
    Matrix<T,Dynamic,1> sticksNew(Knew);
//    cout<<sticksNew.transpose()<<endl;
//    cout<<K_<<endl;
    sticksNew.topRows(K_) = sticks_;
    sticks_.resize(Knew);
    sticks_ = sticksNew; //TODO make sure resizing leaves the old contents intact
    uint32_t nSplit = 0;
    VectorXi labelMap(2*Knew);
    for(uint32_t k=0; k<2*Knew; ++k)
      labelMap(k)=k;
    for(uint32_t k=0; k<K_; ++k)
      if(doSplit(k) > 0)
      {
        uint32_t j = K_ + nSplit++;
        labelMap(2*k) = 2*k; // left cluster
        labelMap(2*k+1) = 2*j; // right cluster

        labelMap(2*j) = 2*k+1; // left cluster
        labelMap(2*j+1) = 2*j+1; // left cluster
        
        assert(fabs(logsumexp(thetas_[k]->stickR(),thetas_[k]->stickL())) < 1e-6);

        sticks_(j) = sticks_(k) + thetas_[k]->stickR();
        sticks_(k) = sticks_(k) + thetas_[k]->stickL();
        
        // create new clusters from scratch with 50% 50% sticks
        thetas_[k].reset(
          new LrCluster<B,T>(thetas_[k]->getL(),thetas_[k]->alpha(),pRndGen_));
        thetas_.push_back(shared_ptr<LrCluster<B,T> >(
          new LrCluster<B,T>(thetas_[k]->getR(),thetas_[k]->alpha(),pRndGen_)));

//#ifndef NDEBUG
//        cout<<"split up "<<k<<endl;
//        thetas_[k]->getUpper()->print();
//        thetas_[k]->getL()->print();
//        thetas_[k]->getR()->print();
//        thetas_[j]->getUpper()->print();
//        thetas_[j]->getL()->print();
//        thetas_[j]->getR()->print();
//#endif
      }  
#ifndef NDEBUG
    cout<<"after splits sticks = "<<sticks_.array().exp().matrix().transpose()
      <<" ||="<<sticks_.array().exp().matrix().sum()<<endl;

    assert(checkLabelMap(labelMap,2*Knew));
#endif
    // update labels so we can compute posteriors
#pragma omp parallel for
    for(uint32_t i=0; i<z_.size(); ++i)
//      if(labelMap(z_(i)) >= 0) 
      if(doSplit(z_(i)/2) > 0)
      {
        boost::uniform_01<> unif;
        //TODO: should i assign the sublcusters randomly??
        z_(i) = labelMap(z_(i)) + (unif(*pRndGen_)<0.5?0:1); //+ z_(i)%2;
      }
    K_ = Knew;
#ifndef NDEBUG
    for(uint32_t k=0; k<K_; ++k)
    {
      cout<<"Checking cluster pointer "<<k<<endl;
      assert(thetas_[k].get());
    }
#endif
    //TODO: idealy only relinearize the clusters that need to be relinearized
//    relinearize();
    // update posteriors
    nSplit = 0;
    for(uint32_t k=0; k<doSplit.size(); ++k)
      if(doSplit(k) > 0)
      {
        uint32_t j = doSplit.size() + nSplit++;
        // relinearize k and j
        cout<<"relinearizing k and j after split"<<endl;
        relinearize(k);
        relinearize(j);
        cout<<"posterior of k and j after split"<<endl;
        // compute posteriors and sample parameters
        thetas_[k]->posterior(*spx_,z_,2*k);
        thetas_[j]->posterior(*spx_,z_,2*j); 
        // to decide internally about splitability
//        thetas_[k]->updateAvgLogLikeData(); 
      }
    // TODO: I dont think I need to compute posteriors here - will done 
    // jointly by sampleParameters after correctly relinearizing
//    nSplit = 0;
//    for(uint32_t k=0; k<K_; ++k)
//      if(doSplit(k) > 0)
//      {
//        uint32_t j = K_+nSplit++;
//        thetas_[k]->posterior(*spx_,z_,2*k); 
//        thetas_[j]->posterior(*spx_,z_,2*j); 
//
////        thetas_[k]->posteriorLR(*spx_,z_,2*k); 
////        thetas_[k]->posteriorUpper();
////        thetas_[j]->posteriorLR(*spx_,z_,2*j); 
////        thetas_[j]->posteriorUpper();
//
//#ifndef NDEBUG
//        cout<<"split up posterior "<<k<<endl;
//        thetas_[k]->getUpper()->print();
//        thetas_[k]->getL()->print();
//        thetas_[k]->getR()->print();
//        thetas_[j]->getUpper()->print();
//        thetas_[j]->getL()->print();
//        thetas_[j]->getR()->print();
//#endif
//      }
  }
//  if((doReset.sum() + K_ > 0)||(doSplit.sum() + K_ > 0))
//  {
//    //TODO: idealy only relinearize the clusters that need to be relinearized
//    relinearize();
//
//    for(uint32_t k=0; k<K_; ++k)
//      if(doReset(k) > 0 || doSplit(k) > 0)
//      {
//        // compute posteriors and sample parameters
//        thetas_[k]->posterior(*spx_,z_,2*k);
//        // to decide internally about splitability
//        thetas_[k]->updateAvgLogLikeData(); 
//      }
//  }
};

template <class B, typename T>
double DpSubclusterMM<B,T>::logJoint() 
{
  // p(\bz; \alpha) p(\bx; \Delta, \nu)
  double logJoint = K_*log(alpha_) + boost::math::lgamma(alpha_) 
    - boost::math::lgamma(alpha_+ z_.size());
  for(uint32_t k=0; k<K_; ++k)
    if (thetas_[k]->getUpper()->count() >0)
    {
      logJoint += boost::math::lgamma(thetas_[k]->getUpper()->count());
      logJoint += thetas_[k]->getUpper()->logPdfUnderPriorMarginalized();
    }
  return logJoint;
}

template<class B, typename T>
MatrixXu DpSubclusterMM<B,T>::mostLikelyInds(uint32_t n, 
  Matrix<T,Dynamic,Dynamic>& logLikes)
{
  MatrixXu inds = MatrixXu::Zero(n,K_);
  logLikes = Matrix<T,Dynamic,Dynamic>::Ones(n,K_);
  logLikes *= -99999.0;
  
#pragma omp parallel for 
  for (uint32_t k=0; k<K_; ++k)
    if (thetas_[k]->count() > 0)
  {
    for (uint32_t i=0; i<z_.size(); ++i)
      if(z_(i)/2 == k) // left right clusters are seen as one
      {
        T logLike = thetas_[z_[i]/2]->logLikelihood(spq_->col(i));
        for (uint32_t j=0; j<n; ++j)
          if(logLikes(j,k) < logLike)
          {
            for(uint32_t l=n-1; l>j; --l)
            {
              logLikes(l,k) = logLikes(l-1,k);
              inds(l,k) = inds(l-1,k);
            }
            logLikes(j,k) = logLike;
            inds(j,k) = i;
//            cout<<"after update "<<logLike<<endl;
//            Matrix<T,Dynamic,Dynamic> out(n,K_*2);
//            out<<logLikes.cast<T>(),inds.cast<T>();
//            cout<<out<<endl;
            break;
          }
      }
  } 
  cout<<"::mostLikelyInds: logLikes"<<endl;
  cout<<logLikes<<endl;
  cout<<"::mostLikelyInds: inds"<<endl;
  cout<<inds<<endl;
  return inds;
};

template<class B, typename T>
bool DpSubclusterMM<B,T>::checkLabelMap(const VectorXi & labelMap, uint32_t Knew)
{
    cout<<"DpSubclusterMM<B,T>::checkLabelMap: Knew: "<<Knew<<endl;
    cout<<"DpSubclusterMM<B,T>::checkLabelMap: labelMap: "
      <<labelMap.transpose()<<endl;
    VectorXi labelCount(Knew); labelCount.setZero();
    for(uint32_t k=2; k<labelMap.size(); ++k)
      if(labelMap(k) == labelMap(k-1) && labelMap(k-1) == labelMap(k-2)) 
        return false;
    for(uint32_t k=0; k<labelMap.size(); ++k)
      labelCount(labelMap(k)) ++;
    cout<<"DpSubclusterMM<B,T>::checkLabelMap: labelCount: "
      <<labelCount.transpose()<<endl;
    cout<< !(labelCount.array() == 0).any() <<endl;
    cout<< !(labelCount.array() > 2).any() <<endl;
    uint32_t count2= 0;
    for(uint32_t k=0; k<labelCount.size(); ++k)
      if(labelCount(k) == 2) count2 ++;

    cout<< count2<<" vs "<<(labelMap.size()-Knew)<<endl;
    assert( count2 == (labelMap.size()-Knew));
    return !(labelCount.array() == 0).any() && !(labelCount.array() > 2).any();
};

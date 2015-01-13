/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>                    
 * Licensed under the MIT license. See the license file LICENSE.                
 */

#include <iostream>
#include <stdint.h>
#include <vector>
#include <Eigen/Dense>

#include <boost/shared_ptr.hpp>

#include "dpMM.hpp"
#include "cat.hpp"
#include "dir.hpp"
#include "niw.hpp"
#include "sampler.hpp"
#include "clData.hpp"
#include "niwSphere.hpp"

using namespace Eigen;
using std::endl; using std::cout;
using boost::shared_ptr;

/* Dirichlet Mixture using ClData for data storage - should be FAST */
template <class H, class T>
class DirMMcld : public DpMM<T>
{
  uint32_t K_;
  Dir<Cat<T>,T> dir_;
  Cat<T> pi_;
#ifdef CUDA
  SamplerGpu<T>* sampler_;
#else 
  Sampler<T>* sampler_;
#endif
  Matrix<T,Dynamic,Dynamic> pdfs_;
//  Cat cat_;
  shared_ptr<BaseMeasure<T> > theta0_;
  vector<shared_ptr<BaseMeasure<T> > > thetas_;

  shared_ptr<ClData<T> > cld_;
  
public:
  DirMMcld(const Dir<Cat<T>,T>& alpha, const shared_ptr<BaseMeasure<T> >& theta);
  DirMMcld(const Dir<Cat<T>,T>& alpha, const vector<shared_ptr<BaseMeasure<T> > >& thetas);
  ~DirMMcld();

  void initialize(const shared_ptr<ClData<T> >& cld);
  void initialize(const Matrix<T,Dynamic,Dynamic>& x)
    {cout<<"not supported"<<endl; assert(false);};

  void sampleLabels();
  void sampleParameters();

  double logJoint();
  const VectorXu& z(){return (cld_->z());};
  const VectorXu& labels(){return (cld_->z());};
  const VectorXu& getLabels(){return (cld_->z());};
  const Matrix<T,Dynamic,1>& counts(){return cld_->counts();};
  const Matrix<T,Dynamic,Dynamic>& means(){return cld_->means();};

  virtual uint32_t getK() const { return K_;};

  Matrix<T,Dynamic,1> getCounts();

private: 
};

// --------------------------------------- impl -------------------------------

template <class H, class T>
DirMMcld<H,T>::DirMMcld(const Dir<Cat<T>,T>& alpha, 
    const shared_ptr<BaseMeasure<T> >& theta) 
  : K_(alpha.K_), dir_(alpha), pi_(dir_.sample()), theta0_(theta)
{};

template <class H, class T>
DirMMcld<H,T>::DirMMcld(const Dir<Cat<T>,T>& alpha, 
    const vector<shared_ptr<BaseMeasure<T> > >& thetas) :
  K_(alpha.K_), dir_(alpha), pi_(dir_.sample()), //cat_(dir_.sample()),
  thetas_(thetas)
{};

template <class H, class T>
DirMMcld<H,T>::~DirMMcld()
{
  if (sampler_ != NULL) delete sampler_;
};

template <class H, class T>
Matrix<T,Dynamic,1> DirMMcld<H,T>::getCounts()
{
  return counts();
};


//template <class H, class T>
//void DirMMcld<H,T>::initialize(const Matrix<T,Dynamic,Dynamic>& x)
//{
//
//};

template <class H, class T>
void DirMMcld<H,T>::initialize(const shared_ptr<ClData<T> >& cld)
{
  cld_ = cld;
  assert(cld_->K() == K_);

  cout<<"init"<<endl;
  // randomly init labels from prior
  cout<<"sample pi"<<endl;
  pi_ = dir_.sample(); 
  cout<<"init pi="<<pi_.pdf().transpose()<<endl;
#ifdef CUDA
  sampler_ = new SamplerGpu<T>(cld_->N(),K_,dir_.pRndGen_);
#else 
  sampler_ = new Sampler<T>(dir_.pRndGen_);
#endif
  //TODO: use sampler for this
  //pi_.sample(*(cld_->z()));
  pdfs_.setZero(cld_->N(),K_);
  for(uint32_t i=0; i<cld_->N(); ++i)
    pdfs_.row(i) = pi_.pdf();
  sampler_->sampleDiscPdf(pdfs_,(cld_->z()));
//  cout<<z->transpose()<<endl;
  assert((cld->z().array() < K_).all());
  // init the parameters
  if(thetas_.size() == 0)
  {
    cout<<"creating thetas"<<endl;
    for (uint32_t k=0; k<K_; ++k)
      thetas_.push_back(shared_ptr<BaseMeasure<T> >(theta0_->copy()));
  }
//  cld_->update(K_);
//  for(uint32_t k=0; k<K_; ++k)
//    thetas_[k]->posterior(cld_,k);
//  for (uint32_t k=0; k<K_; ++k)
//    thetas_[k].initialize(x_,z_);
};

template <class H, class T>
void DirMMcld<H,T>::sampleLabels()
{
  // obtain posterior categorical under labels
  pi_ = dir_.posterior(*(cld_->z())).sample();
//  cout<<pi_.pdf().transpose()<<endl;
  
  for(uint32_t i=0; i<cld_->N(); ++i)
  {
    //TODO: could buffer this better
    // compute categorical distribution over label z_i
    VectorXd logPdf_z = pi_.pdf().array().log();
    for(uint32_t k=0; k<K_; ++k)
    {
//      cout<<thetas_[k].logLikelihood(cld_->x()->col(i))<<" ";
      // TODO this does waste time since we are doing Log_p for all x on CPU
      logPdf_z[k] += thetas_[k]->logLikelihood(*(cld_->x()),i);
    }
//    cout<<endl;
    // make pdf sum to 1. and exponentiate
    pdfs_.row(i) = (logPdf_z.array()-logSumExp(logPdf_z)).exp().matrix().transpose();
//    cout<<pi_.pdf().transpose()<<endl;
//    cout<<pdf.transpose()<<" |.|="<<pdf.sum();
//    cout<<" z_i="<<z_[i]<<endl;
  }
  // sample z_i
  sampler_->sampleDiscPdf(pdfs_,cld_->z());
//    cout<<pdfs_<<endl;
//  cout<<" z="<<(*(cld_->z())).transpose()<<endl;
};

// specialization of the sampling function to run on GPU
template<>
void DirMMcld<NiwSphere<float>,float>::sampleLabels()
{
//  cout<<"sampling specialized to DirMMcld<NiwSphere,float>"<<endl;
  // obtain posterior categorical under labels
//  pi_ = dir_.posterior(*(cld_->z())).sample();
  pi_ = dir_.posteriorFromCounts(cld_->counts()).sample();
//  cout<<pi_.pdf().transpose()<<endl;
  vector<Matrix<float,Dynamic,Dynamic> > Sigmas(K_,
      Matrix<float,Dynamic,Dynamic>::Zero(cld_->D(),cld_->D()));
  Matrix<float,Dynamic,1> logNormalizers(K_);
  for(uint32_t k=0; k<K_; ++k)
  {
    Sigmas[k] = dynamic_cast<NiwSphere<float>* >(
        thetas_[k].get())->normalS_.Sigma();
    logNormalizers(k) = -0.5* dynamic_cast<NiwSphere<float>* >(
        thetas_[k].get())->normalS_.logDetSigma();
  }
//  cout<<logNormalizers.transpose()<<endl;
  cld_->sampleGMMpdf(pi_.pdf(), Sigmas, logNormalizers, sampler_);
};

template<>
void DirMMcld<NiwSphere<double>,double>::sampleLabels()
{
//  cout<<"sampling specialized to DirMMcld<NiwSphere,double>"<<endl;
  // obtain posterior categorical under labels
//  pi_ = dir_.posterior(*(cld_->z())).sample();
  pi_ = dir_.posteriorFromCounts(cld_->counts()).sample();
//  cout<<pi_.pdf().transpose()<<endl;
  vector<Matrix<double,Dynamic,Dynamic> > Sigmas(K_,
      Matrix<double,Dynamic,Dynamic>::Zero(cld_->D(),cld_->D()));
  Matrix<double,Dynamic,1> logNormalizers(K_);
  for(uint32_t k=0; k<K_; ++k)
  {
    Sigmas[k] = dynamic_cast<NiwSphere<double>* >(
        thetas_[k].get())->normalS_.Sigma();
    logNormalizers(k) = -0.5* dynamic_cast<NiwSphere<double>* >(
        thetas_[k].get())->normalS_.logDetSigma();
  }
//  cout<<logNormalizers.transpose()<<endl;

  cld_->sampleGMMpdf(pi_.pdf(), Sigmas, logNormalizers, sampler_);
};


template <class H, class T>
void DirMMcld<H,T>::sampleParameters()
{
  cld_->update(K_);
  for(uint32_t k=0; k<K_; ++k)
    thetas_[k]->posterior(cld_,k);
};

template <class H, class T>
double DirMMcld<H,T>::logJoint()
{
  double logJoint = dir_.logPdf(pi_);
  for (uint32_t k=0; k<K_; ++k)
    logJoint += thetas_[k]->logPdfUnderPrior();
  for (uint32_t i=0; i<cld_->N(); ++i)
    logJoint += thetas_[(cld_->z())(i)]->logLikelihood(*(cld_->x()),i);
  return logJoint;
};


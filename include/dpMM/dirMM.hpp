/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */
#pragma once
#include <iostream>
#include <stdint.h>
#include <vector>
#include <Eigen/Dense>

#include <boost/shared_ptr.hpp>

#include <dpMM/dpMM.hpp>
#include <dpMM/cat.hpp>
#include <dpMM/dir.hpp>
#include <dpMM/niw.hpp>
#include <dpMM/sampler.hpp>
#include <dpMM/basemeasure.hpp>

using namespace Eigen;
using std::endl; using std::cout;
using boost::shared_ptr;

template<typename T>
class DirMM : public DpMM<T>
{
public:
  DirMM(const Dir<Cat<T>, T>& alpha, const shared_ptr<BaseMeasure<T> >&
      theta, uint32_t K0);
  DirMM(const Dir<Cat<T>, T>& alpha, const
      vector<shared_ptr<BaseMeasure<T> > >& thetas);
  DirMM(const DirMM<T>& dirMM);
  virtual ~DirMM();

  virtual void initialize(const Matrix<T,Dynamic,Dynamic>& x);
  virtual void initialize(const shared_ptr<ClGMMData<T> >& cld)
    {cout<<"not supported"<<endl; assert(false);};

  virtual void sampleLabels();
  virtual void sampleParameters();
  virtual void sampleFromPrior();

  virtual T logJoint();
  virtual const Matrix<T,Dynamic,Dynamic>& x() const {return x_;};
  virtual const VectorXu& labels() {return z_;};
  virtual const VectorXu& getLabels() {return z_;};
  virtual void setLabels(const VectorXu& z){z_ = z;};
  virtual uint32_t getK() const { return K_;};
  virtual const shared_ptr<BaseMeasure<T> >& getTheta(uint32_t k) const
    { assert(k<K_); return this->thetas_[k];};
  virtual const vector<shared_ptr<BaseMeasure<T> > >& getThetas() const
    { return this->thetas_;};
  virtual const shared_ptr<BaseMeasure<T> >& getTheta0() const
    { return this->theta0_;};

  virtual const Dir<Cat<T>, T>& Alpha() const { return dir_;}; 
  virtual const Cat<T>& Pi() const { return pi_;}; 

//  virtual MatrixXu mostLikelyInds(uint32_t n);
  virtual MatrixXu mostLikelyInds(uint32_t n, Matrix<T,Dynamic,Dynamic>& logLikes);

  Matrix<T,Dynamic,1> getCounts();

  bool isInit() const { return sampler_!= NULL;};

protected: 
  uint32_t K0_;  // that is the number of clusters that are initialized with data at the beginning (K0_ <= K_)
  uint32_t K_;
  Dir<Cat<T>, T> dir_;
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

  Matrix<T,Dynamic,Dynamic> x_;
  VectorXu z_;
};

// --------------------------------------- impl -------------------------------


template<typename T>
DirMM<T>::DirMM(const Dir<Cat<T>,T>& alpha, const
    shared_ptr<BaseMeasure<T> >& theta, uint32_t K0) :
  K0_(K0), K_(alpha.K_), dir_(alpha), pi_(dir_.sample()), 
  sampler_(NULL),
  theta0_(theta)
{};


template<typename T>
DirMM<T>::DirMM(const Dir<Cat<T>,T>& alpha, 
    const vector<shared_ptr<BaseMeasure<T> > >& thetas) :
  K_(alpha.K_), dir_(alpha), pi_(dir_.sample()), 
  sampler_(NULL), thetas_(thetas),
  theta0_(
      shared_ptr<BaseMeasure<T> >(thetas[0]->copy()))
{
//  cout<<thetas_.size()<<endl;
//  for(uint32_t k=0; k<thetas_.size(); ++k)
//    cout<< "   "<<thetas_[k].get()<<endl;
//  cout<<thetas_[0].get()<<endl;
//  cout<<thetas_[0]->copy()<<endl;
//
//  theta0_ = shared_ptr<BaseMeasure<T> >(thetas_[0]->copy());
};

template<typename T>
DirMM<T>::DirMM(const DirMM<T>& dirMM) 
  : K_(dirMM.getK()), dir_(dirMM.Alpha()), pi_(dirMM.Pi()), 
  sampler_(NULL), 
  theta0_(shared_ptr<BaseMeasure<T> >(
        dirMM.getTheta0()->copy()))
{
  for(uint32_t k=0; k<  dirMM.getK(); ++k)
  {
    thetas_.push_back(shared_ptr<BaseMeasure<T> >(
          dirMM.getTheta(k)->copy())); 
  }
  if(dirMM.isInit())
  {
    x_ = dirMM.x();
    // bad boy
    z_ = const_cast<DirMM<T>* >(&dirMM)->labels();
    pdfs_.setZero(x_.cols(),K_);
#ifdef CUDA
    sampler_ = new SamplerGpu<T>(x_.cols(),K_,dir_.pRndGen_);
#else 
    sampler_ = new Sampler<T>(dir_.pRndGen_);
#endif
  }
};


template<typename T>
DirMM<T>::~DirMM()
{
  if (sampler_ != NULL) delete sampler_;
};

template <typename T>
Matrix<T,Dynamic,1> DirMM<T>::getCounts()
{
  return counts<T,uint32_t>(z_,K_);
};


template<typename T>
void DirMM<T>::initialize(const Matrix<T,Dynamic,Dynamic>& x)
{
//  cout<<"init"<<endl;
  x_ = x;
  // randomly init labels from prior
  z_.setZero(x.cols());
//  cout<<"sample pi"<<endl;
  pi_ = dir_.sample(); 
  if (K0_ < K_)
  {
    Matrix<T,Dynamic,1> pdf = pi_.pdf();
    pdf.bottomRows(K_-K0_).setZero();
    pdf = pdf / pdf.sum(); // renormalize
    pi_.pdf(pdf);
  } 
//  cout<<"init pi="<<pi_.pdf().transpose()<<endl;
  pi_.sample(z_);

  pdfs_.setZero(x.cols(),K_);
#ifdef CUDA
  sampler_ = new SamplerGpu<T>(x.cols(),K_,dir_.pRndGen_);
#else 
  sampler_ = new Sampler<T>(dir_.pRndGen_);
#endif

  // init the parameters
//  if(thetas_.size() == 0)
//  {
  thetas_.clear(); // destrey eny previous thetas
//  cout<<"creating thetas"<<endl;
  for (uint32_t k=0; k<K_; ++k)
    thetas_.push_back(shared_ptr<BaseMeasure<T> >(theta0_->copy()));
//  }
//#pragma omp parallel for
//  for(uint32_t k=0; k<K_; ++k)
//    thetas_[k]->posterior(x_,z_,k);
//  for (uint32_t k=0; k<K_; ++k)
//    thetas_[k].initialize(x_,z_);
};

template<typename T>
void DirMM<T>::sampleLabels()
{
  // obtain posterior categorical under labels
  pi_ = dir_.posterior(z_).sample();
//  cout<<pi_.pdf().transpose()<<endl;
  
#pragma omp parallel for
  for(uint32_t i=0; i<z_.size(); ++i)
  {
    //TODO: could buffer this better
    // compute categorical distribution over label z_i
    VectorXd logPdf_z = pi_.pdf().array().log();
    for(uint32_t k=0; k<K_; ++k)
    {
//      cout<<thetas_[k].logLikelihood(x_.col(i))<<" ";
      logPdf_z[k] += thetas_[k]->logLikelihood(x_.col(i));
    }
//    cout<<endl;
    // make pdf sum to 1. and exponentiate
    pdfs_.row(i) = (logPdf_z.array()-logSumExp(logPdf_z)).exp().matrix().transpose();
//    cout<<pi_.pdf().transpose()<<endl;
//    cout<<pdfs_.row(i)<<" |.|="<<pdfs_.row(i).sum()<<endl;;
//    cout<<" z_i="<<z_[i]<<endl;
  }
  // sample z_i
  sampler_->sampleDiscPdf(pdfs_,z_);
};


template<typename T>
void DirMM<T>::sampleParameters()
{
  Matrix<T,Dynamic,1> Ns = getCounts();
//#pragma omp parallel for 
  for(uint32_t k=0; k<K_; ++k)
  {
//    cout<<"k:"<<k<<" "<<Ns(k)<<" ";
    thetas_[k]->posterior(x_,z_,k);
//    thetas_[k]->print();
  }
};

template<typename T>
void DirMM<T>::sampleFromPrior()
{
//  Matrix<T,Dynamic,1> Ns = getCounts();
//#pragma omp parallel for 
//
// simulate sampling from prior by not giving the posteriors any data
  Matrix<T,Dynamic,Dynamic> x = Matrix<T,Dynamic,Dynamic>::Zero(1,1);
  VectorXu z = VectorXu::Ones(1)*(K_+1);
  for(uint32_t k=0; k<K_; ++k)
  {
//    cout<<"k:"<<k<<" "<<Ns(k)<<" ";
    thetas_[k]->posterior(x,z,k);
//    thetas_[k]->print();
  }
};


template<typename T>
T DirMM<T>::logJoint()
{
  T logJoint = dir_.logPdf(pi_);
  cout<<"  [logJoint="<<logJoint<<" -> ";
#pragma omp parallel for reduction(+:logJoint)  
  for (uint32_t k=0; k<K_; ++k)
    logJoint = logJoint + thetas_[k]->logPdfUnderPrior();
  cout<<" "<<logJoint<<" -> ";
#pragma omp parallel for reduction(+:logJoint)  
  for (uint32_t i=0; i<z_.size(); ++i)
    logJoint = logJoint + thetas_[z_[i]]->logLikelihood(x_.col(i));
  cout<<" "<<logJoint<<"]"<<endl;
  return logJoint;
};


template<typename T>
MatrixXu DirMM<T>::mostLikelyInds(uint32_t n, Matrix<T,Dynamic,Dynamic>& logLikes)
{
  MatrixXu inds = MatrixXu::Zero(n,K_);
  logLikes = Matrix<T,Dynamic,Dynamic>::Ones(n,K_);
  
#pragma omp parallel for 
  for (uint32_t k=0; k<K_; ++k)
  {
    for (uint32_t i=0; i<z_.size(); ++i)
      if(z_(i) == k)
      {
        T logLike = thetas_[z_[i]]->logLikelihood(x_.col(i));
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

//template<class T>
//T DirMM<T>::avgIntraClusterDeviation()
//{
//  Matrix<T,Dynamic,1> deviates(K_);
//  deviates.setZero(K_);
//#pragma omp parallel for 
//  for (uint32_t k=0; k<K_; ++k)
//  {
//    T N_k = 0.0;
//    for (uint32_t i=0; i<N_; ++i)
//      if(z_(i) == k)
//      {
//        T dot = thetas_[k]->transpose()*spx_->col(i);
//        deviates(k) += acos(min(1.0,max(-1.0,dot)));
//        N_k ++;
//      }
//    if(N_k > 0.0) deviates(k) /= N_k;
//  }
//  return deviates.sum()/static_cast<T>(K_);
//}

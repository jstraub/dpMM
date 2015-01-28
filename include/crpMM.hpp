/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */
#pragma once
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
#include "basemeasure.hpp"

using namespace Eigen;
using std::endl; using std::cout;
using boost::shared_ptr;

  
  // algorithm 2 of Neal 
  // http://www.stat.columbia.edu/npbayes/papers/neal_sampling.pdf
template<typename T>
class CrpMM : public DpMM<T>
{
public:
  CrpMM(const T alpha, const shared_ptr<BaseMeasure<T> >& theta, uint32_t K0,
      boost::mt19937* pRndGen);
  virtual ~CrpMM()
  {};

  virtual void initialize(const shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx);
  virtual void initialize(const Matrix<T,Dynamic,Dynamic>& x)
  {this->spx_=shared_ptr<Matrix<T,Dynamic,Dynamic> >(new
      Matrix<T,Dynamic,Dynamic>(x)); initialize(this->spx_);};
  virtual void sampleLabels();
  virtual void sampleParameters();
  virtual void proposeSplits() {};
  virtual void proposeMerges() {};
  // call this right before sampleParameters()
  virtual void resampleFromBase(uint32_t Kmax) {};
  virtual const VectorXu & getLabels(){return z_;};
  virtual const VectorXu & labels(){return z_;};
  virtual Matrix<T,Dynamic,1> getCounts();
  virtual uint32_t getK() const { return K_;};
  virtual double logJoint() { return 0.0;}; 

//  virtual MatrixXu mostLikelyInds(uint32_t n) 
//{ return MatrixXu::Zero(n,1);};
  virtual MatrixXu mostLikelyInds(uint32_t n, Matrix<T,Dynamic,Dynamic>& logLikes)
  { return MatrixXu::Zero(n,1);};

private:
  void removeEmptyClusters();

  uint32_t K_;
  T alpha_; // dp concentration paramter

  shared_ptr<BaseMeasure<T> > theta0_;
  vector<shared_ptr<BaseMeasure<T> > > thetas_;

  VectorXu z_;

  boost::mt19937* pRndGen_;
};

// ------------------------- impl -----------------------------------
template<typename T>
CrpMM<T>::CrpMM(const T alpha, const shared_ptr<BaseMeasure<T> >& theta,
    uint32_t K0,  boost::mt19937* pRndGen)
  : K_(K0), alpha_(alpha), theta0_(theta), pRndGen_(pRndGen)
{};

template <typename T>
Matrix<T,Dynamic,1> CrpMM<T>::getCounts()
{
  return counts<T,uint32_t>(z_,K_);
};

template<typename T>
void CrpMM<T>::initialize(const shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx)
{
  cout<<"init"<<endl;
  this->spx_ = spx;
  // randomly init labels from prior
  z_.setZero(spx->cols());

  Matrix<T,Dynamic,1> alpha = (alpha_/K_)*Matrix<T,Dynamic,1>::Ones(K_); 
  Cat<T> pi = Dir<Cat<T>,T>(alpha,pRndGen_).sample(); 
  for(uint32_t i=0; i<z_.size(); ++i)
  {
    z_(i) = pi.sample();
  }

  // init the parameters
  if(thetas_.size() == 0)
  {
    cout<<"creating thetas"<<endl;
    for (uint32_t k=0; k<K_; ++k)
      thetas_.push_back(shared_ptr<BaseMeasure<T> >(theta0_->copy()));
  }
};

template<typename T>
void CrpMM<T>::sampleLabels()
{
  Matrix<T,Dynamic,1> Nk(K_+1);
  Nk.topRows(K_) = counts<T,uint32_t>(z_,K_);
  Nk(K_) = alpha_;
//  cout<<"Nk="<<Nk.transpose()<<" alpha="<<alpha_<<endl;
  for(uint32_t i=0; i<z_.size(); ++i)
  {
    Nk(z_(i)) --; // take the data-point out of that cluster
    Matrix<T,Dynamic,1> pdf = Nk.array().log().matrix();
    pdf = pdf.array() - log(Nk.sum());
//    cout<<pdf.transpose()<< " -> ";
    // add the log posteriors of the individual clusters
    for(uint32_t k=0; k<K_; ++k)
      pdf(k) += thetas_[k]->logLikelihood(this->spx_->col(i)); 
    // add the log likelihood of data under the hyperparameters
    pdf(K_) += theta0_->logPdfUnderPriorMarginalized(this->spx_->col(i));
//    cout<<pdf.transpose()<<" exp: ";
    // normalize and exponentiate pdf
    pdf = (pdf.array() - logSumExp(pdf)).exp().matrix();
//    cout<<pdf.transpose()<<endl;
    z_(i) = Catd(pdf,pRndGen_).sample();
    Nk(z_(i)) ++; // add data-point into new cluster

    // add a new cluster from the base measure
    if(z_(i)==K_){
      thetas_.push_back(shared_ptr<BaseMeasure<T> >(theta0_->copy()));
      // TODO might want to add z_i to the SS of the new cluster?
      thetas_[z_(i)]->posterior(*this->spx_,z_,K_); // TODO slow
      ++K_; 
    }
    if(i %(z_.size()/100) == 0) cout<<" CrpMM<T>::sampleLabel: "<<(i/(z_.size()/100))<<"% done"<<endl;
  }
  this->removeEmptyClusters();
};

template<typename T>
void CrpMM<T>::sampleParameters()
{
//#pragma omp parallel for 
  for(uint32_t k=0; k<K_; ++k)
  {
    thetas_[k]->posterior(*this->spx_,z_,k);
//    cout<<"k:"<<k<<" ";
//    thetas_[k]->print();
  }
};

template <typename T>
void CrpMM<T>::removeEmptyClusters()
{
//  cout<<"K="<<K_<<" z="<<z_.transpose()<<endl;
  std::vector<bool> toDelete(K_,true);
//#pragma omp parallel for 
  for(uint32_t k=0; k<K_; ++k) 
  {
    toDelete[k] = true;
    for(uint32_t i=0; i<z_.size(); ++i)
      toDelete[k] = toDelete[k] && (z_(i)!=k);
  }

  std::vector<uint32_t> labelMap(K_,0);
  {
    uint32_t k=0;
    for(k=0; k<K_; ++k) if(!toDelete[k]) break;
    for(k=k+1; k<K_; ++k)
      if(toDelete[k]) 
        labelMap[k] = labelMap[k-1];
      else
        labelMap[k] = labelMap[k-1]+1;
//    for(k=0; k<K_; ++k) cout<<toDelete[k]<<" "; cout<<endl;
//    for(k=0; k<K_; ++k) cout<<labelMap[k]<<" "; cout<<endl;
  }
#pragma omp parallel for
  for(uint32_t i=0; i<z_.size(); ++i)
    z_(i) = labelMap[z_(i)];
//  cout<<z_.transpose()<<endl;

  for(int32_t k=K_-1; k>=0; --k)
    if (toDelete[k])
      thetas_.erase(thetas_.begin()+k);
  K_ = labelMap[K_-1]+1;
}

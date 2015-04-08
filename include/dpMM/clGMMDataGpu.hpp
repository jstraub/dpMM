/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */
 

#pragma once

#include <vector>
#include <Eigen/Dense>

#include <dpMM/gpuMatrix.hpp>
#include <dpMM/clGMMData.hpp>
#include <dpMM/timer.hpp>

using namespace Eigen;
using std::vector;

extern void sufficientStatistics_gpu(double *d_x, uint32_t *d_z , uint32_t N, 
    uint32_t k0, uint32_t K, double *d_SSs);
extern void gmmPdf(double * d_x, double *d_invSigmas, 
    double *d_logNormalizers, double *d_logPi, double* d_logPdf, uint32_t N, 
    uint32_t K_);

extern void sufficientStatistics_gpu(float *d_x, uint32_t *d_z , uint32_t N, 
    uint32_t k0, uint32_t K, float *d_SSs);
extern void gmmPdf(float * d_x, float *d_invSigmas, 
    float *d_logNormalizers, float *d_logPi, float* d_logPdf, uint32_t N, 
    uint32_t K_);

template<typename T>
class ClGMMDataGpu : public ClGMMData<T>
{
protected:
  GpuMatrix<uint32_t> d_z_; // indicators on GPU
  GpuMatrix<T> d_x_; // data-points
  
  GpuMatrix<T> d_Ss_; // sufficient statistics

  shared_ptr<GpuMatrix<T> > d_pdfs_; //
  GpuMatrix<T> d_logPi_; //
  GpuMatrix<T> d_logNormalizers_; //
  GpuMatrix<T> d_invSigmas_; //
 
public: 
  ClGMMDataGpu(const shared_ptr<Matrix<T,Dynamic,Dynamic> >& x, 
      const spVectorXu& z, uint32_t K);
  virtual ~ClGMMDataGpu(){;};
 
  virtual void init();

  virtual void updateLabels(uint32_t K);

  virtual void computeSufficientStatistics();

  virtual void computeLogLikelihoods(const Matrix<T,Dynamic,1>& pi, 
    const vector<Matrix<T,Dynamic,Dynamic> >& Sigmas, 
    const Matrix<T,Dynamic,1>& logNormalizers);
  virtual void sampleGMMpdf(const Matrix<T,Dynamic,1>& pi, 
      const vector<Matrix<T,Dynamic,Dynamic> >& Sigmas, 
      const Matrix<T,Dynamic,1>& logNormalizers, Sampler<T> *sampler);

  Matrix<T,Dynamic,Dynamic> pdfs(){ return d_pdfs_->get();};
  shared_ptr<GpuMatrix<T> >& d_pdfs(){ return d_pdfs_;};
  GpuMatrix<T>& d_x(){ return d_x_;};
  GpuMatrix<uint32_t>& d_z(){ return d_z_;};

protected:

  virtual void computeSufficientStatistics(uint32_t k0, uint32_t K);
};

typedef ClGMMDataGpu<double> ClGMMDataGpud;
typedef ClGMMDataGpu<float> ClGMMDataGpuf;

// ------------------------------------ impl --------------------------------
template<typename T>
ClGMMDataGpu<T>::ClGMMDataGpu(const shared_ptr<Matrix<T,Dynamic,Dynamic> >& x, 
    const spVectorXu& z, uint32_t K)
  : ClGMMData<T>(x,z,K), d_z_(this->N_,1), 
    d_x_(this->D_,this->N_),
    d_Ss_((this->D_-1)+(this->D_-1)*(this->D_-1)+1,this->K_),
    d_pdfs_(new GpuMatrix<T>(this->N_,this->K_)), d_logPi_(this->K_), 
    d_logNormalizers_(this->K_), 
    d_invSigmas_((this->D_-1)*(this->D_-1),this->K_)
{};

template<typename T>
void ClGMMDataGpu<T>::init()
{
  d_x_.set(*this->x_);
};


template<typename T>
void ClGMMDataGpu<T>::updateLabels(uint32_t K)
{
  this->K_ = K>0?K:this->z_->maxCoeff()+1;
  assert(this->z_->maxCoeff() < this->K_); // no indicators \geq K
  assert(this->z_->minCoeff() >= 0); // no indicators \le 0
  assert((this->z_->array() < this->K_).all());
  // update the labels on the GPU
  d_z_.set(this->z_);
}

template<typename T>
void ClGMMDataGpu<T>::computeSufficientStatistics(uint32_t k0, uint32_t K)
{
  // TODO: could probably move this up to bas class!
  assert(k0+K <= this->K_);
  assert(K<=6); //limitation of the GPU code - mainly of GPU shared mem
//  cout << d_x_.get() <<endl;

  Matrix<T,Dynamic,Dynamic> Ss = Matrix<T,Dynamic,Dynamic>::Zero(
      (this->D_-1)+(this->D_-1)*(this->D_-1)+1,K);
  d_Ss_.set(Ss);
  //  cout<<ps_.rows()<<"x"<<ps_.cols()<<endl;
//  cout<<"SS around"<<endl<<d_ps_.get()<<endl;
  // does reuse linearized datapoints
  sufficientStatistics_gpu(d_x_.data(), d_z_.data(), this->N_, k0, K, 
      d_Ss_.data());
  // does compute linearization by itself
//  sufficientStatisticsOnTpS2_gpu(d_ps_.data(), d_northRps_.data(), d_q_.data(), 
//      d_z_.data() , this->N_, k0, K, d_Ss_.data());
  d_Ss_.get(Ss); 
  cout<<Ss<<endl; 
  for (uint32_t k=0; k<K; ++k)
  {
    this->Ss_[k+k0](0,0) =  Ss(2,k);
    this->Ss_[k+k0](0,1) =  Ss(3,k);
    this->Ss_[k+k0](1,0) =  Ss(4,k);
    this->Ss_[k+k0](1,1) =  Ss(5,k);
    this->Ns_(k+k0) = Ss(6,k);
    if(this->Ns_(k+k0) > 0.) 
      this->means_.col(k+k0) = Ss.block(0,k,2,1)/this->Ns_(k+k0);
    // SS on GPU is just sum of outer products of all data -> make it a scatter
    // matrix now!
    //    cout<<"S"<<endl<<this->Ss_[k]<<endl;;
    this->Ss_[k+k0] -= this->Ns_(k+k0)*this->means_.col(k+k0)*this->means_.col(k+k0).transpose();
    //    cout<<"S"<<endl<<this->Ss_[k]<<endl;;
  }
}

template<typename T>
void ClGMMDataGpu<T>::computeSufficientStatistics()
{
  uint32_t k0 = 0;
  for (k0=0; k0<this->K_; k0+=6)
  {
    computeSufficientStatistics(k0,6); // max 6 SSs per kernel due to shared mem
  }
  if(this->K_ - k0 > 0)
    computeSufficientStatistics(k0,this->K_-k0);
  //cout<<xSums_<<endl;
  //cout<<Ns_<<endl;
}


template<typename T>
void ClGMMDataGpu<T>::computeLogLikelihoods(const Matrix<T,Dynamic,1>& pi, 
    const vector<Matrix<T,Dynamic,Dynamic> >& Sigmas, 
    const Matrix<T,Dynamic,1>& logNormalizers)
{
//  cout<<"ClGMMDataGpu<T>::sampleGMMpdf"<<endl;
  assert(pi.size() == this->K_);
  assert(logNormalizers.size() == this->K_);

  Matrix<T,Dynamic,Dynamic> invSigmas((this->D_-1)*(this->D_-1),this->K_);
//  Matrix<T,Dynamic,1> logNormalizer(this->K_);
  for(uint32_t k=0; k<this->K_; ++k)
  {
    Matrix<T,Dynamic,Dynamic> invS =Sigmas[k].inverse();
    for(uint32_t i=0; i<invS.cols(); ++i)
      for(uint32_t j=0; j<invS.rows(); ++j)
        invSigmas(i*invS.rows()+j,k) = invS(j,i);
//    logNormalizer(k) = -0.5*Sigmas[k]
  }
  assert( fabs(pi.sum()-1.0) <1e-6);
  Matrix<T,Dynamic,1> logPi = pi.array().log().matrix();
  //copy parameters into memory
  this->d_logPi_.set(logPi);  
  this->d_invSigmas_.set(invSigmas);
  this->d_logNormalizers_.set(logNormalizers);

//  cout<<"sphereGMMPdf"<<endl;
//  cout<<"logNormaloizers"<<endl<<logNormalizers.transpose()<<endl;
//  cout<<"logPi"<<endl<<logPi.transpose()<<endl;
//  cout<<invSigmas<<endl;
//  
//  d_ps_.data();
//  d_northRps_.data();
//  d_q_.data();
//  d_invSigmas_.data();
//  d_logNormalizers_.data();
//  d_logPi_.data();

  Matrix<T,Dynamic,Dynamic> pdfs(this->N_,this->K_); 
  if(!this->d_pdfs_->isInit())
  { 
    this->d_pdfs_->setZero();
//    pdfs = Matrix<T,Dynamic,Dynamic>::Zero(this->N_,this->K_);
//    d_pdfs_.set(pdfs);
  }

  gmmPdf(d_x_.data(), d_invSigmas_.data(), d_logNormalizers_.data(), 
    d_logPi_.data(), d_pdfs_->data(), this->N_, this->K_);
  
};

template<typename T>
void ClGMMDataGpu<T>::sampleGMMpdf(const Matrix<T,Dynamic,1>& pi, 
    const vector<Matrix<T,Dynamic,Dynamic> >& Sigmas, 
    const Matrix<T,Dynamic,1>& logNormalizers, Sampler<T> *sampler)
{

  computeLogLikelihoods(pi,Sigmas,logNormalizers);

  sampler->sampleDiscPdf(d_pdfs_->data(),this->z_);

//  cout<< this->z_->transpose()<<endl;
//  d_pdfs_.get(pdfs);
//  cout<<pdfs<<endl;
  
};

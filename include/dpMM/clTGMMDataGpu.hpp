/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */
 

#pragma once

#include <vector>
#include <Eigen/Dense>

#include <dpMM/gpuMatrix.hpp>
#include <dpMM/sphere.hpp>
#include <dpMM/clGMMDataGpu.hpp>
#include <dpMM/timer.hpp>

using namespace Eigen;
using std::vector;

extern  void Log_p_gpu(double *p, double *q, double *Rs, uint32_t *z, 
  uint32_t K, uint32_t N, double *x);
extern void meanInTpS2_gpu(double *d_p, double *d_mu_karch, double *d_q, 
    uint32_t *d_z , uint32_t N, uint32_t K);
extern void sufficientStatisticsOnTpS2_gpu(double *d_p, double *d_Rnorths,
    double *d_q, uint32_t *d_z , uint32_t N, uint32_t k0, uint32_t K, 
    double *d_SSs);
extern void sphereGMMPdf(double *d_p, double *d_Rnorths, double * d_q,
    double *d_invSigmas, double *d_logNormalizer, double *d_logPi, 
    double* d_logPdf, uint32_t N, uint32_t K);

extern  void Log_p_gpu(float *p, float *q, float *Rs, uint32_t *z, 
  uint32_t K, uint32_t N, float *x);
extern void meanInTpS2_gpu(float *d_p, float *d_mu_karch, float *d_q, 
    uint32_t *d_z , uint32_t N, uint32_t K);
extern void sufficientStatisticsOnTpS2_gpu(float *d_p, float *d_Rnorths,
    float *d_q, uint32_t *d_z , uint32_t N, uint32_t k0, uint32_t K, 
    float *d_SSs);
extern void sphereGMMPdf(float *d_p, float *d_Rnorths, float * d_q,
    float *d_invSigmas, float *d_logNormalizer, float *d_logPi, 
    float* d_logPdf, uint32_t N, uint32_t K);

template<typename T>
class ClTGMMDataGpu : public ClGMMDataGpu<T>
{
protected:
  GpuMatrix<T> d_q_; // normals on GPU
//  GpuMatrix<uint32_t> d_z_; // indicators on GPU
//  GpuMatrix<T> d_x_; // points in tangent plane (dim. is 1 less than q)
  
  //uint32_t K_; // number of different tangent planes 
  GpuMatrix<T> d_ps_; // points around which we have tangent spaces
  GpuMatrix<T> d_muKarch_; // karcher means
//  GpuMatrix<T> d_Ss_; // sufficient statistics in tangent spaces
  GpuMatrix<T> d_northRps_; // rotations from tangent spaces to north

//  GpuMatrix<T> d_pdfs_; //
//  GpuMatrix<T> d_logPi_; //
//  GpuMatrix<T> d_logNormalizers_; //
//  GpuMatrix<T> d_invSigmas_; //
 
  Matrix<T,Dynamic,Dynamic> ps_;
  
  Sphere<T> sphere_;

public: 
  ClTGMMDataGpu(const boost::shared_ptr<Matrix<T,Dynamic,Dynamic> >& q, 
      const spVectorXu& z, uint32_t K);
  ~ClTGMMDataGpu(){;};
  virtual void init(const Matrix<T,Dynamic,Dynamic>& ps);
  
  /* normal in tangent space around p rotate to the north pole
   * -> the last dimension will always be 0
   * -> return only first 2 dims
   */
  void Log_p_north(const Matrix<T,Dynamic,Dynamic>& ps, 
      const VectorXu& z, Matrix<T,Dynamic,Dynamic>& x, int32_t K=-1);
  void Log_p_north(const Matrix<T,Dynamic,Dynamic>& ps, 
      const VectorXu& z, int32_t K=-1);

  Matrix<T,Dynamic,Dynamic> Log_p_north(const Matrix<T,Dynamic,Dynamic>& ps, 
      int32_t K=-1)
  {
    Matrix<T,Dynamic,Dynamic> x(this->d_x_.rows(),this->d_x_.cols());
    Log_p_north(ps, *this->z_, x, K);
    return x;
  };

  virtual void update(uint32_t K);

  void relinearize(const Matrix<T,Dynamic,Dynamic>& ps);

  Matrix<T,Dynamic,Dynamic> karcherMeans(const Matrix<T,Dynamic,Dynamic>& p0,
      uint32_t maxIter = 50);

  virtual void computeLogLikelihoods(const Matrix<T,Dynamic,1>& pi, 
    const vector<Matrix<T,Dynamic,Dynamic> >& Sigmas, 
    const Matrix<T,Dynamic,1>& logNormalizers);
  virtual void sampleGMMpdf(const Matrix<T,Dynamic,1>& pi, 
      const vector<Matrix<T,Dynamic,Dynamic> >& Sigmas, 
      const Matrix<T,Dynamic,1>& logNormalizers, Sampler<T> *sampler);

  const Matrix<T,Dynamic,Dynamic>& ps() const {return ps_;};
  const Matrix<T,Dynamic,1>& p(uint32_t k) const {return ps_.col(k);};
  // TODO: broke compatibility! this used to be ps
//  Matrix<T,Dynamic,1> mean(uint32_t k) const {return ps_.col(k);};
//  const Matrix<T,Dynamic,Dynamic>& means() const {return ps_;};


  virtual const boost::shared_ptr<Matrix<T,Dynamic,Dynamic> >& q() const 
  {return this->x_;};
  virtual const Matrix<T,Dynamic,Dynamic>& qMat() const 
  {return (*this->x_);};

  GpuMatrix<T>& d_q(){ return d_q_;};

protected:
  Matrix<T,Dynamic,Dynamic> meanInTpS2(const Matrix<T,Dynamic,Dynamic>& ps);
  Matrix<T,Dynamic,Dynamic> karcherMeans__(const Matrix<T,Dynamic,Dynamic>& p0,
      uint32_t maxIter = 50);
};

typedef ClTGMMDataGpu<double> ClTGMMDataGpud;
typedef ClTGMMDataGpu<float> ClTGMMDataGpuf;

// ------------------------------------ impl --------------------------------
template<typename T>
ClTGMMDataGpu<T>::ClTGMMDataGpu(const boost::shared_ptr<Matrix<T,Dynamic,Dynamic> >& q, 
    const spVectorXu& z, uint32_t K)
  : ClGMMDataGpu<T>(q,z,K), d_q_(this->D_,this->N_), 
    d_ps_(this->D_,this->K_), d_muKarch_(this->D_+1,this->K_), 
    d_northRps_(this->D_-1,this->K_*this->D_),
    ps_(Matrix<T,Dynamic,Dynamic>::Zero(this->D_,this->K_)), 
    sphere_(this->D_)
{};

template<typename T>
void ClTGMMDataGpu<T>::init(const Matrix<T,Dynamic,Dynamic>& ps)
{
  for(uint32_t i=0; i<this->N_; ++i)
    if(fabs(this->x_->col(i).norm()-1.0) > 1e-5 )
    {
        cout<<"ClTGMMDataGpu<T>::ClTGMMDataGpu: warning: renormalizing normal "
          <<this->x_->col(i).norm()<<endl;
      this->x_->col(i) /= this->x_->col(i).norm();
    }
  d_q_.set(*this->x_);
  this->d_x_.setZero();

  // resize the statistyics to be one less dim, since we are in 
  // the tangent planes
  this->Ns_.setZero(this->K_);
  this->means_.setZero(this->D_-1,this->K_);
  for(uint32_t k=0; k<this->K_; ++k)
    this->Ss_[k].setZero(this->D_-1,this->D_-1);
  
  ps_ = ps;
//  for(uint32_t k=0; k<this->K_; ++k)
//    ps_.col(k) = sphere_.sampleUnif(this->pRndGen_);
};


template<typename T>
void ClTGMMDataGpu<T>::update(uint32_t K)
{
  this->K_ = K>0?K:this->z_->maxCoeff()+1;
  assert(this->z_->maxCoeff() < this->K_); // no indicators \geq K
  assert(this->z_->minCoeff() >= 0); // no indicators \le 0
  assert((this->z_->array() < this->K_).all());
  // update the labels on the GPU
  this->d_z_.set(this->z_);
  //TODO shouldnt need to do this everytime?
  //d_q_.set(*this->x_);
  // run karcher means to obtain ps
//  Matrix<T,Dynamic,Dynamic> q = d_q_.get();
//  cout<<q<<endl;
  Timer t;
  ps_ = karcherMeans__(ps_);
  t.toctic("karcherMeansFull");
  cout<<"linearize around: \n"<<ps_<<endl;
  // relinearize around the new centers
  t.tic();
  relinearize(ps_);
  t.toctic("relinearize");

//  // put points into tangent spaces - relinearize
//  Log_p_gpu(d_ps_.data(), d_q_.data(), d_northRps_.data(), this->d_z_.data(), this->K_, 
//      d_q_.cols(),d_this->x_.data());
  // compute moments
  this->computeSufficientStatistics();
  t.toctic("computeSufficientStatistics");
}

template<typename T>
void ClTGMMDataGpu<T>::relinearize(const Matrix<T,Dynamic,Dynamic>& ps)
{
  // TODO: need to call udapte Labels before this function
  
  // compute rotations to north and push them to GPU
  Matrix<T,Dynamic,Dynamic> Rs(ps.rows()-1, this->K_*3);
  for(uint32_t k=0; k<this->K_; ++k)
    Rs.middleCols(k*3,3) = sphere_.north_R_TpS2(ps.col(k)).topRows(2);
 
  d_ps_.set(ps);
  d_northRps_.set(Rs);

  // q is already in GPU since construction
//  d_ps_.print(); d_q_.print(); d_northRps_.print(); this->d_z_.print(); d_q_.print();
//  this->d_x_.print();
  
  Log_p_gpu(d_ps_.data(), d_q_.data(), d_northRps_.data(), this->d_z_.data(), 
      this->K_, d_q_.cols(), this->d_x_.data());
  
//  d_northRps_.get(Rs);
//  cout<<Rs<<endl;
//  Matrix<T,Dynamic,Dynamic> pps(ps.rows(),ps.cols());
//  d_ps_.get(pps);
//  cout<<pps<<endl;
};

template<typename T>
Matrix<T,Dynamic,Dynamic> ClTGMMDataGpu<T>::meanInTpS2(
    const Matrix<T,Dynamic,Dynamic>& ps)
{
  assert(ps.cols() == this->K_);
  assert(ps.rows() == this->D_);

  // one more dimension to hold the counts
  Matrix<T,Dynamic,Dynamic> muKarch = 
    Matrix<T,Dynamic,Dynamic>::Zero(this->D_+1,this->K_);
  d_muKarch_.set(muKarch);
  d_ps_.set(ps);

//  cout<<"ClTGMMDataGpu<T>::meanInTpS2: mean starting from "<<endl<<ps<<endl;
//  cout<<d_ps_.rows()<<" "<<d_ps_.cols()<<endl;
//  cout<<d_muKarch_.rows()<<" "<<d_muKarch_.cols()<<endl;
//  cout<<d_q_.rows()<<" "<<d_q_.cols()<<endl;
//  cout<<this->d_z_.rows()<<" "<<this->d_z_.cols()<<endl;

//  cout<<d_ps_.get()<<endl;
//  cout<<d_muKarch_.get()<<endl;
//  cout<<d_q_.get()<<endl;
//  cout<<this->d_z_.get()<<endl;

  meanInTpS2_gpu(d_ps_.data(), d_muKarch_.data(), d_q_.data(), this->d_z_.data(),
      this->N_,this->K_);
  //meanInTpS2GPU(h_p, d_p_, h_mu_karch, d_mu_karch_, d_q_, d_z, w_, h_);
  d_muKarch_.get(muKarch);

  Matrix<T,Dynamic,Dynamic> mu = muKarch.topRows(this->D_);
  for(uint32_t k=0; k<this->K_; ++k)
    if(muKarch(this->D_,k) > 0)
    {
      mu.col(k) /= muKarch(this->D_,k);
    }

//  cout<<muKarch<<endl<<endl;
//  cout<<mu<<endl;
  return mu;
}

template<typename T>
Matrix<T,Dynamic,Dynamic> ClTGMMDataGpu<T>::karcherMeans(const Matrix<T,Dynamic,Dynamic>& p0, uint32_t maxIter)
{
  // stand alone karcher means
  assert(this->z_->maxCoeff() < this->K_); // no indicators \geq K
  assert(this->z_->minCoeff() >= 0); // no indicators \le 0
  assert((this->z_->array() < this->K_).all());
  // update the labels on the GPU
  this->d_z_.set(this->z_);

  Matrix<T,Dynamic,Dynamic> p(p0.rows(),p0.cols());
  Timer t;
  p = karcherMeans__(p0);
  t.toctic("karcherMeansFull");
  return p;
};

template<typename T>
Matrix<T,Dynamic,Dynamic> ClTGMMDataGpu<T>::karcherMeans__(const Matrix<T,Dynamic,Dynamic>& p0, uint32_t maxIter)
{

  assert(p0.rows() == d_ps_.rows()); // we dont want dimension change
  Matrix<T,Dynamic,Dynamic> p = p0;
//  cout<<"p0"<<endl<<p<<endl;

  Matrix<T,Dynamic,1> residual(this->K_);
  residual.setOnes(this->K_);
//  cout<<(this->z_->transpose())<<endl;
  for(uint32_t i=0; i< maxIter; ++i)
  {
//    Timer t0;
    Matrix<T,Dynamic,Dynamic> mu_karch = meanInTpS2(p);
//    t0.toctic("meanInTpS2_GPU");
//    cout<<"mu_karch"<<endl<<mu_karch<<endl;
//    cout<<"p"<<endl<<p<<endl;
    for (uint32_t k=0; k<this->K_; ++k)
    {
      p.col(k) = sphere_.Exp_p(p.col(k),mu_karch.col(k));
//      cout<<p.col(k).norm()<<endl;
      residual(k) = mu_karch.col(k).norm();
    }
//    cout<<"p"<<endl<<p<<endl;
    //cout<<"karcherMeans "<<i<<" residual="<<residual<<endl;
//    cout<<"@"<<i<<" residual = "<<residual.transpose()<<endl;
    if((residual.array() < 1e-5).all())
    {
      cout<<"ClTGMMDataGpu<T>::karcherMeans__: converged after "<<i
        <<" residual = "<<residual.transpose()<<endl;
      assert((residual.array() != 0.0).any());
//      assert(i>0); // first itaration convergence is rare
      break;
    }
  }
//  cout<<"p"<<endl<<p<<endl;
  return p;
}


template<typename T>
void ClTGMMDataGpu<T>::Log_p_north(const Matrix<T,Dynamic,Dynamic>& ps, 
    const VectorXu& z, Matrix<T,Dynamic,Dynamic>& x, int32_t K)
{
  Log_p_north(ps,z,K);
  this->d_x_.get(x);
};


template<typename T>
void ClTGMMDataGpu<T>::Log_p_north(const Matrix<T,Dynamic,Dynamic>& ps, 
    const VectorXu& z,  int32_t K)
{
  if(K > 0){
    this->K_ = K>0?K:this->z_->maxCoeff()+1;
//    cout<<this->K_<<" "<< uint32_t(this->z_->maxCoeff()) <<endl;
    assert(this->z_->maxCoeff() < this->K_); // no indicators \geq K
    assert(this->z_->minCoeff() >= 0); // no indicators \le 0
    assert((this->z_->array() < this->K_).all());
  }
  // updates d_ps_, d_northRps_, this->d_z_, this->K_
  relinearize(ps);
  this->d_z_.set(z);
  //TODO shouldnt need to do this everytime?
  d_q_.set(*this->x_);
  // q is already in GPU since construction
//  d_ps_.print();
//  d_q_.print();
//  d_northRps_.print();
//  this->d_z_.print();
//  d_q_.print();
//  this->d_x_.print();
  
  Log_p_gpu(d_ps_.data(), d_q_.data(), d_northRps_.data(), this->d_z_.data(), 
      this->K_, d_q_.cols(),this->d_x_.data());
};

template<typename T>
void ClTGMMDataGpu<T>::computeLogLikelihoods(const Matrix<T,Dynamic,1>& pi, 
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
//  assert(fabs(pi.sum()-1.0) <1e-6);
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
//    d_pdfs_->set(pdfs);
  }

  sphereGMMPdf(d_ps_.data(), d_northRps_.data(), d_q_.data(), 
    this->d_invSigmas_.data(), this->d_logNormalizers_.data(), 
    this->d_logPi_.data(), this->d_pdfs_->data(), this->N_, this->K_);
};

template<typename T>
void ClTGMMDataGpu<T>::sampleGMMpdf(const Matrix<T,Dynamic,1>& pi, 
    const vector<Matrix<T,Dynamic,Dynamic> >& Sigmas, 
    const Matrix<T,Dynamic,1>& logNormalizers, Sampler<T> *sampler)
{
  computeLogLikelihoods(pi,Sigmas,logNormalizers);

  sampler->sampleDiscPdf(this->d_pdfs_->data(),this->z_);

//  cout<< this->z_->transpose()<<endl;
//  d_pdfs_->get(pdfs);
//  cout<<pdfs<<endl;
  
};

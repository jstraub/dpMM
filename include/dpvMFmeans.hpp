#pragma once

#include <Eigen/Dense>
#include <iostream>

#include <boost/random/mersenne_twister.hpp>
#include <boost/shared_ptr.hpp>

#include "dpmeans.hpp"

using namespace Eigen;
using std::cout;
using std::endl;

template<class T>
class DPvMFMeans : public DPMeans<T>
{
public:
  DPvMFMeans(const boost::shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx, uint32_t K0,
      double lambda, boost::mt19937* pRndGen);
  virtual ~DPvMFMeans();

//  virtual void updateLabels();
//  virtual void updateCenters();
  
  
  virtual T dist(const Matrix<T,Dynamic,1>& a, const Matrix<T,Dynamic,1>& b);
  virtual bool closer(T a, T b);
  virtual uint32_t indOfClosestCluster(int32_t i);
  virtual Matrix<T,Dynamic,1> computeCenter(uint32_t k);

//protected:
//  double lambda_;
};
// --------------------------- impl -------------------------------------------

template<class T>
DPvMFMeans<T>::DPvMFMeans(
    const boost::shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx, uint32_t K0, double lambda,
    boost::mt19937* pRndGen)
  : DPMeans<T>(spx,K0, lambda, pRndGen)
{
  assert(-2.0 < this->lambda_ && this->lambda_ < 0.0);
}

template<class T>
DPvMFMeans<T>::~DPvMFMeans()
{}

template<class T>
T DPvMFMeans<T>::dist(const Matrix<T,Dynamic,1>& a, const Matrix<T,Dynamic,1>& b)
{
//  return acos(min(1.0,max(-1.0,(a.transpose()*b)(0)))); // angular similarity
  return a.transpose()*b; // cosine similarity 
};

template<class T>
bool DPvMFMeans<T>::closer(T a, T b)
{
//  return a<b; // if dist a is greater than dist b a is closer than b (angular dist)
  return a>b; // if dist a is greater than dist b a is closer than b (cosine dist)
};

template<class T>
Matrix<T,Dynamic,1> DPvMFMeans<T>::computeCenter(uint32_t k)
{
  this->Ns_(k) = 0.0;
  Matrix<T,Dynamic,1> mean_k(this->D_);
  mean_k.setZero(this->D_);
  for(uint32_t i=0; i<this->N_; ++i)
    if(this->z_(i) == k)
    {
      mean_k += this->spx_->col(i); 
      this->Ns_(k) ++;
    }
  if(this->Ns_(k) > 0)
    return mean_k/mean_k.norm();
  else
    return mean_k;
}

template<class T>
uint32_t DPvMFMeans<T>::indOfClosestCluster(int32_t i)
{
  // use cosine similarity because it is faster since acos is not computed
  int z_i = this->K_;
  T sim_closest = this->lambda_ +1.;// new formulation -2<lambda<0
  for(uint32_t k=0; k<this->K_; ++k)
  {
    T sim_k = this->ps_.col(k).transpose()* this->spx_->col(i);
    if( sim_k > sim_closest) // because of cosine distance
    {
      sim_closest = sim_k;
      z_i = k;
    }
  }
  return z_i;
};

//template<class T>
//void DPvMFMeans<T>::updateLabels()
//{
////#pragma omp parallel for 
//// TODO not sure how to parallelize
//  for(uint32_t i=0; i<this->N_; ++i)
//  {
//
////    Matrix<T,Dynamic,1> sim(this->K_+1);
////    sim(this->K_) = lambda_;
////    for(uint32_t k=0; k<this->K_; ++k)
////      sim(k) = this->ps_.col(k).transpose()*this->spx_->col(i);
////    int z_i,dummy;
//////    cout<<sim.transpose()<<endl;
////    sim.maxCoeff(&z_i,&dummy);
//
//    int z_i = this->K_;
//    T sim_max = lambda_;
//    for (uint32_t k=0; k<this->K_; ++k)
//    {
//      T sim_k = this->ps_.col(k).transpose()*this->spx_->col(i);
//      if(sim_k > sim_max)
//      {
//        sim_max = sim_k;
//        z_i = k;
//      }
//    }
//
//    if(z_i == this->K_) 
//    {
//      MatrixXd psNew(this->D_,this->K_+1);
//      psNew.leftCols(this->K_) = this->ps_;
//      psNew.col(this->K_) = this->spx_->col(i);
//      this->ps_ = psNew;
//      this->K_ ++;
//    }
//    this->z_(i) = z_i;
//  }
//}
//
//template<class T>
//void DPvMFMeans<T>::updateCenters()
//{
//  vector<bool> toDelete(this->K_,false);
//#pragma omp parallel for 
//  for(uint32_t k=0; k<this->K_; ++k)
//  {
//    T N_k=0;
//    Matrix<T,Dynamic,1> mean_k(this->D_);
//    mean_k.setZero(this->D_);
//    for(uint32_t i=0; i<this->N_; ++i)
//      if(this->z_(i) == k)
//      {
//        mean_k += this->spx_->col(i); 
//        N_k ++;
//      }
//    if (N_k > 0) 
//      this->ps_.col(k) = mean_k/mean_k.norm();
//    else
//      toDelete[k] = true;
//  }
//
//  uint32_t kNew = this->K_;
//  for(int32_t k=this->K_-1; k>-1; --k)
//    if(toDelete[k])
//    {
//      cout<<"cluster k "<<k<<" empty"<<endl;
//#pragma omp parallel for 
//      for(uint32_t i=0; i<this->N_; ++i)
//      {
//        if(this->z_(i) >= k) this->z_(i)--;
//      }
//      kNew --;
//    }
//
//  MatrixXd psNew(this->D_,kNew);
//  int32_t offset = 0;
//  for(uint32_t k=0; k<this->K_; ++k)
//    if(toDelete[k])
//    {
//      offset ++;
//    }else{
//      psNew.col(k-offset) = this->ps_.col(k);
//    }
//  this->ps_ = psNew;
//  this->K_ = kNew;
//
////  cout<<"centers="<<endl<<this->ps_<<endl;
//}
//



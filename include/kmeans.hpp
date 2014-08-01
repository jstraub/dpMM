#pragma once

#include <Eigen/Dense>
#include <iostream>

#include <boost/random/mersenne_twister.hpp>
#include <boost/shared_ptr.hpp>

#include "sphere.hpp"
#include "clusterer.hpp"
#include "dir.hpp"
#include "cat.hpp"

using namespace Eigen;
using std::cout;
using std::endl;

template<class T>
class KMeans : public Clusterer<T>
{
public:
  KMeans(const shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx, uint32_t K,
    boost::mt19937* pRndGen);
  virtual ~KMeans();

//  void initialize(const Matrix<T,Dynamic,Dynamic>& x);

  virtual void updateLabels();
  virtual void updateCenters();
  virtual MatrixXu mostLikelyInds(uint32_t n, Matrix<T,Dynamic,Dynamic>& deviates);
  virtual T avgIntraClusterDeviation();
  
protected:
  Sphere<T> S_; 
};

// --------------------------- impl -------------------------------------------

template<class T>
KMeans<T>::KMeans(
    const shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx, uint32_t K,
    boost::mt19937* pRndGen)
  : Clusterer<T>(spx,K, pRndGen), S_(this->D_) 
{
  Matrix<T,Dynamic,1> alpha(this->K_);
  alpha.setOnes(this->K_);
  Dir<T> dir(alpha,this->pRndGen_);
  Cat<T> pi = dir.sample(); 
  cout<<"init pi="<<pi.pdf().transpose()<<endl;
  pi.sample(this->z_);

  this->ps_.setZero(); 
//  updateCenters();
//  for(uint32_t k=0; k<this->K_; ++k)
//    this->ps_.col(k) = S_.sampleUnif(this->pRndGen_);
//  cout<<"init centers"<<endl<<this->ps_<<endl;
}

template<class T>
KMeans<T>::~KMeans()
{}

template<class T>
void KMeans<T>::updateLabels()
{
#pragma omp parallel for 
  for(uint32_t i=0; i<this->N_; ++i)
  {
    Matrix<T,Dynamic,1> sim(this->K_);
    for(uint32_t k=0; k<this->K_; ++k)
      sim(k) = (this->ps_.col(k) - this->spx_->col(i)).norm();
    int z_i,dummy;
//    cout<<sim.transpose()<<endl;
    sim.minCoeff(&z_i,&dummy);
    this->z_(i) = z_i;
  }
}

template<class T>
void KMeans<T>::updateCenters()
{
#pragma omp parallel for 
  for(uint32_t k=0; k<this->K_; ++k)
  {
    T N_k=0;
    Matrix<T,Dynamic,1> mean_k(this->D_);
    mean_k.setZero(this->D_);
    for(uint32_t i=0; i<this->N_; ++i)
      if(this->z_(i) == k)
      {
        mean_k += this->spx_->col(i); 
        N_k ++;
      }
    if (N_k >0) 
      this->ps_.col(k) = mean_k/N_k;
    else 
      this->ps_.col(k) = S_.sampleUnif(this->pRndGen_);
  }
}

template<class T>
MatrixXu KMeans<T>::mostLikelyInds(uint32_t n, 
    Matrix<T,Dynamic,Dynamic>& deviates)
{
  MatrixXu inds = MatrixXu::Zero(n,this->K_);
  deviates = Matrix<T,Dynamic,Dynamic>::Ones(n,this->K_);
  deviates *= 99999.0;
  
#pragma omp parallel for 
  for (uint32_t k=0; k<this->K_; ++k)
  {
    for (uint32_t i=0; i<this->N_; ++i)
      if(this->z_(i) == k)
      {
        T deviate = (this->ps_.col(k) - this->spx_->col(i)).norm();
        for (uint32_t j=0; j<n; ++j)
          if(deviates(j,k) > deviate)
          {
            for(uint32_t l=n-1; l>j; --l)
            {
              deviates(l,k) = deviates(l-1,k);
              inds(l,k) = inds(l-1,k);
            }
            deviates(j,k) = deviate;
            inds(j,k) = i;
//            cout<<"after update "<<logLike<<endl;
//            Matrix<T,Dynamic,Dynamic> out(n,this->K_*2);
//            out<<logLikes.cast<T>(),inds.cast<T>();
//            cout<<out<<endl;
            break;
          }
      }
  } 
  cout<<"::mostLikelyInds: deviates"<<endl;
  cout<<deviates<<endl;
  cout<<"::mostLikelyInds: inds"<<endl;
  cout<<inds<<endl;
  return inds;
};

template<class T>
T KMeans<T>::avgIntraClusterDeviation()
{
  Matrix<T,Dynamic,1> deviates(this->K_);
  deviates.setZero(this->K_);
#pragma omp parallel for 
  for (uint32_t k=0; k<this->K_; ++k)
  {
    T N_k = 0.0;
    for (uint32_t i=0; i<this->N_; ++i)
      if(this->z_(i) == k)
      {
        deviates(k) += (this->ps_.col(k) - this->spx_->col(i)).norm();
        N_k ++;
      }
    if(N_k > 0.0) deviates(k) /= N_k;
  }
  return deviates.sum()/static_cast<T>(this->K_);
}

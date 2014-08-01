#pragma once

#include <Eigen/Dense>
#include <iostream>

#include <boost/random/mersenne_twister.hpp>
#include <boost/shared_ptr.hpp>

#include "sphere.hpp"
#include "karcherMean.hpp"
#include "clusterer.hpp"

using namespace Eigen;
using std::cout;
using std::endl;

template<class T>
class SphericalKMeans : public Clusterer<T>
{
public:
  SphericalKMeans(const shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx, uint32_t K,
    boost::mt19937* pRndGen);
  virtual ~SphericalKMeans();

//  void initialize(const Matrix<T,Dynamic,Dynamic>& x);

  virtual void updateLabels();
  virtual void updateCenters();
  virtual MatrixXu mostLikelyInds(uint32_t n, Matrix<T,Dynamic,Dynamic>& deviates);
  virtual T avgIntraClusterDeviation();

protected:
  Sphere<T> S_; 
};

template<class T>
class SphericalKMeansKarcher : public SphericalKMeans<T>
{
public:
  SphericalKMeansKarcher(const shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx, uint32_t K,
    boost::mt19937* pRndGen);
  ~SphericalKMeansKarcher();

//  void initialize(const Matrix<T,Dynamic,Dynamic>& x);

  void updateCenters();
};

// --------------------------------- impl -------------------------------------
template<class T>
SphericalKMeans<T>::SphericalKMeans(
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
//  for(uint32_t k=0; k<this->K_; ++k)
//    this->ps_.col(k) = S_.sampleUnif(this->pRndGen_);
//  cout<<"init centers"<<endl<<this->ps_<<endl;
}

template<class T>
SphericalKMeans<T>::~SphericalKMeans()
{}

template<class T>
void SphericalKMeans<T>::updateLabels()
{
#pragma omp parallel for 
  for(uint32_t i=0; i<this->N_; ++i)
  {
    Matrix<T,Dynamic,1> sim(this->K_);
    for(uint32_t k=0; k<this->K_; ++k)
      sim(k) = this->ps_.col(k).transpose()*this->spx_->col(i);
    int z_i,dummy;
//    cout<<sim.transpose()<<endl;
    sim.maxCoeff(&z_i,&dummy);
    this->z_(i) = z_i;
  }
}

template<class T>
void SphericalKMeans<T>::updateCenters()
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
      this->ps_.col(k) = mean_k/mean_k.norm();
    else 
      this->ps_.col(k) = S_.sampleUnif(this->pRndGen_);
  }
}

template<class T>
MatrixXu SphericalKMeans<T>::mostLikelyInds(uint32_t n, 
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
        T dot = this->ps_.col(k).transpose()*this->spx_->col(i);
        T deviate = acos(min(1.0,max(-1.0,dot)));
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
T SphericalKMeans<T>::avgIntraClusterDeviation()
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
        T dot = this->ps_.col(k).transpose()*this->spx_->col(i);
        deviates(k) += acos(min(1.0,max(-1.0,dot)));
        N_k ++;
      }
    if(N_k > 0.0) deviates(k) /= N_k;
  }
  return deviates.sum()/static_cast<T>(this->K_);
}

//template<class T>
//void SphericalKMeans<T>::initialize(const Matrix<T,Dynamic,Dynamic>& x)
//{
//  
//}
//
template<class T>
SphericalKMeansKarcher<T>::SphericalKMeansKarcher(
    const shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx, uint32_t K,
    boost::mt19937* pRndGen)
  : SphericalKMeans<T>(spx,K,pRndGen)
{}

template<class T>
SphericalKMeansKarcher<T>::~SphericalKMeansKarcher()
{}

template<class T>
void SphericalKMeansKarcher<T>::updateCenters()
{
  Matrix<T,Dynamic,Dynamic> xPs(this->spx_->rows(),this->spx_->cols());
#pragma omp parallel for
  for(uint32_t k=0; k<this->K_; ++k)
  {
//    Matrix<T,Dynamic,1> w(this->N_);
//    for(uint32_t i=0; i<this->N_; ++i)
//      if(this->z_(i) == k) 
//        w(i) = 1.0;
//      else
//        w(i) = 0.0;
//    this->ps_.col(k) = karcherMeanWeighted<T>(this->ps_.col(k), *(this->spx_), w, 100);
    this->ps_.col(k) = karcherMean<T>(this->ps_.col(k), *(this->spx_), 
        xPs, this->z_, k, 100,1);
  }
}

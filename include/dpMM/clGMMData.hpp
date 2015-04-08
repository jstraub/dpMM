/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>                    
 * Licensed under the MIT license. See the license file LICENSE.                
 */
#pragma once

#include <vector>
#include <algorithm>
#include <Eigen/Dense>

#include <dpMM/global.hpp>
#include <dpMM/sampler.hpp>

using std::vector;
using std::cout;
using std::endl;
using std::min;
using std::max;
using boost::shared_ptr;

/* clustered data */
template <class T>
class ClGMMData
{
protected:
  spVectorXu z_; // labels
  boost::shared_ptr<Matrix<T,Dynamic,Dynamic> > x_; // data

  uint32_t K_; // number of classes
  uint32_t N_;
  uint32_t D_;

  Matrix<T,Dynamic,1> Ns_; // counts
  Matrix<T,Dynamic,Dynamic> means_; // means
  vector<Matrix<T,Dynamic,Dynamic> > Ss_; //scatter matrices

public:
  ClGMMData(const boost::shared_ptr<Matrix<T,Dynamic,Dynamic> >& x, 
      const spVectorXu& z, uint32_t K);
  virtual ~ClGMMData();

  virtual void init();

  /* after changing z_ outside - we can use update to get new statistics */
  virtual void update(uint32_t K);

  virtual void sampleGMMpdf(const Matrix<T,Dynamic,1>& pi, 
      const vector<Matrix<T,Dynamic,Dynamic> >& Sigmas, 
      const Matrix<T,Dynamic,1>& logNormalizers, Sampler<T> *sampler)
  {cout<<"NOT implemented"<<endl;assert(false);};

//  virtual const spVectorXu& z() const {return z_;};
  virtual VectorXu& z() {return *z_;};
  virtual uint32_t z(uint32_t i) const {return (*z_)(i);};
  virtual const spVectorXu& labels() const {return z_;};
  virtual const boost::shared_ptr<Matrix<T,Dynamic,Dynamic> >& x() const {return x_;};
  virtual const Matrix<T,Dynamic,Dynamic>& xMat() const {return (*x_);};
  virtual uint32_t N() const {return N_;};
  virtual uint32_t K() const {return K_;};
  virtual uint32_t D() const {return D_;};

  virtual const Matrix<T,Dynamic,1>& counts() const {return Ns_;};
  virtual T count(uint32_t k) const {return Ns_(k);};

  virtual const Matrix<T,Dynamic,Dynamic>& means() const {return means_;};
  virtual Matrix<T,Dynamic,1> mean(uint32_t k) const {return means_.col(k);};

  virtual const vector<Matrix<T,Dynamic,Dynamic> >& scatters() const 
  {return Ss_;};
  virtual const Matrix<T,Dynamic,Dynamic>& S(uint32_t k) const 
  {return Ss_[k];};
};

typedef ClGMMData<float> ClGMMDataf;
typedef ClGMMData<double> ClGMMDatad;


template<typename T>
struct Spherical 
{
  static T dist(const Matrix<T,Dynamic,1>& a, const Matrix<T,Dynamic,1>& b)
  { return a.transpose()*b; };

  static T dissimilarity(const Matrix<T,Dynamic,1>& a, const Matrix<T,Dynamic,1>& b)
  { return acos(min(static_cast<T>(1.0),max(static_cast<T>(-1.0),(a.transpose()*b)(0)))); };

  static bool closer(const T a, const T b)
  { return a > b; };
};

template<typename T>
struct Euclidean 
{
  static T dist(const Matrix<T,Dynamic,1>& a, const Matrix<T,Dynamic,1>& b)
  { return (a-b).squaredNorm(); };

  static T dissimilarity(const Matrix<T,Dynamic,1>& a, const Matrix<T,Dynamic,1>& b)
  { return (a-b).squaredNorm();};

  static bool closer(const T a, const T b) { return a<b; };
};

template<class T, class DS>
T silhouette(const ClGMMData<T>& cld)
{ 
  if(cld.K()<2) return -1.0;
//  assert(Ns_.sum() == N_);
  cout<<"N="<<cld.N()<<" K="<<cld.K()<<endl;
  cout<<"counts="<<cld.counts().transpose()<<endl;
  Matrix<T,Dynamic,1> sil(cld.N());
//#pragma omp parallel for
  for(uint32_t i=0; i<cld.N(); ++i)
  {
    Matrix<T,Dynamic,1> b = Matrix<T,Dynamic,1>::Zero(cld.K());
    for(uint32_t j=0; j<cld.N(); ++j)
      if(j != i)
      {
        b(cld.z(j)) += DS::dissimilarity(cld.x()->col(i),cld.x()->col(j));
      }
    for (uint32_t k=0; k<cld.K(); ++k) 
      b(k) /= cld.count(k);
//    b *= Ns_.cast<T>().cwiseInverse(); // Assumes Ns are up to date!
    T a_i = b(cld.z(i)); // average dist to own cluster
    T b_i = cld.z(i)==0 ? b(1) : b(0); // avg dist do closest other cluster
    for(uint32_t k=0; k<cld.K(); ++k)
      if(k != cld.z(i) && b(k) == b(k) && b(k) < b_i && cld.count(k) > 0)
      {
        b_i = b(k);
      }
    if(a_i < b_i)
      sil(i) = 1.- a_i/b_i;
    else if(a_i > b_i)
      sil(i) = b_i/a_i - 1.;
    else
      sil(i) = 0.;
//    cout<<"@"<<i<<" b="<<b.transpose()<<" b_i="<<b_i<<" a_i="<<a_i<<endl;
  }
  cout<<" sil.sum="<<sil.sum()<<endl;
  return sil.sum()/static_cast<T>(cld.N());
};

// -------------------------- impl --------------------------------------------
template<class T>
ClGMMData<T>::ClGMMData(const boost::shared_ptr<Matrix<T,Dynamic,Dynamic> >& x, 
    const spVectorXu& z, uint32_t K)
 : z_(z), x_(x), K_(K>0?K:z->maxCoeff()+1), N_(x->cols()), D_(x->rows()),
   Ss_(K_,Matrix<T,Dynamic,Dynamic>::Zero(D_,D_))
{
  cout<<"D="<<D_<<" N="<<N_<<" K="<<K_<<endl;
};

template<class T>
ClGMMData<T>::~ClGMMData()
{};

template<typename T>
void ClGMMData<T>::init()
{};

template<class T>
void ClGMMData<T>::update(uint32_t K)
{
  K_ = K;
  Ns_.setZero(K_);
  means_.setZero(D_,K_);

  for (uint32_t i=0; i<N_; ++i)
//  {
//    cout<<"N="<<N_<<" z_i="<<(*z_)(i)<<" i="<<i<<" K="<<K_<<endl;
    Ns_((*z_)(i))++;
//  }

  for(uint32_t i=0; i<N_; ++i)
    means_.col((*z_)(i)) += x_->col(i);
  for(uint32_t k=0; k<K_; ++k)
  {
    means_.col(k) /= Ns_(k);
    Ss_[k].setZero(D_,D_);
  }

//  for(uint32_t i=0; i<N_; ++i)
//    Ss_[(*z_)(i)] += (x_->col(i) - means_.col((*z_)(i)))*
//      (x_->col(i) - means_.col((*z_)(i))).transpose();
}


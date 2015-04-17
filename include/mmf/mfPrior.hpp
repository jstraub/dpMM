/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license.  See the license file LICENSE.
 */
 
#pragma once

#include <Eigen/Dense>

#include <dpMM/normalSphere.hpp>
#include <dpMM/dirMM.hpp>

/* 
 * Prior distribution for a Manhattan Frame. Assumes a uniform prior
 * over the rotation R of the MF
 */
template<typename T>
class MfPrior
{
public:
  // prior over the mixture over the six axes
  DirMM<T> dirMM_; 

  MfPrior(const MfPrior<T>& mf);
  ~MfPrior();

  MfPrior<T>* copy();

  MfPrior<T> posterior(const Matrix<T,Dynamic,Dynamic>& x, const
      VectorXu& z, uint32_t k);
  MfPrior<T> posterior(const vector<Matrix<T,Dynamic,Dynamic> >&x, const
      VectorXu& z, uint32_t k);
  // assumes vector [N, sum(x), flatten(sum(outer(x,x)))]
  MfPrior<T> posteriorFromSS(const Matrix<T,Dynamic,1>& x);
  MfPrior<T> posteriorFromSS(const vector<Matrix<T,Dynamic,1> >&x, const
      VectorXu& z, uint32_t k);

//  MfPrior<T> posterior() const;
//  void sample() const;
private:
  // data for this MF needed since we iterate over this data for
  // posterior inference
  Matrix<T,Dynamic,Dynamic> x_; 
  // how many iterations to sample internally for the MF posterior
  uint32_t T_;

  void sample_(uint32_t T);
};

// ----------------------------------------
template<typename T>
MfPrior<T>::MfPrior(const Dir<Cat<T>, T>& alpha, const
    shared_ptr<NiwSphereFull<T> >& theta0, uint32_t T)
  : dirMM_(alpha,theta,6), T_(T)
{};

template<typename T>
MfPrior<T>::sample_(uint32_t T)
{
  for(uint32_t t=0; t<T; ++t)
  {
    dirMM_.sampleLabels();
    dirMM_.sampleParameters();
    // some output
    cout<<"@t "<<t<<": logJoint = "<<dirMM_.logJoint() 
      <<" #s "<<dirMM_.getCounts().transpose()
      <<endl;
  }
}

template<typename T>
MfPrior<T>::posterior(const Matrix<T,Dynamic,Dynamic>& x, const
      VectorXu& z, uint32_t k)
{
  // count to know how big to make the data matrix
  uint32_t Nk = 0;
//#pragma omp parallel for
  for (int i=0; i<z.size(); ++i)
    if(z[i] == k) ++Nk;
  if(Nk > 0)
  {
    // fill data matrix
    x_.resize(x.rows(),Nk);
    uint32_t j=0; 
    for (int i=0; i<z.size(); ++i)
      if(z[i] == k) 
      {
        x_.col(j) = x.col(i);
        ++j;
      }
    dirMM_.initialize(x_);
    sample_(T_);
  }else{
    // sample from prior
    dirMM_.sampleFromPrior();
  }
}

template<typename T>
MfPrior<T>::

template<typename T>
MfPrior<T>::

template<typename T>
MfPrior<T>::

template<typename T>
MfPrior<T>::


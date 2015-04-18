/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license.  See the license file LICENSE.
 */
 
#pragma once

#include <Eigen/Dense>

#include <dpMM/dirMM.hpp>
#include <dpMM/iwTangent.hpp>
#include <dpMM/karcherMean.hpp>
#include <mmf/optimizationSO3_approx.hpp>
#include <mmf/mf.hpp>

/* 
 * Prior distribution for a Manhattan Frame. Assumes a uniform prior
 * over the rotation R of the MF
 */
template<typename T>
class MfPrior
{
public:
  MfPrior(const Dir<Cat<T>, T>& alpha, const
    std::vector<shared_ptr<IwTangent<T> > >& thetas, uint32_t nIter);

//  MfPrior(const MfPrior<T>& mf);
  ~MfPrior();

//  MfPrior<T>* copy();

  MF<T> posteriorSample(const Matrix<T,Dynamic,Dynamic>& x, const
      VectorXu& z, uint32_t k);
//  MfPrior<T> posterior(const vector<Matrix<T,Dynamic,Dynamic> >&x, const
//      VectorXu& z, uint32_t k);
//  // assumes vector [N, sum(x), flatten(sum(outer(x,x)))]
//  MfPrior<T> posteriorFromSS(const Matrix<T,Dynamic,1>& x);
//  MfPrior<T> posteriorFromSS(const vector<Matrix<T,Dynamic,1> >&x, const
//      VectorXu& z, uint32_t k);

//  MfPrior<T> posterior() const;
//  void sample() const;
  Matrix<T,3,3> R() const {return optSO3_.R();};
  Matrix<T,Dynamic,Dynamic> M() const {return optSO3_.M();};
private:
  // data for this MF needed since we iterate over this data for
  // posterior inference
  Matrix<T,Dynamic,Dynamic> x_; 
  // how many iterations to sample internally for the MF posterior
  uint32_t T_;

  std::vector<shared_ptr<IwTangent<T> > > thetas_;
  // prior over the mixture over the six axes
  DirMM<T> dirMM_; 

  OptSO3ApproxCpu<T> optSO3_;
  Matrix<T,3,3> R_;

  void sample_(uint32_t nIter);
};

// ----------------------------------------
template<typename T>
MfPrior<T>::MfPrior(const Dir<Cat<T>, T>& alpha, const
    std::vector<shared_ptr<IwTangent<T> > >& thetas, uint32_t nIter)
  : T_(nIter), 
    thetas_(thetas),
    dirMM_(alpha,thetas_[0],6), optSO3_(5.0,0.05)
{
    R_ = Matrix<T,3,3>::Identity();
};

template<typename T>
MfPrior<T>::~MfPrior()
{};

template<typename T>
void MfPrior<T>::sample_(uint32_t nIter)
{
  // get the pointers to the thetas in the DirMM so we can set the
  // means
  for(uint32_t k=0; k<6; ++k)
  {
    thetas_[k].reset(dynamic_cast<IwTangent<T>* >(
          dirMM_.getTheta(k).get()));
    thetas_[k]->setMean(optSO3_.M().col(k));
  }
  for(uint32_t t=0; t<nIter; ++t)
  {
    cout<<" -------------------- "<<endl;
    // sample labels - assignments to MF axes
    dirMM_.sampleLabels();
    // compute karcher means
    Matrix<T,Dynamic,Dynamic> xTpS(x_.rows(),x_.cols()); 
    Matrix<T,Dynamic, Dynamic> qKarch = karcherMeanMultiple<T>(
        optSO3_.M(),x_,xTpS,dirMM_.labels(),6,30);
    cout<<"qKarch: "<<endl<<qKarch<<endl;
    // sample new MF rotation
    optSO3_.conjugateGradient(R_, qKarch, dirMM_.getCounts(), 100);
    R_ = optSO3_.R();
//    cout<<"R: "<<endl<<R_<<endl;
    cout<<"M: "<<endl<<optSO3_.M()<<endl;
    // set tangent points of the iwTangent
    for(uint32_t k=0; k<6; ++k)
    {
      thetas_[k]->setMean(optSO3_.M().col(k));
      dirMM_.getTheta(k)->print();
    }
    // sample covariances in the tangent spaces
    dirMM_.sampleParameters();
    // some output
    cout<<"@t "<<t<<": logJoint = "<<dirMM_.logJoint() 
      <<" #s "<<dirMM_.getCounts().transpose()
      <<endl;
  }
}

template<typename T>
MF<T> MfPrior<T>::posteriorSample(const Matrix<T,Dynamic,Dynamic>& x,
    const VectorXu& z, uint32_t k)
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
  std::vector<NormalSphere<T> > TGs;
  for(uint32_t k=0; k<6; ++k)
  {
    TGs.push_back(thetas_[k]->normalS_);
  }
  return MF<T>(optSO3_.R(),dirMM_.Pi(),TGs);
}

//template<typename T>
//MfPrior<T>::
//
//template<typename T>
//MfPrior<T>::
//
//template<typename T>
//MfPrior<T>::
//
//template<typename T>
//MfPrior<T>::
//

/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license.  See the license file LICENSE.
 */
 
#pragma once

#include <Eigen/Dense>

#include <boost/random/uniform_01.hpp>

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
  MfPrior(const MfPrior<T>& mf); 
  ~MfPrior();

//  MfPrior<T>* copy();
  T logPdf(const MF<T>& mf) const;

  MF<T> posteriorSample(const Matrix<T,Dynamic,Dynamic>& x, const
      VectorXu& z, uint32_t k);
//  MF<T> posteriorSample(const vector<Matrix<T,Dynamic,Dynamic> >&x, const
//      VectorXu& z, uint32_t k);
//  // assumes vector [N, sum(x), flatten(sum(outer(x,x)))]
//  MfPrior<T> posteriorFromSS(const Matrix<T,Dynamic,1>& x);
  MF<T> posteriorFromSSsample(const vector<Matrix<T,Dynamic,1> >&x, 
      const VectorXu& z, uint32_t k);

//  MfPrior<T> posterior() const;
  MF<T> sample() const;

  Matrix<T,3,3> R() const {return optSO3_.R();};

  Matrix<T,Dynamic,Dynamic> M() const {return optSO3_.M();};

  const DirMM<T>& dirMM() const {return dirMM_;};
  const shared_ptr<IwTangent<T> >& theta(uint32_t k) const 
  {return thetas_[k];};
  vector<shared_ptr<IwTangent<T> > > thetasDeepCopy() const 
  {
    vector<shared_ptr<IwTangent<T> > > thetas;
    for(uint32_t k=0; k<6; ++k)
      thetas.push_back(shared_ptr<IwTangent<T> >(
            thetas_[k]->copyNative()));
    return thetas;
  };


  // how many iterations to sample internally for the MF posterior
  uint32_t T_;
  Matrix<T,3,3> R_;
private:
//  // data for this MF needed since we iterate over this data for
//  // posterior inference
//  Matrix<T,Dynamic,Dynamic> x_; 

  std::vector<shared_ptr<IwTangent<T> > > thetas_;
  // prior over the mixture over the six axes
  DirMM<T> dirMM_; 

  OptSO3ApproxCpu<T> optSO3_;

  void sample_(const Matrix<T,Dynamic,Dynamic>& x, uint32_t nIter);
  void sample_(const Matrix<T,Dynamic,Dynamic>& x, 
    vector<Matrix<T,Dynamic,1> >& SS, uint32_t nIter);
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
MfPrior<T>::MfPrior(const MfPrior<T>& mfP)
  : T_(mfP.T_), 
  R_(mfP.R_), 
  thetas_(mfP.thetasDeepCopy()),
  dirMM_(mfP.dirMM()),
  optSO3_(5.0,0.05)
{};

template<typename T>
MfPrior<T>::~MfPrior()
{};

template<typename T>
void MfPrior<T>::sample_(const Matrix<T,Dynamic,Dynamic>& x,
    uint32_t nIter)
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
    Matrix<T,Dynamic,Dynamic> xTpS(x.rows(),x.cols()); 
    Matrix<T,Dynamic, Dynamic> qKarch = karcherMeanMultiple<T>(
        optSO3_.M(),x,xTpS,dirMM_.labels(),6,30);
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
void MfPrior<T>::sample_(const Matrix<T,Dynamic,Dynamic>& x,
    vector<Matrix<T,Dynamic,1> >& SS,
    uint32_t nIter)
{
  // TODO: the dirMM sampler ignores the SS (i.e. the number of data as
  // well as the scatter of the data in their respective tangent
  // space!)

  // get the pointers to the thetas in the DirMM so we can set the
  // means
  for(uint32_t k=0; k<6; ++k)
  {
    thetas_[k].reset(dynamic_cast<IwTangent<T>* >(
          dirMM_.getTheta(k).get()));
    thetas_[k]->setMean(optSO3_.M().col(k));
  }

  Matrix<T,Dynamic,1> w(x.cols()); 
  for(uint32_t i=0; i<SS.size(); ++i)
    w(i) = SS[i](0);

  for(uint32_t t=0; t<nIter; ++t)
  {
    cout<<" -------------------- "<<endl;
    // sample labels - assignments to MF axes
    dirMM_.sampleLabels();
    // compute karcher means
    Matrix<T,Dynamic,Dynamic> xTpS(x.rows(),x.cols()); 
    Matrix<T,Dynamic, Dynamic> qKarch = karcherMeanMultipleWeighted<T>(
        optSO3_.M(),x,xTpS,w,dirMM_.labels(),6,30);
    cout<<"qKarch: "<<endl<<qKarch<<endl;
    cout<<"Mbefore: "<<endl<<optSO3_.M()<<endl;
    cout<<"R: "<<endl<<R_<<endl;
    // compute counts for each of the 6 directions from SS
    Matrix<T,Dynamic,1> Ws = Matrix<T,Dynamic,1>::Zero(6); 
    for(uint32_t i=0; i<SS.size(); ++i)
      Ws(dirMM_.labels()(i)) += w(i);
    cout<<"Ws: "<<Ws.transpose()<<endl;
    cout<<"#s: "<<dirMM_.getCounts().transpose()<<endl;
    // sample new MF rotation
    optSO3_.conjugateGradient(R_, qKarch, Ws, 100);
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
    Matrix<T,Dynamic,Dynamic> x_(x.rows(),Nk);
    uint32_t j=0; 
    for (int i=0; i<z.size(); ++i)
      if(z[i] == k) 
      {
        x_.col(j) = x.col(i);
        ++j;
      }
    dirMM_.initialize(x_);
    sample_(x_,T_);
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

template<typename T>
MF<T> MfPrior<T>::posteriorFromSSsample(const 
    vector<Matrix<T,Dynamic,1> >&x, const VectorXu& z, uint32_t k)
{
  // count to know how big to make the data matrix
  uint32_t Nk = 0;
//#pragma omp parallel for
  for (int i=0; i<z.size(); ++i)
    if(z[i] == k) ++Nk;
  if(Nk > 0)
  {
    // fill data matrix
    vector<Matrix<T,Dynamic,1> > SS(Nk); // SS for this cluster
    Matrix<T,Dynamic,Dynamic> x_(3,Nk);  // karcher means of this cl
    uint32_t j=0; 
    for (int i=0; i<z.size(); ++i)
      if(z[i] == k) 
      {
        SS[j] = x[i];
        x_.col(j) = x[i].middleRows(1,3);
        ++j;
      }
    dirMM_.initialize(x_);
    sample_(x_,SS,T_);
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

template<typename T>
T MfPrior<T>::logPdf(const MF<T>& mf) const
{
  //TODO hmmm this assumes R_ = mf.R() have to think more about what
  //happens if the rotations are different
  T logPdf = 0.;
  for(uint32_t k =0; k<6; ++k)
  {
    logPdf += thetas_[k]->iw0_.logPdf(mf.Sigma(k));
  }
  return logPdf;
};

template<typename T>
MF<T> MfPrior<T>::sample() const
{
  //TODO sample from all SO3
  Matrix<T,3,3> R = Matrix<T,3,3>::Identity();

  boost::uniform_01<T> unif;
  double theta = unif(*(thetas_[0]->iw0_.pRndGen_));
  theta *= 2.*M_PI;
  R << cos(theta), -sin(theta), 0.0,
    sin(theta), cos(theta), 0.0,
    0.0,0.0,1.0;

  Cat<T> pi = const_cast<Dir<Cat<T>,T>* >(&(dirMM_.Alpha()))->sample();
  std::vector<NormalSphere<T> > TGs;
  for(uint32_t k =0; k<6; ++k)
  {
    //TODO fishy
    thetas_[0]->sample();
    TGs.push_back(NormalSphere<T>(
          OptSO3ApproxCpu<T>::Rot2M(R).col(k), 
          thetas_[0]->normalS_.Sigma(), thetas_[0]->iw0_.pRndGen_
          ));
  }
  return MF<T>(R_,pi,TGs);
};

//template<typename T>
//MfPrior<T>::
//
//template<typename T>
//MfPrior<T>::
//

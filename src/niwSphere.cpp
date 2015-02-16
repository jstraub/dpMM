/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */

#include "niwSphere.hpp"
// ----------------------------------------------------------------------------

template<typename T>
NiwSphere<T>::NiwSphere(const IW<T>& iw, boost::mt19937* pRndGen)
  : iw0_(iw), S_(iw.D_+1), normalS_(S_.sampleUnif(pRndGen),iw0_.sample(),pRndGen)
{};

template<typename T>
NiwSphere<T>::~NiwSphere()
{};

template<typename T>
BaseMeasure<T>* NiwSphere<T>::copy()
{
  NiwSphere<T>* niwSp = new NiwSphere<T>(iw0_,normalS_.pRndGen_);
  niwSp->normalS_ = normalS_;
  return niwSp;
};

template<typename T>
NiwSphere<T>* NiwSphere<T>::copyNative()
{
  NiwSphere<T>* niwSp = new NiwSphere<T>(iw0_,normalS_.pRndGen_);
  niwSp->normalS_ = normalS_;
  return niwSp;
};

template<typename T>
T NiwSphere<T>::logLikelihood(const Matrix<T,Dynamic,1>& q) const
{
  return normalS_.logPdf(q);
};

template<typename T>
T NiwSphere<T>::logLikelihoodNorth(const Matrix<T,Dynamic,1>& x) const
{
  return normalS_.logPdfNorth(x);
};

template<typename T>
void NiwSphere<T>::sample()
{
  normalS_.setSigma(iw0_.posterior().sample());
};

template<typename T>
void NiwSphere<T>::posteriorFromPtsInTpS(const Matrix<T,Dynamic,Dynamic>& x, 
    const VectorXu& z, uint32_t k)
{
  normalS_.setSigma(iw0_.posterior(x,z,k).sample());
#ifndef NDEBUG
  cout<<"NiwSphere<T>::posteriorFromPtsInTpS"<<endl
    <<normalS_.getMean().transpose()<<endl
    <<normalS_.Sigma()<<endl
    <<"Count: "<<iw0_.count()<<endl
    <<"mu:    "<<normalS_.getMean().transpose()<<endl
    <<iw0_.scatter()<<endl;
#endif
};

template<typename T>
void NiwSphere<T>::posterior(const Matrix<T,Dynamic,Dynamic>& q, 
    const VectorXu& z, uint32_t k)
{
  Matrix<T,Dynamic,1> w(z.size()); 
  w.setZero(z.size());
#pragma omp parallel for
  for (int32_t i=0; i<z.size(); ++i)
    if(z[i] == k) w[i]=1.0;
  if(w.sum() > 0)
  {
    normalS_.setMean(karcherMeanWeighted<T>(normalS_.getMean(), q, w, 100));
    //TODO: slight permutation here for mu to allow proper sampling
    // TODO: wastefull since it computes stuff for normals that are not used in the 
    // later computations
    Matrix<T,Dynamic,Dynamic> x_mu = S_.Log_p_north(normalS_.getMean(),q);
    normalS_.setSigma(iw0_.posterior(x_mu,z,k).sample());
  
  }else{
    normalS_.setMean(S_.sampleUnif(normalS_.pRndGen_));
    iw0_.resetSufficientStatistics();
    normalS_.setSigma(iw0_.sample());
  }
//  sample();
  //  cout<<"Delta: \n"<<iw0_.posterior(x_mu,z,k).Delta_<<endl;
  //  cout<<"Sigma: \n"<<normalS_.Sigma()<<endl;
//  cout<<"Sigma Eigs:"<<normalS_.Sigma().eigenvalues()<<endl;
#ifndef NDEBUG
  cout<<"NiwSphere<T>::posterior"<<endl
    <<normalS_.getMean().transpose()<<endl
    <<normalS_.Sigma()<<endl;
#endif
};

template<typename T>
void NiwSphere<T>::posterior( const shared_ptr<ClGMMData<T> >& cldp, uint32_t k)
{
  normalS_.setMean(cldp->mean(k));
  iw0_.scatter() = cldp->S(k);
  iw0_.count() = cldp->count(k);
  normalS_.setSigma(iw0_.posterior().sample());
#ifndef NDEBUG
  cout<<"NiwSphere<T>::posterior"<<endl
    <<normalS_.getMean().transpose()<<endl
    <<normalS_.Sigma()<<endl;
#endif
};

template<typename T>
T NiwSphere<T>::logPdfUnderPrior() const
{
//TODO prior for normal onto sphere
  return normalS_.logPdfNorth(iw0_.scatter(),iw0_.mean(),iw0_.count()) 
    + iw0_.logPdf(normalS_.Sigma()) - S_.logSurfaceArea();
};

template<typename T>
T NiwSphere<T>::logPdfUnderPriorMarginalized() const
{
//TODO prior for normal onto sphere
  return iw0_.logLikelihoodMarginalized()  - S_.logSurfaceArea();
};

template<typename T>
T NiwSphere<T>::logPdfUnderPriorMarginalizedMerged(
    const shared_ptr<NiwSphere<T> >& other) const
{
  Matrix<T,Dynamic,Dynamic> scatterMerged(iw0_.D_,iw0_.D_);
  Matrix<T,Dynamic,1> muMerged(iw0_.D_);
  T countMerged=0;
  return iw0_.logLikelihoodMarginalized(scatterMerged, countMerged) 
    - S_.logSurfaceArea();
};

template<typename T>
void NiwSphere<T>::computeMergedSS( const NiwSphere<T>& niwA, 
    const NiwSphere<T>& niwB, Matrix<T,Dynamic,Dynamic>& scatterM, 
    Matrix<T,Dynamic,1>& muM, T& countM) const
{
  countM = niwA.count() + niwB.count();
#ifndef NDEBUG
  cout<<countM<<" "<<niwA.count()<<" "<<niwB.count()<<endl;
#endif
  if(countM == 0)
  {
    scatterM.setZero();
    muM = niwA.getMean();
    return;
  }

//  Matrix<T,Dynamic,Dynamic> q(niwA.normalS_.mu_.rows(),2);
//  Matrix<T,Dynamic,1> w(2);
//  q<< niwA.normalS_.mu_, niwB.normalS_.mu_;
//  w<< niwA.count(), niwB.count();
//  this->normalS_.mu_ = karcherMeanWeighted<T>(niwA.normalS_.mu_,q,w,100);
//  cout<<this->normalS_.mu_<<endl;

  // instead of using iterative karcher means I can compute the weighted mean 
  // simply moving along the geodesic path from meanA to meanB.
  muM = rotationFromAtoB<T>(niwA.getMean(), niwB.getMean(), 
      niwB.count()/countM)*niwA.getMean();
//  cout<<this->normalS_.mu_<<endl;
//  assert(false);

//  cout<<muM.transpose()<<endl;
//  cout<<niwA.getMean().transpose()<<endl;
//  cout<<niwB.getMean().transpose()<<endl;
  Matrix<T,Dynamic,1> muAinTthis = S_.Log_p_north(muM, niwA.getMean());
  Matrix<T,Dynamic,1> muBinTthis = S_.Log_p_north(muM, niwB.getMean());

  scatterM = niwA.scatter() + niwB.scatter();
  scatterM += niwA.count()*muAinTthis*muAinTthis.transpose();
  scatterM += niwB.count()*muBinTthis*muBinTthis.transpose();
#ifndef NDEBUG
  cout<<"scatter before correction"<<endl<<scatterM<<endl;
  cout<<"logP under Prior: "<<iw0_.logLikelihoodMarginalized(scatterM, countM)<<endl;
#endif

#ifndef NDEBUG
  cout<<"scatter after correction"<<endl<<scatterM<<endl;
  cout<<"logP under Prior: "<<iw0_.logLikelihoodMarginalized(scatterM, countM)<<endl;
#endif
};

template<typename T>
void NiwSphere<T>::print() const
{
  cout<<"mu:     "<<normalS_.getMean().transpose()<<endl;
  cout<<"Sigma:  "<<endl<<normalS_.Sigma()<<endl;
  cout<<"Scatter:"<<endl<<scatter()<<endl;
  cout<<"count:  "<<count()<<endl;
};

template<typename T>
NiwSphere<T>* NiwSphere<T>::merge(const NiwSphere<T>& other)
{
  NiwSphere<T>* merged = this->copyNative();
  merged->fromMerge(*this,other);
  return merged;
};

template<typename T>
void NiwSphere<T>::fromMerge(const NiwSphere<T>& niwA, const NiwSphere<T>& niwB)
{
  Matrix<T,Dynamic,1> muM(iw0_.D_+1);
  computeMergedSS(niwA, niwB, iw0_.scatter(), muM, iw0_.count());
  this->setMean(muM);

//  this->normalS_.normal_ = Normal<T>(iw0_.posterior().sample(), normalS_.pRndGen_);
  this->normalS_.setSigma(iw0_.posterior().sample());
};

// ---------------------------------------------------------------------------
template class NiwSphere<double>;
template class NiwSphere<float>;

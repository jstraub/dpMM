/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */
#include "niwSphereFull.hpp"
// ----------------------------------------------------------------------------

template<typename T>
NiwSphereFull<T>::NiwSphereFull(const IW<T>& iw, 
      boost::mt19937* pRndGen)
  : D_(iw.D_+1), iw0_(iw), S_(D_), 
  normalS_(S_.sampleUnif(pRndGen),iw0_.sample(),pRndGen)
{
  meanKarch_.setZero(D_);
};

template<typename T>
NiwSphereFull<T>::~NiwSphereFull()
{};

template<typename T>
BaseMeasure<T>* NiwSphereFull<T>::copy()
{
  NiwSphereFull<T>* niwSp = new NiwSphereFull<T>(iw0_, normalS_.pRndGen_);
  niwSp->normalS_ = normalS_;
  niwSp->meanKarch_= meanKarch_;

  return niwSp;
};

template<typename T>
NiwSphereFull<T>* NiwSphereFull<T>::copyNative()
{
  NiwSphereFull<T>* niwSp = new NiwSphereFull<T>(iw0_, normalS_.pRndGen_);
  niwSp->normalS_ = normalS_;
  niwSp->meanKarch_= meanKarch_;

  return niwSp;
};

template<typename T>
T NiwSphereFull<T>::logLikelihood(const Matrix<T,Dynamic,1>& q) const
{
//  cout<<" NiwSphereFull<T>::logLikelihood: "<<normalS_.logPdf(x)<<endl;
  return normalS_.logPdf(q);
};

template<typename T>
T NiwSphereFull<T>::logLikelihoodNorth(const Matrix<T,Dynamic,1>& x) const
{
//  cout<<" NiwSphereFull<T>::logLikelihood: "<<normalS_.logPdf(x)<<endl;
  return normalS_.logPdfNorth(x);
};


template<typename T>
void NiwSphereFull<T>::sampleMergedParams()
{
  // sample new mean from proposal
  NormalSphere<T> mergeProposal(meanKarch_, iw0_.posterior().mode(),
    iw0_.pRndGen_);
  normalS_.setMean(mergeProposal.sample());
//  cout<<"merge proposed mean "<<normalS_.getMean().transpose()<<endl
//    <<"   sampled from karcher mean "<<meanKarch_.transpose()<<endl;
  // this updates the sample mean in TpS since we want the SS in T_muS
  // scatter stays the same in new mu 
  iw0_.mean() = S_.Log_p_north(normalS_.getMean(), meanKarch_);

  // sample Sigma conditioned on the new mu
  normalS_.setSigma(iw0_.posterior().sample());
}

template<typename T>
void NiwSphereFull<T>::sample()
{
  // sample covariance matrix from posterior IW
  if (iw0_.count() == 0)
  {
    // sample from prior distribution
    normalS_.setSigma(iw0_.sample());
    normalS_.setMean(S_.sampleUnif(normalS_.pRndGen_));
    return;
  }

  Matrix<T,Dynamic,Dynamic> Sigma = iw0_.posterior().sample();
  normalS_.setSigma(Sigma);

  // proposal distribution for normalLogMuKarch
  NormalSphere<T> normalLogKarchMu(meanKarch_, 
      Sigma/iw0_.count(),normalS_.pRndGen_);
//TODO
//      Sigma,normalS_.pRndGen_);

//  boost::uniform_01<> unif_;
//  NormalSphere<T> normalLogMuKarch(normalLogKarchMu.sample(),
//    Sigma/(kappa_+iw0_.count()),normalS_.pRndGen_);
//  for(uint32_t t=0; t<100; ++t)
//  {
//    T logPdf = normalLogMuKarch.logPdf(muPost);
//    cout<<"logPdf = "<<logPdf<<" of "
//      <<normalLogMuKarch.getMean().transpose()<<endl;
//    if(unif_(*iw0_.pRndGen_) < exp(logPdf))
//    {
//      cout<<" accepting mu = "<<normalLogMuKarch.getMean().transpose()<<endl;
//      break;
//    }
//    normalLogMuKarch.setMean(normalLogKarchMu.sample());
//  }

  // update normal on sphere from sampled new mean and sampled covariance
  NormalSphere<T> normalS(normalLogKarchMu.getMean(), Sigma, normalS_.pRndGen_);

  IW<T> iwAfter(iw0_);
  T HR = 0, logJointBefore=0, logJointAfter=0, qBefore=0, qAfter=0;

  uint32_t TT=5;
  for(uint32_t t=0; t<TT; ++t)
  {
//    cout<<"sampling from normalLogMuKarch --- karcherMean = "
//      <<muPost.transpose()
//      <<" starting from "<<normalS_.getMean()<<endl;
//    boost::uniform_01<> unif_;
//    // proposal distribution before
//    NormalSphere<T> normalLogMuMu__(normalS_.getMean(), 
//        Sigma/(kappa_+iw0_.count()),normalS_.pRndGen_);
//    // proposal distribution after
//    NormalSphere<T> normalLogMuMu(normalLogMuMu__.sample(), 
//        Sigma/(kappa_+iw0_.count()),normalS_.pRndGen_);
//    // true distri before
//    NormalSphere<T> normalLogMuKarch__(normalLogMuMu__.getMean(),
//        Sigma/(kappa_+iw0_.count()),normalS_.pRndGen_);
//    // true distri after
//    NormalSphere<T> normalLogMuKarch(normalLogMuMu.getMean(),
//        Sigma/(kappa_+iw0_.count()),normalS_.pRndGen_);
//    for(uint32_t t=0; t<10; ++t)
//    {
//      HR = normalLogMuKarch.logPdf(muPost)  // p(\bar{q}|\hat{\mu})
//        - normalLogMuKarch__.logPdf(muPost) // p(\bar{q}|\mu)
//        + normalLogMuMu__.logPdf(normalLogMuMu.getMean())  // q(\mu|\hat{\mu})
//        - normalLogMuMu.logPdf(normalLogMuMu__.getMean()); // q(\hat{\mu}|\mu)
//#ifndef NDEBUG
//      cout<<"HR: "<<exp(HR)<< " = "
//        << normalLogMuKarch.logPdf(muPost)<<" - "
//        << normalLogMuKarch__.logPdf(muPost)<<" + "
//        <<normalLogMuMu__.logPdf(normalLogMuMu.getMean()) <<" - "
//        <<normalLogMuMu.logPdf(normalLogMuMu__.getMean());
//#endif
//      if(HR > 0 || unif_(*iw0_.pRndGen_) < exp(HR))
//      {
//        normalLogMuMu__.setMean(normalLogMuMu.getMean());
//        normalLogMuKarch__.setMean(normalLogMuMu__.getMean());
//#ifndef NDEBUG
//        cout<<" -- HR="<<exp(HR)
//          <<" accepting mu = "<<normalLogMuMu.getMean().transpose()<<endl;
//      }else{
//        cout<<endl;
//#endif
//      }
//      normalLogMuMu.setMean(normalLogMuMu__.sample());
//      normalLogMuKarch.setMean(normalLogMuMu.getMean());
//    }

//    NormalSphere<T> normalLogMuKarch(normalLogKarchMu.getMean(),
//        Sigma/(kappa_+iw0_.count()),normalS_.pRndGen_);
//    Matrix<T,Dynamic,1> w(2); 
//    Matrix<T,Dynamic,Dynamic> mus(D_,w.size());
//    for(uint32_t tt=0; tt<w.size(); ++tt)
//    {
//      mus.col(tt) = normalLogKarchMu.sample();
//      w(tt) = normalLogMuKarch.logPdf(mus.col(tt)) 
//        - normalLogKarchMu.logPdf(mus.col(tt));
//    }
//    w = w.array() - logSumExp<T>(w); // normalize
//    Cat<T> cat(w,iw0_.pRndGen_);
//    normalLogMuKarch.setMean(mus.col(cat.sample()));
//    cout<<"w: "<<w.transpose()<<endl;
    NormalSphere<T> normalLogMuKarch(normalLogKarchMu.sample(),
        Sigma/iw0_.count(),normalS_.pRndGen_);
////TODO
//        Sigma,normalS_.pRndGen_);

//    normalLogMuKarch.setMean(normalLogKarchMu.sample());
//    cout<<" sampled mu = "<<normalLogMuKarch.getMean().transpose()<<endl;

    // sample from proposal distribution
    normalS.setMean(normalLogMuKarch.getMean());

    NormalSphere<T> normalN__(normalS_.getMean(), 
      normalS_.Sigma()/iw0_.count(), normalS_.pRndGen_);
//TODO
//      normalS_.Sigma(), normalS_.pRndGen_);
    NormalSphere<T> normalN(normalS.getMean(), 
      normalS.Sigma()/iw0_.count(), normalS_.pRndGen_);
//TODO
//      normalS.Sigma(), normalS_.pRndGen_);

    Matrix<T,Dynamic,1> q_mu = S_.Exp_p(normalS_.getMean(),
        S_.rotate_north2p(normalS_.getMean() ,iw0_.mean()));
    iwAfter.mean() = S_.Log_p_north(normalS.getMean(), q_mu);

    logJointBefore = normalS_.logPdfNorth(iw0_.scatter(),
        iw0_.mean(),iw0_.count());

    logJointAfter  = normalS.logPdfNorth(iwAfter.scatter(),
        iwAfter.mean(),iwAfter.count());

    //  T qBefore =  (normalN_.logPdf(normalS_.getMean()) 
    //      + iw0_.logPdf(normalS_.Sigma()));
//    qBefore =  normalN__.logPdf(normalS_.getMean());
//    qAfter  = normalN.logPdf(normalS.getMean()) ;


    qBefore =  normalN__.logPdf(meanKarch_);
    qAfter  = normalN.logPdf(meanKarch_);

//    qBefore =  normalN__.logPdf(meanKarch_);
//    qAfter  = normalN.logPdf(meanKarch_);

    HR = logJointAfter - logJointBefore + qBefore - qAfter;

    boost::uniform_01<> unif_;
    if( HR > 0 || unif_(*iw0_.pRndGen_) < exp(HR))
    {
      normalS_ = normalS;
//      cout<<"\x1b[33m -- accepted after "<<t <<" --\x1b[0m"<<endl;
    }
  }

#ifndef NDEBUG
  cout<<"NiwSphereFull<T>::sample before -----------A-------"<<endl
    <<"Sigma:"<<endl<<normalS_.Sigma()<<endl
    <<"data mu:      "<<normalS_.getMean().transpose() <<endl
//    <<" p(theta) = "<< endl
    <<" p(x|theta,z) = "<<normalS_.logPdfNorth(iw0_.scatter(),iw0_.mean(),
        iw0_.count())<<endl
//    <<" p(theta|x,z) = "<<normalN__.logPdf(normalS_.getMean()) 
//      + iwAfter.posterior().logPdf(normalS_.Sigma())
//    << " = "<<normalN__.logPdf(normalS_.getMean()) <<" + "
//      << iwAfter.posterior().logPdf(normalS_.Sigma())<<endl
    <<"karcher mean: "<<meanKarch_.transpose() <<endl
    <<"SS: count: "<<iw0_.count()<<endl
    <<"mean:    "<<iw0_.mean().transpose()<<endl
    <<iw0_.scatter()<<endl;
  cout<<"NiwSphereFull<T>::sample after --------------------"<<endl
    <<"Sigma:"<<endl<<normalS.Sigma()<<endl
    <<"data mu:      "<<normalS.getMean().transpose()<<endl
//    <<"prior Sigma:     "<<endl<<normal0.Sigma().transpose()<<endl
//    <<" p(theta) = "<< normal0.logPdf(normalS.getMean()) 
//      + iwAfter.logPdf(normalS_.Sigma())<<endl
    <<" p(x|theta,z) = "<<normalS.logPdfNorth(iwAfter.scatter(),iwAfter.mean(),
        iwAfter.count())<<endl
//    <<" p(theta|x,z) = "<<normalN.logPdf(normalS.getMean()) 
//      + iwAfter.posterior().logPdf(normalS.Sigma())
//    << " = "<<normalN.logPdf(normalS.getMean()) <<" + "
//      << iwAfter.posterior().logPdf(normalS.Sigma())<<endl
    <<"karcher mean: "<<meanKarch_.transpose() <<endl
    <<"SS: count: "<<iwAfter.count()<<endl
    <<"mean:    "<<iwAfter.mean().transpose()<<endl
    <<iwAfter.scatter()<<endl;

  cout<<"\x1b[33m HR sampleParam: "<<HR
    <<" = (logJointAfter="<<logJointAfter
    <<" - logJointBefore="<<logJointBefore
    <<" + qBefore="<<qBefore
    <<" - qAfter="<<qAfter
    <<"\x1b[0m"<<endl;
#endif

//  boost::uniform_01<> unif_;
//  if( HR > 0 || unif_(*iw0_.pRndGen_) < exp(HR))
//  {
////    normal0_ = normal0;
//    normalS_ = normalS;
//    normalN_ = normalN;
////    normalS_.setMuInTpS(iwAfter.mean());
//  cout<<"\x1b[33m accepted ------------------------------------------------------------------------- \x1b[0m"<<endl;
//  }
};


template<typename T>
void NiwSphereFull<T>::posteriorFromPtsInTpS(const Matrix<T,Dynamic,Dynamic>& x, 
    const VectorXu& z, uint32_t k, uint32_t zDivider)
{
  //TODO assumes that the mean_ has been updated before

  // sample cov from posterior; compute iw0_.scatter() and iw0_.count()
  Matrix<T,Dynamic,Dynamic> Sigma=iw0_.posterior(x,z,k,zDivider).sample();
  // TODO: if we linearize around karcher mean, we have to update the SS according
  Matrix<T,Dynamic,1> meanNew = S_.Log_p_north(normalS_.getMean(), meanKarch_);
  // get to \sum x x^T
//  iw0_.scatter() += iw0_.count()*iw0_.mean()*iw0_.mean().transpose()
//  // add in the correction
//  // remove the new mean
//    - iw0_.count() * meanNew*meanNew.transpose();
  iw0_.mean() = meanNew;

  
#ifndef NDEBUG
  cout<<"NiwSphereFull<T>::posteriorFromPtsInTpS"<<endl
    <<"posterior Sigma:"<<endl<<normalS_.Sigma()<<endl
    <<"data mean:    "<<iw0_.mean().transpose()<<endl
    <<"sampled data mu: "<<normalS_.getMean().transpose()<<endl
    <<"SS: count: "<<iw0_.count()<<endl
    <<"mean:    "<<iw0_.mean().transpose()<<endl
    <<iw0_.scatter()<<endl;
#endif
};

template<typename T>
void NiwSphereFull<T>::posterior(const Matrix<T,Dynamic,Dynamic>& q, 
    const VectorXu& z, uint32_t k)
{
  assert(false);//TODO adapt!

  Matrix<T,Dynamic,1> w(z.size()); 
  w.setZero(z.size());
#pragma omp parallel for
  for (uint32_t i=0; i<z.size(); ++i)
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
    normalS_.setSigma(iw0_.sample());
  }
//  sample();
  //  cout<<"Delta: \n"<<iw0_.posterior(x_mu,z,k).Delta_<<endl;
  //  cout<<"Sigma: \n"<<normalS_.normal_.Sigma()<<endl;
//  cout<<"Sigma Eigs:"<<normalS_.Sigma().eigenvalues()<<endl;
#ifndef NDEBUG
  cout<<"NiwSphereFull<T>::posterior"<<endl
    <<normalS_.getMean().transpose()<<endl
    <<normalS_.Sigma()<<endl;
#endif
};

template<typename T>
void NiwSphereFull<T>::posterior( const shared_ptr<ClData<T> >& cldp, uint32_t k)
{
  assert(false);//TODO adapt!

//  normalS_.setMean(cldp->mean(k));
//  normalS_.normal_ = Normal<T>(iw0_.posterior(cldp->S(k),cldp->count(k)).sample(),normalS_.pRndGen_);
//#ifndef NDEBUG
//  cout<<"NiwSphereFull<T>::posterior"<<endl
//    <<normalS_.getMean().transpose()<<endl
//    <<normalS_.Sigma()<<endl;
//#endif
};

template<typename T>
T NiwSphereFull<T>::logPdfUnderPrior() const
{
  T logPdf = normalS_.logPdfNorth(iw0_.scatter(),iw0_.mean(),iw0_.count()) 
    + iw0_.logPdf(normalS_.Sigma()) 
    - S_.logSurfaceArea();
//    normalS_.logPdfNorth(iw0_.scatter(),iw0_.mean(),iw0_.count()) 
//    + iw0_.logPdf(normalS_.Sigma()) 
//    + normal0_.logPdf(normalS_.getMean());
//    + iw0_.logPdf(normalS_.Sigma()) 

// TODO prior for normal onto sphere
#ifndef NDEBUG
  cout<<"count = "<<iw0_.count()<<endl;
  cout<<"scatter"<<endl<<iw0_.scatter()<<endl;
  cout<<"mean in TpS"<<iw0_.mean().transpose()<<endl;
  cout<<"sigma="<<endl<<normalS_.Sigma()<<endl;
  cout<<"mean in S"<<normalS_.getMean().transpose()<<endl;
#endif
  return logPdf;
};

template<typename T>
T NiwSphereFull<T>::logPdfUnderPriorMarginalized() const
{
//  return logPdfUnderPrior();
// cannot really fully marginalize ... marginalize over Sigma and multiply 
// by uniform distri over mus
  return iw0_.logLikelihoodMarginalized() - S_.logSurfaceArea();
//  return iw0_.logLikelihoodMarginalized()
//    - S_.logSurfaceArea();
//

};

template<typename T>
T NiwSphereFull<T>::qRandomMuProposal() const
{
  NormalSphere<T> mergeProposal(meanKarch_, iw0_.posterior().mode(),
    iw0_.pRndGen_);
//  cout<<"NiwSphereFull<T>::qRandomMuProposal() eval mean "
//    <<normalS_.getMean().transpose()<<endl
//    <<" around karcher mean "<<meanKarch_.transpose()<<endl;
  return mergeProposal.logPdf(normalS_.getMean());
};


template<typename T>
T NiwSphereFull<T>::logPdfUnderPriorMarginalizedMerged(
    const shared_ptr<NiwSphereFull<T> >& other) const
{
  assert(false); //TODO updated - not used right now anyways

  IW<T> iwM(iw0_);
  Matrix<T,Dynamic,1> muMerged(iw0_.D_);
  this->computeMergedSS(*this, *other, iwM.scatter(), iwM.mean(), muMerged, 
      iwM.count());

//  // sample Sigma and mu
//  Matrix<T,Dynamic,Dynamic> Sigma = iwM.posterior().sample();
//  // posterior distribution for mean
//  NormalSphere<T> normalN(muMerged,Sigma/(kappa_+iwM.count()),normalS_.pRndGen_);
//  Matrix<T,Dynamic,1> muSampled = normalN.sample();
//  // distribution of data
//  NormalSphere<T> normalS(muSampled,Sigma,normalS_.pRndGen_);
//  // prior distribution on mean
//  NormalSphere<T> normal0(normal0_.getMean(), Sigma/kappa_,normalS_.pRndGen_);

#ifndef NDEBUG
  cout<<"NiwSphereFull<T>::logPdfUnderPriorMarginalizedMerged"<<endl
//    <<"posterior Sigma:"<<endl<<Sigma<<endl
//    <<"prior mu:     "<<normal0_.getMean().transpose()<<endl
    <<"data mean:    "<<muMerged<<endl
//    <<"posterior mu: "<<normalN.getMean().transpose()<<endl
//    <<"sampled data mu: "<<normalS.getMean().transpose()<<endl
    <<"SS: count: "<<iwM.count()<<endl
    <<"mean:    "<<muMerged.transpose()<<endl
    <<iwM.scatter()<<endl;
#endif

//  return normalS.logPdfNorth(iwM.scatter(),iwM.mean(),iwM.count()) 
//    + iw0_.logPdf(normalS.Sigma()) + normal0.logPdf(normalS.getMean());
//
//  //TODO
  return iwM.logLikelihoodMarginalized() - S_.logSurfaceArea();

//  return normalS.logPdfNorth(scatterMerged,countMerged) 
//    + iw0_.logPdf(Sigma) + normal0.logPdf(muSampled)
//    - iw0_.posterior(scatterMerged,countMerged).logPdf(normalS.Sigma())
//    - normalN.logPdf(normalS.getMean());

  // TODO bad in two ways:
  // not marginalized
  // and I am not going to use the same muSampled in case merge is accepted!
//  return iw0_.logPdf(Sigma) + normal0.logPdf(muSampled);
};

template<typename T>
void NiwSphereFull<T>::computeMergedSS( const NiwSphereFull<T>& niwA, 
    const NiwSphereFull<T>& niwB, Matrix<T,Dynamic,Dynamic>& scatterM, 
    Matrix<T,Dynamic,1>& meanM,
    Matrix<T,Dynamic,1>& muM, T& countM) const
{
  countM = niwA.count() + niwB.count();
#ifndef NDEBUG
  cout<<countM<<" "<<niwA.count()<<" "<<niwB.count()<<endl;
#endif
  if(countM == 0)
  {
    scatterM.setZero();
    muM = niwA.getMeanKarch();
    return;
  }

  // instead of using iterative karcher means I can compute the weighted mean 
  // simply moving along the geodesic path from meanA to meanB.
  // TODO: this one uses raw statistics -> should not do that
//  muM = rotationFromAtoB<T>(niwA.getMeanKarch(), niwB.getMeanKarch(), 
//      niwB.count()/countM)*niwA.getMeanKarch();
  muM = rotationFromAtoB<T>(niwA.getMean(), niwB.getMean(), 
      niwB.count()/countM)*niwA.getMean();
//  cout<<this->normalS_.mu_<<endl;
//  assert(false);

//  cout<<muM.transpose()<<endl;
//  cout<<niwA.getMean().transpose()<<endl;
//  cout<<niwB.getMean().transpose()<<endl;
//  Matrix<T,Dynamic,1> muAinTthis = S_.Log_p_north(muM, niwA.getMean());
//  Matrix<T,Dynamic,1> muBinTthis = S_.Log_p_north(muM, niwB.getMean());

  Matrix<T,Dynamic,1> meanA = S_.Exp_p_single(niwA.getMean(),
      S_.rotate_north2p(niwA.getMean(),niwA.mean())); // on sphere
  Matrix<T,Dynamic,1> meanB = S_.Exp_p_single(niwB.getMean(),
      S_.rotate_north2p(niwB.getMean(),niwB.mean())); // on sphere

//  meanA = niwA.getMean();
//  meanB = niwB.getMean();

  Matrix<T,Dynamic,1> muAinTthis = S_.Log_p_north(muM, meanA);
  Matrix<T,Dynamic,1> muBinTthis = S_.Log_p_north(muM, meanB);

  Matrix<T,Dynamic,1> muAinTthis2 = S_.Log_p_north(muM, niwA.getMean());
  Matrix<T,Dynamic,1> muBinTthis2 = S_.Log_p_north(muM, niwB.getMean());

#ifndef NDEBUG
  cout<<muAinTthis.transpose()<<" instead of "<<muAinTthis2.transpose()<<endl;
  cout<<muBinTthis.transpose()<<" instead of "<<muBinTthis2.transpose()<<endl;
#endif

  //TODO could also compute this as a mean on the sphere and then map into TpS
  meanM = (niwA.count()*muAinTthis + niwB.count()*muBinTthis) / countM;

  // add xx^T s for both clusters
  scatterM  = niwA.scatter() + niwA.count()*niwA.mean()*niwA.mean().transpose();
  scatterM += niwB.scatter() + niwB.count()*niwB.mean()*niwB.mean().transpose();
  // apply correction for new means to get new outer product in merged space
  scatterM += niwA.count()*muAinTthis2*muAinTthis2.transpose();
  scatterM += niwB.count()*muBinTthis2*muBinTthis2.transpose();
  // subtract mean in merged space to arrive at correct scatter 
  // \sum (x_i-x)(x_i-x)^T
  scatterM -= countM * meanM*meanM.transpose();

};



template<typename T>
void NiwSphereFull<T>::print() const
{
  cout<<"mu in S^D:  "<<normalS_.getMean().transpose()<<endl;
  cout<<"Sigma:      "<<endl<<normalS_.Sigma()<<endl;
  cout<<"mu in TpS:  "<<normalS_.normal().mu_.transpose()<<endl;
  cout<<"mean in TpS:"<<mean().transpose()<<endl;
  cout<<"Scatter:    "<<endl<<scatter()<<endl;
  cout<<"count:      "<<count()<<endl;
};

template<typename T>
NiwSphereFull<T>* NiwSphereFull<T>::merge(const NiwSphereFull<T>& other)
{
  NiwSphereFull<T>* merged = this->copyNative();
  merged->fromMerge(*this,other);
  merged->sampleMergedParams(); 
  return merged;
};

template<typename T>
void NiwSphereFull<T>::fromMerge(const NiwSphereFull<T>& niwA, const NiwSphereFull<T>& niwB)
{
  Matrix<T,Dynamic,1> meanKarchM;
  IW<T> iwM(iw0_);
  // pass in a different IW object than my own since I might be niwA or niwB
  computeMergedSS(niwA, niwB, iwM.scatter(), iwM.mean(), meanKarchM,
      iwM.count());
  iw0_ = iwM;
  meanKarch_ = meanKarchM;

#ifndef NDEBUG
  cout<<"NiwSphereFull<T>::fromMerge"<<endl
    <<"SS: count: "<<iw0_.count()<<endl
    <<"mean karch: "<<meanKarch_.transpose()<<endl
    <<"mean inTpS: "<<iw0_.mean().transpose()<<endl
    <<"Scatter:    "<<endl<<iw0_.scatter()<<endl;
#endif
};

// ---------------------------------------------------------------------------
template class NiwSphereFull<double>;
template class NiwSphereFull<float>;

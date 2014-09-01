#include "niwSphere.hpp"
// ----------------------------------------------------------------------------

template<typename T>
NiwSphere<T>::NiwSphere(const IW<T>& iw, boost::mt19937* pRndGen)
  : iw0_(iw), S_(iw.D_+1), normalS_(S_.sampleUnif(pRndGen),iw0_.sample(),pRndGen)
{
  qqTSum_.setZero(iw0_.D_+1,iw0_.D_+1);
  qqTAngSum_.setZero(iw0_.D_+1,iw0_.D_+1);
  qqTAngSqSum_.setZero(iw0_.D_+1,iw0_.D_+1);
  qSum_.setZero(iw0_.D_+1);
  qSumAngle_.setZero(iw0_.D_+1);
  qSumAngleSq_.setZero(iw0_.D_+1);
  sumAngle_ = 0;
  sumAngleSq_ = 0;
};

template<typename T>
NiwSphere<T>::~NiwSphere()
{};

template<typename T>
BaseMeasure<T>* NiwSphere<T>::copy()
{
  NiwSphere<T>* niwSp = new NiwSphere<T>(iw0_,normalS_.pRndGen_);
  niwSp->normalS_ = normalS_;

  //extension
  niwSp->qqTSum_ = qqTSum_;
  niwSp->qqTAngSum_ = qqTAngSum_;
  niwSp->qqTAngSqSum_ = qqTAngSqSum_;
  niwSp->qSum_ = qSum_;
  niwSp->qSumAngle_ = qSumAngle_;
  niwSp->qSumAngleSq_ = qSumAngleSq_;
  niwSp->sumAngle_ = sumAngle_;
  niwSp->sumAngleSq_ = sumAngleSq_;
  return niwSp;
};

template<typename T>
NiwSphere<T>* NiwSphere<T>::copyNative()
{
  NiwSphere<T>* niwSp = new NiwSphere<T>(iw0_,normalS_.pRndGen_);
  niwSp->normalS_ = normalS_;

  //extension
  niwSp->qqTSum_ = qqTSum_;
  niwSp->qqTAngSum_ = qqTAngSum_;
  niwSp->qqTAngSqSum_ = qqTAngSqSum_;
  niwSp->qSum_ = qSum_;
  niwSp->qSumAngle_ = qSumAngle_;
  niwSp->qSumAngleSq_ = qSumAngleSq_;
  niwSp->sumAngle_ = sumAngle_;
  niwSp->sumAngleSq_ = sumAngleSq_;
  return niwSp;
};

template<typename T>
T NiwSphere<T>::logLikelihood(const Matrix<T,Dynamic,1>& q) const
{
//  cout<<" NiwSphere<T>::logLikelihood: "<<normalS_.logPdf(x)<<endl;
  return normalS_.logPdf(q);
};

template<typename T>
T NiwSphere<T>::logLikelihoodNorth(const Matrix<T,Dynamic,1>& x) const
{
//  cout<<" NiwSphere<T>::logLikelihood: "<<normalS_.logPdf(x)<<endl;
  return normalS_.logPdfNorth(x);
};

template<typename T>
void NiwSphere<T>::sample()
{
//cout<<iw0_.scatter()<<endl;
//cout<<iw0_.count()<<endl;
  normalS_.setSigma(iw0_.posterior().sample());
};

template<typename T>
void NiwSphere<T>::posteriorFromPtsInTpS(const Matrix<T,Dynamic,Dynamic>& x, 
    const VectorXu& z, uint32_t k)
{
//  Matrix<T,Dynamic,Dynamic> x_mu = S_.Log_p_north(normalS_.mu_,x);
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
  
    // extension 
  qqTSum_.setZero(iw0_.D_+1,iw0_.D_+1);
  qqTAngSum_.setZero(iw0_.D_+1,iw0_.D_+1);
  qqTAngSqSum_.setZero(iw0_.D_+1,iw0_.D_+1);
  qSum_.setZero(iw0_.D_+1);
  qSumAngle_.setZero(iw0_.D_+1);
  qSumAngleSq_.setZero(iw0_.D_+1);
  sumAngle_ = 0;
  sumAngleSq_ = 0;
#pragma omp parallel for
  for (uint32_t i=0; i<z.size(); ++i)
    if(z[i] == k)
    {
      T theta = x_mu.col(i).norm();
      qqTSum_       += q.col(i)*q.col(i).transpose();
      qqTAngSum_    += q.col(i)*q.col(i).transpose()*theta;
      qqTAngSqSum_  += q.col(i)*q.col(i).transpose()*theta*theta;
      qSum_         += q.col(i);
      qSumAngle_    += q.col(i)*theta;
      qSumAngleSq_  += q.col(i)*theta*theta;
      sumAngle_     += theta;
      sumAngleSq_   += theta*theta;
    }
   

  }else{
    normalS_.setMean(S_.sampleUnif(normalS_.pRndGen_));
    normalS_.setSigma(iw0_.sample());

    // extension 
  qqTSum_.setZero(iw0_.D_+1,iw0_.D_+1);
  qqTAngSum_.setZero(iw0_.D_+1,iw0_.D_+1);
  qqTAngSqSum_.setZero(iw0_.D_+1,iw0_.D_+1);
  qSum_.setZero(iw0_.D_+1);
  qSumAngle_.setZero(iw0_.D_+1);
  qSumAngleSq_.setZero(iw0_.D_+1);
  sumAngle_ = 0;
  sumAngleSq_ = 0;
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
void NiwSphere<T>::posterior( const boost::shared_ptr<ClData<T> >& cldp, uint32_t k)
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
    const boost::shared_ptr<NiwSphere<T> >& other) const
{
  Matrix<T,Dynamic,Dynamic> scatterMerged(iw0_.D_,iw0_.D_);
  Matrix<T,Dynamic,1> muMerged(iw0_.D_);
  T countMerged=0;
  this->computeMergedSS(*this, *other, scatterMerged, muMerged, countMerged);
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

  //extension
//  scatterM += niwA.scatterCorrection1(muM);
//  scatterM += niwB.scatterCorrection1(muM);

#ifndef NDEBUG
  cout<<"scatter after correction"<<endl<<scatterM<<endl;
  cout<<"logP under Prior: "<<iw0_.logLikelihoodMarginalized(scatterM, countM)<<endl;
#endif
};

template<typename T>
Matrix<T,Dynamic,Dynamic> NiwSphere<T>::scatterCorrection1(
    const Matrix<T,Dynamic,1>& p) const
{
  // rotate everything up to north to work in the same tangent space as the SS 
  // are computed
  Matrix<T,Dynamic,Dynamic> northR = rotationFromAtoB(p,S_.north());
  Matrix<T,Dynamic,1> muNorth = northR * getMean();
  Matrix<T,Dynamic,1> pNorth = northR*p;
  Matrix<T,Dynamic,1> mu_p = S_.Log_p_single(pNorth,muNorth);
  Matrix<T,Dynamic,1> qSumNorth = northR*qSum_;
  Matrix<T,Dynamic,1> qSumAngleNorth = northR * qSumAngle_;
  Matrix<T,Dynamic,1> qSumAngleSqNorth = northR * qSumAngleSq_;

  T dot = pNorth.transpose()*muNorth;
  T theta_pmu = acos(min(static_cast<T>(1.0),max(static_cast<T>(-1.0), dot)));
 
  T invSinc = 0; 
  T b = 0; 
  T a2 = 0;
  if(fabs(theta_pmu) < 1e-4)
  {
    invSinc = 1.;
    // tayolr series around theta_pmu=0
    b = -2.*theta_pmu/3.-4*theta_pmu*theta_pmu*theta_pmu/45.;
    a2 = -theta_pmu/3. - 7.*theta_pmu*theta_pmu*theta_pmu/90.;
  }else{
    invSinc = theta_pmu/sin(theta_pmu);
    T tanTheta = tan(theta_pmu);
    b = -theta_pmu  + 1./tanTheta - theta_pmu/(tanTheta*tanTheta);
    a2 = -1./sin(theta_pmu) + invSinc/tanTheta;
  }
//  cout<<"theta_pmu: "<<theta_pmu<<endl;
  T a1 = invSinc -1;
  T c = 1. - invSinc;

//  cout<<"mu_p:     \t"<<mu_p.transpose()<<endl;
//  cout<<"muNorth:  \t"<<muNorth.transpose()<<endl;
//  cout<<"pNorth:   \t"<<pNorth.transpose()<<endl;
//  cout<<"a1="<<a1<<" a2="<<a2<<" b="<<b<<" c="<<c<<endl;
//  cout<<"qSum:     \t"<<qSumNorth.transpose()<<endl;
//  cout<<"qSumAngle:\t"<<qSumAngleNorth.transpose()<<endl;
//  cout<<"sumAngle: \t"<<sumAngle_<<endl;

  Matrix<T,Dynamic,1> sumDelta = 
    a1*qSumNorth 
    + a2*qSumAngleNorth 
    + b*sumAngle_*pNorth 
    + iw0_.count()*c*muNorth;
  Matrix<T,Dynamic,Dynamic> Sc = 2.*sumDelta*mu_p.transpose();
  cout<<"NiwSphere<T>::scatterCorrection1"<<endl<<Sc<<endl;


//  cout<<qqTAngSum_<<endl;
//  cout<<northR*qqTAngSum_*northR.transpose()<<endl;
//

  Matrix<T,Dynamic,Dynamic> sumDeltaDelta = 
    a1*northR*qqTSum_*northR.transpose()
    + a2*northR*qqTAngSqSum_*northR.transpose()
    + b*sumAngleSq_*pNorth*pNorth.transpose()
    + c*iw0_.count()* muNorth*muNorth.transpose()
    + 2.*a1*a2*northR*qqTAngSum_*northR.transpose()
    + 2.*a1*b*qSumAngleNorth*pNorth.transpose()
    + 2.*a1*c*qSumNorth*muNorth.transpose()
    + 2.*a2*b*qSumAngleSqNorth*pNorth.transpose()
    + 2.*a2*c*qSumAngleNorth*muNorth.transpose()
    + 2.*b*c*sumAngle_*pNorth*muNorth.transpose();
  Sc +=sumDeltaDelta;
  cout<<"NiwSphere<T>::scatterCorrection2"<<endl<<Sc<<endl;

//  Sc = 0.5*(Sc+Sc.transpose()); // make symmetric
//  cout<<"sumDelta:    \t"<<sumDelta.transpose()<<endl;
//  cout<<"NiwSphere<T>::scatterCorrection1"<<endl<<Sc<<endl;
  return Sc.topLeftCorner(iw0_.D_,iw0_.D_);
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
//  Matrix<T,Dynamic,Dynamic> q(normalS_.mu_.rows(),2);
//  Matrix<T,Dynamic,1> w(2);
//  q<<normalS_.mu_,other.normalS_.mu_;
//  w<<iw0_.count(), other.count();
//  Matrix<T,Dynamic,1> muNew = karcherMeanWeighted<T>(normalS_.mu_,q,w,100);
//  Matrix<T,Dynamic,1> muThisAtMuNew = S_.Log_p_north(muNew, normalS_.mu_);
//  Matrix<T,Dynamic,1> muOtherAtMuNew = S_.Log_p_north(muNew, other.normalS_.mu_);
  
  NiwSphere<T>* merged = this->copyNative();
  merged->fromMerge(*this,other);
//
//  merged->scatter() += other.scatter();
//  merged->scatter() += iw0_.count()*muThisAtMuNew*muThisAtMuNew.transpose();
//  merged->scatter() += other.count()*muOtherAtMuNew*muOtherAtMuNew.transpose();
//  merged->count() += other.count();
//
//  merged->normalS_.normal_ = Normal<T>(iw0_.posterior(iw0_.scatter(),iw0_.count()).sample(),
//      normalS_.pRndGen_);
//  merged->normalS_.mu_ = muNew;
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

//template<typename T>
//const NiwSphere& NiwSphere<T>::split(const Matrix<T,Dynamic,Dynamic>& x, 
//    const VectorXu& z, uint32_t k, uint32_t j)
//{
//  this->posterior(x,z,k);
//  
//};
// ---------------------------------------------------------------------------
template class NiwSphere<double>;
template class NiwSphere<float>;

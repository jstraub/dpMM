
#include "niwTangent.hpp"
// ----------------------------------------------------------------------------

template<typename T>
NiwTangent<T>::NiwTangent(const NIW<T>& niw, boost::mt19937* pRndGen)
  : niw0_(niw), S_(niw.D_+1), 
  normalS_(S_.sampleUnif(pRndGen),niw0_.sample(),pRndGen),
  scatter_(Matrix<T,Dynamic,Dynamic>::Zero(niw.D_,niw.D_)), count_(0)
{
  qqTSum_.setZero(niw0_.D_+1,niw0_.D_+1);
  qqTAngSum_.setZero(niw0_.D_+1,niw0_.D_+1);
  qqTAngSqSum_.setZero(niw0_.D_+1,niw0_.D_+1);
  qSum_.setZero(niw0_.D_+1);
  qSumAngle_.setZero(niw0_.D_+1);
  qSumAngleSq_.setZero(niw0_.D_+1);
  sumAngle_ = 0;
  sumAngleSq_ = 0;
};

template<typename T>
NiwTangent<T>::~NiwTangent()
{};

template<typename T>
BaseMeasure<T>* NiwTangent<T>::copy()
{
  NiwTangent<T>* niwSp = new NiwTangent<T>(niw0_,normalS_.pRndGen_);
  niwSp->normalS_ = normalS_;
  niwSp->scatter() = this->scatter();
  niwSp->count() = this->count();

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
NiwTangent<T>* NiwTangent<T>::copyNative()
{
  NiwTangent<T>* niwSp = new NiwTangent<T>(niw0_,normalS_.pRndGen_);
  niwSp->normalS_ = normalS_;
  niwSp->scatter() = this->scatter();
  niwSp->count() = this->count();

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
T NiwTangent<T>::logLikelihood(const Matrix<T,Dynamic,1>& q) const
{
//  cout<<" NiwTangent<T>::logLikelihood: "<<normalS_.logPdf(x)<<endl;
  return normalS_.logPdf(q);
};

template<typename T>
T NiwTangent<T>::logLikelihoodNorth(const Matrix<T,Dynamic,1>& x) const
{
//  cout<<" NiwTangent<T>::logLikelihood: "<<normalS_.logPdf(x)<<endl;
  return normalS_.logPdfNorth(x);
};

template<typename T>
void NiwTangent<T>::sample()
{
//cout<<scatter_<<endl;
//cout<<count_<<endl;
  normalS_.setNormal(niw0_.posterior().sample());
};

template<typename T>
void NiwTangent<T>::posteriorFromPtsInTpS(const Matrix<T,Dynamic,Dynamic>& x, 
    const VectorXu& z, uint32_t k)
{
//  Matrix<T,Dynamic,Dynamic> x_mu = S_.Log_p_north(normalS_.mu_,x);
  normalS_.setNormal(niw0_.posterior(x,z,k).sample());

#ifndef NDEBUG
  cout<<"NiwTangent<T>::posteriorFromPtsInTpS"<<endl;
  print();
#endif
};

template<typename T>
void NiwTangent<T>::posterior(const Matrix<T,Dynamic,Dynamic>& q, 
    const VectorXu& z, uint32_t k)
{
//  Matrix<T,Dynamic,1> w(z.size()); 
//  w.setZero(z.size());
//#pragma omp parallel for
//  for (uint32_t i=0; i<z.size(); ++i)
//    if(z[i] == k) w[i]=1.0;
//  if(w.sum() > 0)
//  {
//    normalS_.setMean(karcherMeanWeighted<T>(normalS_.getMean(), q, w, 100));
    //TODO: slight permutation here for mu to allow proper sampling
    // TODO: wastefull since it computes stuff for normals that are not used in the 
    // later computations
    Matrix<T,Dynamic,Dynamic> x_mu = S_.Log_p_north(normalS_.getMean(),q);
    normalS_.setNormal(niw0_.posterior(x_mu,z,k).sample());
  
    // extension 
  qqTSum_.setZero(niw0_.D_+1,niw0_.D_+1);
  qqTAngSum_.setZero(niw0_.D_+1,niw0_.D_+1);
  qqTAngSqSum_.setZero(niw0_.D_+1,niw0_.D_+1);
  qSum_.setZero(niw0_.D_+1);
  qSumAngle_.setZero(niw0_.D_+1);
  qSumAngleSq_.setZero(niw0_.D_+1);
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
   

//  }else{
////    normalS_.setMean(S_.sampleUnif(normalS_.pRndGen_));
//    normalS_.normal_ = Normal<T>(niw0_.sample(),normalS_.pRndGen_);
//    scatter_.setZero(niw0_.D_,niw0_.D_);
//    count_ = 0;
//
//    // extension 
//  qqTSum_.setZero(niw0_.D_+1,niw0_.D_+1);
//  qqTAngSum_.setZero(niw0_.D_+1,niw0_.D_+1);
//  qqTAngSqSum_.setZero(niw0_.D_+1,niw0_.D_+1);
//  qSum_.setZero(niw0_.D_+1);
//  qSumAngle_.setZero(niw0_.D_+1);
//  qSumAngleSq_.setZero(niw0_.D_+1);
//  sumAngle_ = 0;
//  sumAngleSq_ = 0;
//  }
//  sample();
  //  cout<<"Delta: \n"<<niw0_.posterior(x_mu,z,k).Delta_<<endl;
  //  cout<<"Sigma: \n"<<normalS_.normal_.Sigma_<<endl;
//  cout<<"Sigma Eigs:"<<normalS_.normal_.Sigma_.eigenvalues()<<endl;
#ifndef NDEBUG
  cout<<"NiwTangent<T>::posterior"<<endl
    <<normalS_.getMean().transpose()<<endl
    <<normalS_.Sigma()<<endl;
#endif
};

template<typename T>
void NiwTangent<T>::posterior( const boost::shared_ptr<ClData<T> >& cldp, uint32_t k)
{
  assert(false);
//  normalS_.setMean(cldp->mean(k));
//  normalS_.normal_ = Normal<T>(niw0_.posterior(cldp->S(k),cldp->count(k)).sample(),normalS_.pRndGen_);
//#ifndef NDEBUG
//  cout<<"NiwTangent<T>::posterior"<<endl
//    <<normalS_.getMean().transpose()<<endl
//    <<normalS_.normal_.Sigma_<<endl;
//#endif
};

template<typename T>
T NiwTangent<T>::logPdfUnderPrior() const
{
//TODO prior for normal onto sphere
  return niw0_.logPdf(normalS_.normal());
};

template<typename T>
T NiwTangent<T>::logPdfUnderPriorMarginalized() const
{
//TODO prior for normal onto sphere
  return niw0_.logPdfMarginalized();
};

template<typename T>
T NiwTangent<T>::logPdfUnderPriorMarginalizedMerged(
    const boost::shared_ptr<NiwTangent<T> >& other) const
{
  Matrix<T,Dynamic,Dynamic> scatterMerged(niw0_.D_,niw0_.D_);
  Matrix<T,Dynamic,1> pMerged(niw0_.D_+1);
  Matrix<T,Dynamic,1> meanMerged(niw0_.D_);
  T countMerged=0;
  this->computeMergedSS(*this, *other, scatterMerged, meanMerged, pMerged, 
      countMerged);
  return niw0_.logLikelihoodMarginalized(scatterMerged, meanMerged, countMerged);
};

template<typename T>
void NiwTangent<T>::computeMergedSS( const NiwTangent<T>& niwA, 
    const NiwTangent<T>& niwB, Matrix<T,Dynamic,Dynamic>& scatterM, 
    Matrix<T,Dynamic,1>& meanM,
    Matrix<T,Dynamic,1>& pM, T& countM) const
{
  countM = niwA.count() + niwB.count();
#ifndef NDEBUG
  cout<<countM<<" "<<niwA.count()<<" "<<niwB.count()<<endl;
#endif
  if(countM == 0)
  {
    scatterM.setZero();
    meanM.setZero();
    pM = niwA.getMean();
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
  pM = rotationFromAtoB<T>(niwA.getMean(), niwB.getMean(), 
      niwB.count()/countM)*niwA.getMean();
//  cout<<this->normalS_.mu_<<endl;
//  assert(false);

//  cout<<muM.transpose()<<endl;
//  cout<<niwA.getMean().transpose()<<endl;
//  cout<<niwB.getMean().transpose()<<endl;
//  Matrix<T,Dynamic,1> muAinTthis = S_.Log_p_north(pM, niwA.getMean());
//  Matrix<T,Dynamic,1> muBinTthis = S_.Log_p_north(pM, niwB.getMean());
  Matrix<T,Dynamic,1> muA = S_.Exp_p_single(niwA.getMean(),
      S_.rotate_north2p(niwA.getMean(),niwA.normalS_.normal().mu_));
  cout<<muA.transpose()<<endl;
  Matrix<T,Dynamic,1> muB = S_.Exp_p_single(niwB.getMean(),
      S_.rotate_north2p(niwB.getMean(),niwB.normalS_.normal().mu_));
  Matrix<T,Dynamic,1> muAinTthis = S_.Log_p_north(pM, muA);
  Matrix<T,Dynamic,1> muBinTthis = S_.Log_p_north(pM, muB);

//  scatterM = niwA.scatter() + niwB.scatter();
//  scatterM += niwA.count()*muAinTthis*muAinTthis.transpose();
//  scatterM += niwB.count()*muBinTthis*muBinTthis.transpose();

  NIW<T> niwATpS(niw0_);
  niwATpS.scatter() = niwA.scatter()+niwA.count()*muAinTthis*muAinTthis.transpose();
//  niwATpS.scatter() += niwA.scatterCorrection1(pM);
  niwATpS.mean() = muAinTthis;
  niwATpS.count() = niwA.count();
  

  NIW<T> niwBTpS(niw0_);
  niwBTpS.scatter() = niwB.scatter()+niwB.count()*muBinTthis*muBinTthis.transpose();
//  niwBTpS.scatter() += niwB.scatterCorrection1(pM);
  niwBTpS.mean() = muBinTthis;
  niwBTpS.count() = niwB.count();

  niwA.niw0_.computeMergedSS(niwATpS,niwBTpS,scatterM, meanM, countM);
  cout<<"NiwTangent<T>::computeMergedSS"<<endl;
  cout<<"TpS at "<<pM.transpose()<<endl;
  cout<<"mean Left  "<<niwATpS.mean().transpose()<<endl;
  cout<<"mean Right "<<niwBTpS.mean().transpose()<<endl;
  cout<<"mean "<<meanM.transpose()<<endl;
  cout<<"scatter Left  "<<endl<<niwATpS.scatter()<<endl;
  cout<<"scatter Right "<<endl<<niwBTpS.scatter()<<endl;
  cout<<"scatter "<<endl<<scatterM<<endl;
  cout<<"count "<<countM<<endl;
//#ifndef NDEBUG
//  cout<<"scatter before correction"<<endl<<scatterM<<endl;
//  cout<<"logP under Prior: "<<niw0_.logLikelihoodMarginalized(scatterM, countM)<<endl;
//#endif

  //extension
//  scatterM += niwA.scatterCorrection1(muM);
//  scatterM += niwB.scatterCorrection1(muM);

//#ifndef NDEBUG
//  cout<<"scatter after correction"<<endl<<scatterM<<endl;
//  cout<<"logP under Prior: "<<niw0_.logLikelihoodMarginalized(scatterM, countM)<<endl;
//#endif
};

template<typename T>
Matrix<T,Dynamic,Dynamic> NiwTangent<T>::scatterCorrection1(
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
    + count_*c*muNorth;
  Matrix<T,Dynamic,Dynamic> Sc = 2.*sumDelta*mu_p.transpose();
  cout<<"NiwTangent<T>::scatterCorrection1"<<endl<<Sc<<endl;


//  cout<<qqTAngSum_<<endl;
//  cout<<northR*qqTAngSum_*northR.transpose()<<endl;
//

  Matrix<T,Dynamic,Dynamic> sumDeltaDelta = 
    a1*northR*qqTSum_*northR.transpose()
    + a2*northR*qqTAngSqSum_*northR.transpose()
    + b*sumAngleSq_*pNorth*pNorth.transpose()
    + c*count_* muNorth*muNorth.transpose()
    + 2.*a1*a2*northR*qqTAngSum_*northR.transpose()
    + 2.*a1*b*qSumAngleNorth*pNorth.transpose()
    + 2.*a1*c*qSumNorth*muNorth.transpose()
    + 2.*a2*b*qSumAngleSqNorth*pNorth.transpose()
    + 2.*a2*c*qSumAngleNorth*muNorth.transpose()
    + 2.*b*c*sumAngle_*pNorth*muNorth.transpose();
  Sc +=sumDeltaDelta;
  cout<<"NiwTangent<T>::scatterCorrection2"<<endl<<Sc<<endl;

//  Sc = 0.5*(Sc+Sc.transpose()); // make symmetric
//  cout<<"sumDelta:    \t"<<sumDelta.transpose()<<endl;
//  cout<<"NiwTangent<T>::scatterCorrection1"<<endl<<Sc<<endl;
  return Sc.topLeftCorner(niw0_.D_,niw0_.D_);
};

template<typename T>
void NiwTangent<T>::print() const
{
    cout<<"Tps around: "<<normalS_.getMean().transpose()<<endl
    <<"Sigma: "<<endl<<normalS_.Sigma()<<endl
    <<"Count: "<<niw0_.count()<<endl
    <<"mu in TpS: "<<normalS_.normal().mu_.transpose()<<endl
    <<"Scatter: "<<endl<<niw0_.scatter()<<endl;
};

template<typename T>
NiwTangent<T>* NiwTangent<T>::merge(const NiwTangent<T>& other)
{
//  Matrix<T,Dynamic,Dynamic> q(normalS_.mu_.rows(),2);
//  Matrix<T,Dynamic,1> w(2);
//  q<<normalS_.mu_,other.normalS_.mu_;
//  w<<count_, other.count();
//  Matrix<T,Dynamic,1> muNew = karcherMeanWeighted<T>(normalS_.mu_,q,w,100);
//  Matrix<T,Dynamic,1> muThisAtMuNew = S_.Log_p_north(muNew, normalS_.mu_);
//  Matrix<T,Dynamic,1> muOtherAtMuNew = S_.Log_p_north(muNew, other.normalS_.mu_);
  
  NiwTangent<T>* merged = this->copyNative();
  merged->fromMerge(*this,other);
//
//  merged->scatter() += other.scatter();
//  merged->scatter() += count_*muThisAtMuNew*muThisAtMuNew.transpose();
//  merged->scatter() += other.count()*muOtherAtMuNew*muOtherAtMuNew.transpose();
//  merged->count() += other.count();
//
//  merged->normalS_.normal_ = Normal<T>(niw0_.posterior(scatter_,count_).sample(),
//      normalS_.pRndGen_);
//  merged->normalS_.mu_ = muNew;
  return merged;
};

template<typename T>
void NiwTangent<T>::fromMerge(const NiwTangent<T>& niwA, const NiwTangent<T>& niwB)
{
  Matrix<T,Dynamic,1> pM(niw0_.D_+1);
  Matrix<T,Dynamic,1> meanM(niw0_.D_);
  Matrix<T,Dynamic,Dynamic> scatterM(niw0_.D_,niw0_.D_);
  T countM = 0;
  computeMergedSS(niwA, niwB, scatterM, meanM, pM, countM);
  setMean(pM);
  niw0_.scatter() = scatterM;
  niw0_.count() = countM;
  niw0_.mean() = meanM;

  normalS_.setNormal(niw0_.posterior().sample());
  cout<<"NiwTangent<T>::fromMerge"<<endl;
  print();
};

//template<typename T>
//const NiwTangent& NiwTangent<T>::split(const Matrix<T,Dynamic,Dynamic>& x, 
//    const VectorXu& z, uint32_t k, uint32_t j)
//{
//  this->posterior(x,z,k);
//  
//};
// ---------------------------------------------------------------------------
template class NiwTangent<double>;
template class NiwTangent<float>;

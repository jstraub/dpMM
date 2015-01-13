/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */


#include <iostream>
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE dpSubclusterMM test
#include <boost/test/unit_test.hpp>

#include "lrCluster.hpp"
#include "niwBaseMeasure.hpp"
#include "niwSphere.hpp"
#include "niwSphereFull.hpp"
#include "distribution.hpp"
#include "normalSphere.hpp"
#include "dpSubclusterMM.hpp"

#include <Eigen/Dense>

using namespace Eigen;
using std::cout;
using std::endl;


BOOST_AUTO_TEST_CASE(lrCluster_merge_test)
{

  return;
  boost::mt19937 rndGen(91);
  
  uint32_t N=80;
  uint32_t D=3;
  uint32_t K=1;
  shared_ptr<MatrixXd> spx(new MatrixXd(D,N));
  sampleClustersOnSphere<double>(*spx, K);
  
//  uint32_t K0=2;
//  double alpha = 1.;
  double nu = D+1.0;
  MatrixXd Delta(D-1,D-1);
  Delta.setIdentity();
  cout<<"Delta/nu"<<endl<<(Delta/nu)<<endl;

  IWd iw(Delta,nu,&rndGen);
  shared_ptr<NiwSphered> theta(new NiwSphered(iw,&rndGen));
  shared_ptr<NiwSphered> theta2(new NiwSphered(iw,&rndGen));

  shared_ptr<LrCluster<NiwSphered,double> > lrTheta(new 
    LrCluster<NiwSphered,double>(theta,1.0,&rndGen));
  VectorXd mean(3);
  mean << 0,0,1;
  lrTheta->getUpper()->normalS_.setMean( mean);
  lrTheta->getUpper()->scatter() = MatrixXd::Identity(D-1,D-1);
  lrTheta->getUpper()->count() = 10;

  shared_ptr<LrCluster<NiwSphered,double> > lrTheta2(new
    LrCluster<NiwSphered,double>(theta2,1.0,&rndGen));
  mean << 0,0,1;
  lrTheta2->getUpper()->normalS_.setMean(mean);
  lrTheta2->getUpper()->scatter() = MatrixXd::Identity(D-1,D-1);
//  lrTheta2->getUpper()->scatter() *= 2.0;
  lrTheta2->getUpper()->count() = 10;

  cout<<" -- -- -- lrCluster 1"<<endl;
  lrTheta->getUpper()->print();
  cout<<" -- -- -- lrCluster 2"<<endl;
  lrTheta2->getUpper()->print();
//  cout<<"copy of LR "<<lrTheta2.get()<<endl;
//  cout<<lrTheta->getUpper()<<endl;
//  cout<<lrTheta2->getUpper()<<endl;
  shared_ptr<LrCluster<NiwSphered,double> > lrMerged(lrTheta->merge(lrTheta2));

  cout<<" -- -- -- lrCluster merged"<<endl;
  lrMerged->getUpper()->print();
  cout<<" -------------- left right tests ----------"<<endl;

  lrMerged->getL().reset(lrTheta->getUpper()->copyNative());  
  lrMerged->getR().reset(lrTheta2->getUpper()->copyNative());  
  cout<<" -- -- -- lrCluster L"<<endl;
  lrMerged->getL()->print();
  cout<<" -- -- -- lrCluster R"<<endl;
  lrMerged->getR()->print();

  lrMerged->getUpper()->fromMerge(*lrMerged->getL(),*lrMerged->getR());
  cout<<" -- -- -- lrCluster Upper"<<endl;
  lrMerged->getUpper()->print();


}

BOOST_AUTO_TEST_CASE(dpSubclusterMM_merge_test)
{
  return;

  boost::mt19937 rndGen(91);
  
  uint32_t N=80;
  uint32_t D=3;
  uint32_t K=1;
  shared_ptr<MatrixXd> spx(new MatrixXd(D,N));
  sampleClustersOnSphere<double>(*spx, K);
  
  uint32_t K0=2;
  double alpha = 1.;
  double nu = D+1.0;
  MatrixXd Delta(D-1,D-1);
  Delta.setIdentity();
  Delta *= nu*(4.*PI)/180.0;
  cout<<"Delta/nu"<<endl<<(Delta/nu)<<endl;

  IWd iw(Delta,nu,&rndGen);
  shared_ptr<NiwSphered> theta(new NiwSphered(iw,&rndGen));
  shared_ptr<LrCluster<NiwSphered,double> > lrTheta(new 
    LrCluster<NiwSphered,double>(theta,1.0,&rndGen));
  DpSubclusterMM<NiwSphered,double> dp(alpha, lrTheta, K0, &rndGen);
   
  dp.initialize(spx);
  cout<<"z:  "<<dp.getLabels().transpose()<<endl;
  uint32_t T=3;
  for(uint32_t t=0; t<T; ++t)
  {
    cout<<" -------- sample params ("<<t<<")"<<endl;
    dp.sampleParameters_();
    cout<<"z:  "<<dp.getLabels().transpose()<<endl;

    cout<<" -------- sample labels ("<<t<<")"<<endl;
    dp.sampleLabels();
    cout<<"z:  "<<dp.getLabels().transpose()<<endl;

    cout<<" -------- propose merges ("<<t<<")"<<endl;
    dp.proposeMerges();
    cout<<"z:  "<<dp.getLabels().transpose()<<endl;
   
//    cout<<" -------- propose splits ("<<t<<")"<<endl;
//    dp.proposeSplits();
//    cout<<"z:  "<<dp.getLabels().transpose()<<endl;
  }
}

BOOST_AUTO_TEST_CASE(dpSubclusterMM_split_test)
{
  return;
  cout<<" ---------------------- dpSubclusterMM_split_test -------------"<<endl;
  cout<<" ---------------------- dpSubclusterMM_split_test -------------"<<endl;
  cout<<" ---------------------- dpSubclusterMM_split_test -------------"<<endl;
  cout<<" ---------------------- dpSubclusterMM_split_test -------------"<<endl;

  boost::mt19937 rndGen(91);
  
  uint32_t N=80;
  uint32_t D=3;
  uint32_t K=2;
  shared_ptr<MatrixXd> spx(new MatrixXd(D,N));
  cout<<"true mus:"<<endl<<sampleClustersOnSphere<double>(*spx, K)<<endl;
  
  uint32_t K0=1;
  double alpha = 1.;
  double nu = D+1.0;
  MatrixXd Delta(D-1,D-1);
  Delta.setIdentity();
  Delta *= nu*(1.*PI)/180.0;
  cout<<"Delta/nu"<<endl<<(Delta/nu)<<endl;

  IWd iw(Delta,nu,&rndGen);
  shared_ptr<NiwSphered> theta(new NiwSphered(iw,&rndGen));
  shared_ptr<LrCluster<NiwSphered,double> > lrTheta(new 
    LrCluster<NiwSphered,double>(theta,1.0,&rndGen));
  DpSubclusterMM<NiwSphered,double> dp(alpha, lrTheta, K0, &rndGen);
   
  dp.initialize(spx);
  cout<<"z:  "<<dp.getLabels().transpose()<<endl;
  uint32_t T=30;
  for(uint32_t t=0; t<T; ++t)
  {
    cout<<" -------- sample params ("<<t<<")"<<endl;
    dp.sampleParameters_();
    cout<<"z:  "<<dp.getLabels().transpose()<<endl;

    cout<<" -------- sample labels ("<<t<<")"<<endl;
    dp.sampleLabels();
    cout<<"z:  "<<dp.getLabels().transpose()<<endl;
    cout<<"log Joint = "<<dp.logJoint()<<endl;

    cout<<" -------- propose merges ("<<t<<")"<<endl;
    dp.proposeMerges();
    cout<<"z:  "<<dp.getLabels().transpose()<<endl;
   
    cout<<" -------- propose splits ("<<t<<")"<<endl;
    dp.proposeSplits();
    cout<<"z:  "<<dp.getLabels().transpose()<<endl;
    for(uint32_t k=0; k<dp.getK(); ++k)
      dp.get(k)->print();
  }
}

BOOST_AUTO_TEST_CASE(dpSubclusterMM_NiwSphereFull_split_test)
{
  return ;
  cout<<" ----------------- dpSubclusterMM_NiwSphereFull_split_test -------------"<<endl;
  cout<<" ----------------- dpSubclusterMM_NiwSphereFull_split_test -------------"<<endl;
  cout<<" ----------------- dpSubclusterMM_NiwSphereFull_split_test -------------"<<endl;
  cout<<" ----------------- dpSubclusterMM_NiwSphereFull_split_test -------------"<<endl;

  boost::mt19937 rndGen(91);
  
  uint32_t N=80;
  uint32_t D=3;
  uint32_t K=2;
  shared_ptr<MatrixXd> spx(new MatrixXd(D,N));
  cout<<"true mus:"<<endl<<sampleClustersOnSphere<double>(*spx, K)<<endl;
  
  uint32_t K0=1;
  double alpha = 1.;
  double nu = D+1.0;
  VectorXd mu(D); mu.setZero();
  mu(D-1) = 1.0;
  MatrixXd Delta(D-1,D-1);
  Delta.setIdentity();
  Delta *= nu*(1.*PI)/180.0;
  cout<<"Delta/nu"<<endl<<(Delta/nu)<<endl;

  IWd iw(Delta,nu,&rndGen);
  boost::shared_ptr<NiwSphereFulld> theta(new NiwSphereFulld(iw,&rndGen));
  boost::shared_ptr<LrCluster<NiwSphereFulld,double> > lrTheta(new 
    LrCluster<NiwSphereFulld,double>(theta,1.0,&rndGen));
  DpSubclusterMM<NiwSphereFulld,double> dp(alpha, lrTheta, K0, &rndGen);
   
  dp.initialize(spx);
  cout<<"z:  "<<dp.getLabels().transpose()<<endl;
  uint32_t T=30;
  for(uint32_t t=0; t<T; ++t)
  {
    cout<<" -------- sample params ("<<t<<")"<<endl;
    dp.sampleParameters_();
    cout<<"z:  "<<dp.getLabels().transpose()<<endl;

    cout<<" -------- sample labels ("<<t<<")"<<endl;
    dp.sampleLabels();
    cout<<"z:  "<<dp.getLabels().transpose()<<endl;
    cout<<"log Joint = "<<dp.logJoint()<<endl;

    cout<<" -------- propose merges ("<<t<<")"<<endl;
    dp.proposeMerges();
    cout<<"z:  "<<dp.getLabels().transpose()<<endl;
   
    cout<<" -------- propose splits ("<<t<<")"<<endl;
    dp.proposeSplits();
    cout<<"z:  "<<dp.getLabels().transpose()<<endl;
    for(uint32_t k=0; k<dp.getK(); ++k)
      dp.get(k)->print();
  }
}

//BOOST_AUTO_TEST_CASE(dpSubclusterMM_NiwTangent_split_test)
//{
//  cout<<" ----------------- dpSubclusterMM_NiwTangent_split_test -------------"<<endl;
//  cout<<" ----------------- dpSubclusterMM_NiwTangent_split_test -------------"<<endl;
//  cout<<" ----------------- dpSubclusterMM_NiwTangent_split_test -------------"<<endl;
//  cout<<" ----------------- dpSubclusterMM_NiwTangent_split_test -------------"<<endl;
//
//  boost::mt19937 rndGen(91);
//  
//  uint32_t N=80;
//  uint32_t D=3;
//  uint32_t K=2;
//  shared_ptr<MatrixXd> spx(new MatrixXd(D,N));
//  cout<<"true mus:"<<endl<<sampleClustersOnSphere<double>(*spx, K)<<endl;
//  
//  uint32_t K0=1;
//  double alpha = 1.;
//  double nu = D+1.0;
//  double kappa = 1.0;
//  VectorXd mu(D-1); mu.setZero();
//  MatrixXd Delta(D-1,D-1);
//  Delta.setIdentity();
//  Delta *= nu*(1.*PI)/180.0;
//  cout<<"Delta/nu"<<endl<<(Delta/nu)<<endl;
//
//  NIWd niw(Delta,mu,nu,kappa,&rndGen);
//  shared_ptr<NiwTangentd> theta(new NiwTangentd(niw,&rndGen));
//  shared_ptr<LrCluster<NiwTangentd,double> > lrTheta(new 
//    LrCluster<NiwTangentd,double>(theta,1.0,&rndGen));
//  DpSubclusterMM<NiwTangentd,double> dp(alpha, lrTheta, K0, &rndGen);
//   
//  dp.initialize(spx);
//  cout<<"z:  "<<dp.getLabels().transpose()<<endl;
//  uint32_t T=30;
//  for(uint32_t t=0; t<T; ++t)
//  {
//    cout<<" -------- sample params ("<<t<<")"<<endl;
//    dp.sampleParameters_();
//    cout<<"z:  "<<dp.getLabels().transpose()<<endl;
//
//    cout<<" -------- sample labels ("<<t<<")"<<endl;
//    dp.sampleLabels();
//    cout<<"z:  "<<dp.getLabels().transpose()<<endl;
//    cout<<"log Joint = "<<dp.logJoint()<<endl;
//
//    cout<<" -------- propose merges ("<<t<<")"<<endl;
//    dp.proposeMerges();
//    cout<<"z:  "<<dp.getLabels().transpose()<<endl;
//   
//    cout<<" -------- propose splits ("<<t<<")"<<endl;
//    dp.proposeSplits();
//    cout<<"z:  "<<dp.getLabels().transpose()<<endl;
//    for(uint32_t k=0; k<dp.getK(); ++k)
//      dp.get(k)->print();
//  }
//}
//

//BOOST_AUTO_TEST_CASE(dpSubclusterMM_niw_split_test)
//{ 
//  cout<<" -------------------------------------------------------------"<<endl;
//  cout<<" -------------------------------------------------------------"<<endl;
//  cout<<" -------------------------------------------------------------"<<endl;
//  boost::mt19937 rndGen(91);
//  
//  uint32_t N=80;
//  uint32_t D=3;
//  uint32_t K=2;
//  VectorXu z(N);
//  shared_ptr<MatrixXd> spx(new MatrixXd(D,N));
//  cout<<"true mus:"<<endl<<sampleClusters<double>(*spx, z, K)<<endl;
//  
//  uint32_t K0=1;
//  double alpha = 1.;
//  double nu = D+10.0;
//  double kappa = 1.0;
//  VectorXd theta(D);
//  theta.setZero();
//  MatrixXd Delta(D,D);
//  Delta.setIdentity();
//  Delta *= nu*(2.*PI)/180.0;
//  cout<<"Delta/nu"<<endl<<(Delta/nu)<<endl;
//
//  NIWd niw(Delta,theta,nu,kappa,&rndGen);
//  shared_ptr<NiwSampledd> niwSampl(new NiwSampledd(niw));
//  shared_ptr<LrCluster<NiwSampledd,double> > lrTheta(new 
//    LrCluster<NiwSampledd,double>(niwSampl,1.0,&rndGen));
//  DpSubclusterMM<NiwSampledd,double> dp(alpha, lrTheta, K0, &rndGen);
//   
//  dp.initialize(spx);
//  cout<<"z:  "<<dp.getLabels().transpose()<<endl;
//  uint32_t T=100;
//  for(uint32_t t=0; t<T; ++t)
//  {
//    cout<<" -------- sample labels ("<<t<<")"<<endl;
//    dp.sampleLabels();
//    cout<<"z:  "<<dp.getLabels().transpose()<<endl;
//
//    cout<<" -------- sample params ("<<t<<")"<<endl;
//    dp.sampleParameters_();
//    cout<<"z:  "<<dp.getLabels().transpose()<<endl;
//
//    cout<<" -------- propose merges ("<<t<<")"<<endl;
//    dp.proposeMerges();
//    cout<<"z:  "<<dp.getLabels().transpose()<<endl;
//   
//    cout<<" -------- propose splits ("<<t<<")"<<endl;
//    dp.proposeSplits();
//    cout<<"z:  "<<dp.getLabels().transpose()<<endl;
//    
//    for(uint32_t k=0; k<dp.getK(); ++k)
//      dp.get(k)->print();
//  }
//}

/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */

#include <iostream>
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE crpMM test
#include <boost/test/unit_test.hpp>

#include "crpMM.hpp"
#include "niwBaseMeasure.hpp"
#include "niwSphere.hpp"
#include "vmfBaseMeasure.hpp"

BOOST_AUTO_TEST_CASE(crpMM_test)
{
  uint32_t N=20;
  MatrixXd x(3,N);
  for(uint32_t i=0; i<N; ++i)
    if(i<N/2)
      x.col(i) << 0.0,0.0,0.0;
    else
      x.col(i) << 10.0,10.0,10.0;

  double nu = 4.0;
  double kappa = 4.0;
  MatrixXd Delta(3,3);
  Delta << .1,0.0,0.0,
        0.0,.1,0.0,
        0.0,0.0,.1;
  VectorXd theta(3);
  theta << 0.0,0.0,0.0;
  double alpha = 1.0;

  boost::mt19937 rndGen(9191);
  NIW<double> niw(Delta,theta,nu,kappa,&rndGen);

//  shared_ptr<NiwMarginalized<double> > niwMargBase(
//      new NiwMarginalized<double>(niw));
//  CrpMM<double> dirGMM_marg(alpha,niwMargBase,2,&rndGen);
//  
//  dirGMM_marg.initialize(x);
//  cout<<"------ sampling -- NIW marginalized"<<endl;
//  cout<<dirGMM_marg.labels().transpose()<<endl;
//  for(uint32_t t=0; t<30; ++t)
//  {
//    dirGMM_marg.sampleLabels();
//    dirGMM_marg.sampleParameters();
//    cout<<dirGMM_marg.labels().transpose()
//      <<" logJoint="<<dirGMM_marg.logJoint()<<endl;
//  }

  shared_ptr<NiwSampled<double> > niwSampled(
      new NiwSampled<double>(niw));
  CrpMM<double> dirGMM_samp(alpha,niwSampled,2,&rndGen);
  
  dirGMM_samp.initialize(x);
  cout<<"------ sampling -- NIW sampled"<<endl;
  cout<<dirGMM_samp.labels().transpose()<<endl;
  for(uint32_t t=0; t<30; ++t)
  {
    dirGMM_samp.sampleLabels();
    dirGMM_samp.sampleParameters();
    cout<<dirGMM_samp.labels().transpose()
      <<" logJoint="<<dirGMM_samp.logJoint()<<endl;
  }
};
//
//
BOOST_AUTO_TEST_CASE(crpMM_Sphere_test)
{
  cout<<"------ sampling -- NIW sphere"<<endl;

  double nu = 20.0;
  MatrixXd Delta(2,2);
  Delta << .01,0.0,
        0.0,.01;
  Delta *= nu;

  boost::mt19937 rndGen(9191);
  IW<double> iw(Delta,nu,&rndGen);
  shared_ptr<NiwSphere<double> > niwSp( new NiwSphere<double>(iw,&rndGen));
  double alpha = 1.0;
  CrpMM<double> dirGMM_sp(alpha,niwSp,1,&rndGen);
  
  uint32_t N=20;
  uint32_t K=2;
  MatrixXd x(3,N);
  MatrixXd mus = sampleClustersOnSphere<double>(x, K);

  dirGMM_sp.initialize(x);

  cout<<"true means: "<<endl<<mus<<endl;
  cout<<dirGMM_sp.labels().transpose()<<endl;
  for(uint32_t t=0; t<10; ++t)
  {
    dirGMM_sp.sampleParameters();
//    for(uint32_t k=0; k<dirGMM_sp.getK(); ++k)
//    {
//      cout<<"  k: "<<k<<" "<<endl; 
//      dirGMM_sp.getTheta(k)->print();
//    }
    dirGMM_sp.sampleLabels();
    cout<<"@t="<<t<<" "<<dirGMM_sp.labels().transpose()
      <<" logJoint="<<dirGMM_sp.logJoint()<<endl;
  }
  MatrixXd logLikes;
  MatrixXu inds = dirGMM_sp.mostLikelyInds(5,logLikes);
  cout<<"most likely indices"<<endl;
  cout<<inds<<endl;
  cout<<"----------------------------------------"<<endl;
};

BOOST_AUTO_TEST_CASE(crpMM_vMF_test)
{
  cout<<"------ sampling -- Dir-vMF"<<endl;

  double a0 = 2.0;
  double b0 = 1.7;
  double t0 = 0.01;
  VectorXd m0(3);
  m0 << 1.0,0.0,0.0;

  boost::mt19937 rndGen(9191);

  vMFpriorFull<double> vMFprior(m0,t0,a0,b0,&rndGen);
  shared_ptr<vMFbase<double> > vMFsampled( new vMFbase<double>(vMFprior));
  
  double alpha = 1.0;
  CrpMM<double> dirvMF_sp(alpha,vMFsampled,1,&rndGen);
  
  uint32_t N=20;
  uint32_t K=2;
  MatrixXd x(3,N);
  MatrixXd mus = sampleClustersOnSphere<double>(x, K);

  dirvMF_sp.initialize(x);

  cout<<"true means: "<<endl<<mus<<endl;
  cout<<dirvMF_sp.labels().transpose()<<endl;
  for(uint32_t t=0; t<100; ++t)
  {
    dirvMF_sp.sampleParameters();
//    for(uint32_t k=0; k<dirvMF_sp.getK(); ++k)
//    {
//      cout<<"  k: "<<k<<" "<<endl; 
//      dirvMF_sp.getTheta(k)->print();
//    }
    dirvMF_sp.sampleLabels();
    cout<<"@t="<<t<<" "<<dirvMF_sp.labels().transpose()
      <<" logJoint="<<dirvMF_sp.logJoint()<<endl;
  }
//  MatrixXd logLikes;
//  MatrixXu inds = dirvMF_sp.mostLikelyInds(5,logLikes);
//  cout<<"most likely indices"<<endl;
//  cout<<inds<<endl;
  cout<<"----------------------------------------"<<endl;
};


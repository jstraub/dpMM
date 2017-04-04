/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */

#include <iostream>
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE dirMM test
#include <boost/test/unit_test.hpp>

#include <dpMM/dirMM.hpp>
#include <dpMM/niwBaseMeasure.hpp>
#include <dpMM/niwSphere.hpp>
#include <dpMM/dirMMcld.hpp>
#include <dpMM/clTGMMDataGpu.hpp>
#include <dpMM/distribution.hpp>
#include <dpMM/vmfBaseMeasure.hpp>
#include <dpMM/vmfBaseMeasure3D.hpp>

BOOST_AUTO_TEST_CASE(niwBaseMeasure_test)
{
  MatrixXd Delta(3,3);
  Delta << 1.0,0.0,0.0,
        0.0,1.0,0.0,
        0.0,0.0,1.0;
  VectorXd theta(3);
  theta << 1.0,1.0,1.0;
  double nu = 100.0;
  double kappa = 100.0;

  boost::mt19937 rndGen(1);
  NIW<double> niw(Delta,theta,nu,kappa,&rndGen);

  NiwMarginalized<double> niwMargBase(niw);

  VectorXd x(3);
  x << 1.0,1.0,1.0;
  cout<< niwMargBase.logLikelihood(x)<< endl;

  NiwSampled<double> niwSampledBase(niw);

  x << 1.0,1.0,1.0;
  cout<< niwSampledBase.logLikelihood(x)<< endl;
};

BOOST_AUTO_TEST_CASE(dirMM_test)
{

  double nu = 4.0;
  double kappa = 4.0;
  MatrixXd Delta(3,3);
  Delta << .1,0.0,0.0,
        0.0,.1,0.0,
        0.0,0.0,.1;
  VectorXd theta(3);
  theta << 0.0,0.0,0.0;

  boost::mt19937 rndGen(9191);
  NIW<double> niw(Delta,theta,nu,kappa,&rndGen);

  shared_ptr<NiwMarginalized<double> > niwMargBase(
      new NiwMarginalized<double>(niw));
  VectorXd alpha(2);
  alpha << 10.,10.;
  Dir<Cat<double>, double> dir(alpha,&rndGen); 
  DirMM<double> dirGMM_marg(dir,niwMargBase,2);
  
  uint32_t N=20;
  MatrixXd x(3,N);
  for(uint32_t i=0; i<N; ++i)
    if(i<N/2)
      x.col(i) << 0.0,0.0,0.0;
    else
      x.col(i) << 10.0,10.0,10.0;
  dirGMM_marg.initialize(x);
  cout<<"------ sampling -- NIW marginalized"<<endl;
  cout<<dirGMM_marg.labels().transpose()<<endl;
  for(uint32_t t=0; t<30; ++t)
  {
    dirGMM_marg.sampleLabels();
    dirGMM_marg.sampleParameters();
    cout<<dirGMM_marg.labels().transpose()
      <<" logJoint="<<dirGMM_marg.logJoint()<<endl;
  }

  shared_ptr<NiwSampled<double> > niwSampled(
      new NiwSampled<double>(niw));
  DirMM<double> dirGMM_samp(dir,niwSampled,2);
  
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


BOOST_AUTO_TEST_CASE(dirMM_Sphere_test)
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

  VectorXd alpha(2);
  alpha << 10.,10.;
  Dir<Cat<double>, double> dir(alpha,&rndGen); 
  DirMM<double> dirGMM_sp(dir,niwSp,2);
  
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

typedef double myFlt;

BOOST_AUTO_TEST_CASE(dirMMcld_Sphere_test)
{
  cout<<"------ sampling -- NIW sphere using CLD"<<endl;

  uint32_t N=30; //640*480;
  uint32_t K=6;
  uint32_t D=3;
  boost::mt19937 rndGen(9191);
  // sample datapoints
  shared_ptr<Matrix<myFlt,Dynamic,Dynamic> > sx(new 
      Matrix<myFlt,Dynamic,Dynamic>(D,N));
  Matrix<myFlt,Dynamic,Dynamic> mus =  sampleClustersOnSphere(*sx, 3);

  // alpha
  Matrix<myFlt,Dynamic,1> alpha(K);
  alpha << 1.,1.,.1,.1,.1,.1;
  alpha *= 1;
  
  // niw
  double nu = (1.0)+D+N/100.;
  Matrix<myFlt,Dynamic,Dynamic> Delta(2,2);
  Delta << .01,0.0,
        0.0,.01;
  Delta *= nu;

  IW<myFlt> iw(Delta,nu,&rndGen);
  shared_ptr<NiwSphere<myFlt> > niwSp( new NiwSphere<myFlt>(iw,&rndGen));
//  shared_ptr<NiwSphere<double> > niwSp2( new NiwSphere<double>(iw,&rndGen));

  Dir<Cat<myFlt>, myFlt> dir(alpha,&rndGen); 
  DirMMcld<NiwSphere<myFlt>,myFlt> dirGMM_sp(dir,niwSp);

//  DirMM<myFlt> dirGMM_cpu(dir,niwSp2);

  shared_ptr<ClTGMMDataGpu<myFlt> > clsp(
      new ClTGMMDataGpu<myFlt>(sx, spVectorXu(new VectorXu(N)),K));

  MatrixXd ps(D,K);
  Sphere<double> sphere(D);
  for(uint32_t k=0; k<K; ++k)
    ps.col(k) = sphere.sampleUnif(&rndGen);
  clsp->init(ps);

  Matrix<myFlt,Dynamic,1> mu(D);
  mu<<0.0,0.0,1.0;
  Matrix<myFlt,Dynamic,Dynamic> Sigma = Matrix<myFlt,Dynamic,Dynamic>::Identity(D,D);

  dirGMM_sp.initialize(clsp);
//  dirGMM_cpu.initialize(*sx);
  cout<<counts<myFlt,uint32_t>(dirGMM_sp.labels(),K).transpose()<<endl;
  Timer t;
  for(uint32_t i=0; i<5; ++i)
  {
//    t.tic();
//    dirGMM_cpu.sampleLabels();
//    dirGMM_cpu.sampleParameters();
//    cout<<dirGMM_cpu.labels().transpose()<<endl;
//    t.toctic(" -----------------CPU------------------- fullIteration");
    t.tic();
    dirGMM_sp.sampleParameters();
    dirGMM_sp.sampleLabels();
    cout<<dirGMM_sp.z().transpose()<<endl;
    cout<<dirGMM_sp.counts().transpose()<<endl;
    cout<<dirGMM_sp.means()<<endl;
    t.toctic(" -----------------GPU------------------- fullIteration");
//        <<" logJoint="<<dirGMM_sp.logJoint()<<endl;
  }
  cout<<"true mus: "<<endl;
  cout <<mus<<endl;
};


BOOST_AUTO_TEST_CASE(dirMM_vMFsampled_test)
{
  cout<<"------ sampling -- Dir-vMF (old, monte carlo estimation for marginalization)"<<endl;

  double a0 = 5.0;
  double b0 = 4.7;
  double t0 = 0.1;
  VectorXd m0(3);
  m0 << 1.0,0.0,0.0;

  boost::mt19937 rndGen(9191);

  vMFpriorFull<double> vMFprior(m0,t0,a0,b0,&rndGen);
  shared_ptr<vMFbase<double> > vMFsampled( new vMFbase<double>(vMFprior));

  VectorXd alpha(2);
  alpha << 10.,10.;
  Dir<Cat<double>, double> dir(alpha,&rndGen); 
  DirMM<double> dirvMF_sp(dir,vMFsampled,2);
  
  uint32_t N=20;
  uint32_t K=2;
  MatrixXd x(3,N);
  MatrixXd mus = sampleClustersOnSphere<double>(x, K);

  dirvMF_sp.initialize(x);

  cout<<"true means: "<<endl<<mus<<endl;
  cout<<dirvMF_sp.labels().transpose()<<endl;
  for(uint32_t t=0; t<10; ++t)
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
  cout<<"true means: "<<endl<<mus<<endl;
    for(uint32_t k=0; k<dirvMF_sp.getK(); ++k) 
    { 
      cout<<"  k: "<<k<<endl; 
      dirvMF_sp.getTheta(k)->print(); 
    }
  cout<<"----------------------------------------"<<endl;
};


BOOST_AUTO_TEST_CASE(dirMM_vMF_test)
{
  cout<<"------ sampling -- Dir-vMF (new, analytic marginalization for 3D)"<<endl;

  double a0 = 1.0;
  double b0 = 0.99;
  VectorXd m0(3);
  m0 << 1.0,0.0,0.0;

  boost::mt19937 rndGen(9191);

  vMFprior<double> vMFprior(m0,a0,b0,&rndGen);
  shared_ptr<vMFbase3D<double> > vMFbase( new vMFbase3D<double>(vMFprior));

//  VectorXd alpha(1);
//  alpha << 10.;
//  Dir<Cat<double>, double> dir(alpha,&rndGen); 
//  DirMM<double> dirvMF_sp(dir,vMFbase,1);

  VectorXd alpha(2);
  alpha << 10.,10.;
  Dir<Cat<double>, double> dir(alpha,&rndGen); 
  DirMM<double> dirvMF_sp(dir,vMFbase,2);
  
  uint32_t N=200;
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
  cout<<"true means: "<<endl<<mus<<endl;
    for(uint32_t k=0; k<dirvMF_sp.getK(); ++k) 
    { 
      cout<<"  k: "<<k<<endl; 
      dirvMF_sp.getTheta(k)->print(); 
    }
  cout<<"----------------------------------------"<<endl;

//  vMF<double> vmf(m0, 100, &rndGen);
//  std::cout << m0.transpose() << std::endl << " -- " << std::endl;
//  for (size_t i=0; i<100; ++i) {
//    Eigen::VectorXd x = vmf.sample();
//    std::cout << x.transpose()  << " || " << x.norm() << std::endl;
//  }
//
//  std::cout << " -- " << std::endl;
//  for (size_t i=0; i<100; ++i) {
//    vmf = vMFprior.sample();
//    vmf.print();
////    std::cout << vmf.mu_.transpose()  << " tau " << vmf.tau_ << std::endl;
//  }

};

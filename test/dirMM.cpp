
#include <iostream>
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE dirMM test
#include <boost/test/unit_test.hpp>

#include "dirMM.hpp"
#include "niwBaseMeasure.hpp"
#include "niwSphere.hpp"
#include "dirMMcld.hpp"
#include "clSphereGpu.hpp"
#include "distribution.hpp"

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
  Delta *= nu;
  VectorXd theta(3);
  theta << 0.0,0.0,0.0;

  boost::mt19937 rndGen(9191);
  NIW<double> niw(Delta,theta,nu,kappa,&rndGen);

  shared_ptr<NiwMarginalized<double> > niwMargBase(
      new NiwMarginalized<double>(niw));
  VectorXd alpha(2);
  alpha << 10.,10.;
  Dir<double> dir(alpha,&rndGen); 
  DirMM<double> dirGMM_marg(dir,niwMargBase);
  
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
    dirGMM_marg.sampleIndicators();
    dirGMM_marg.sampleParameters();
    cout<<dirGMM_marg.labels().transpose()
      <<" logJoint="<<dirGMM_marg.logJoint()<<endl;
  }

  shared_ptr<NiwSampled<double> > niwSampled(
      new NiwSampled<double>(niw));
  DirMM<double> dirGMM_samp(dir,niwSampled);
  
  dirGMM_samp.initialize(x);
  cout<<"------ sampling -- NIW sampled"<<endl;
  cout<<dirGMM_samp.labels().transpose()<<endl;
  for(uint32_t t=0; t<30; ++t)
  {
    dirGMM_samp.sampleIndicators();
    dirGMM_samp.sampleParameters();
    cout<<dirGMM_samp.labels().transpose()
      <<" logJoint="<<dirGMM_samp.logJoint()<<endl;
  }
};


BOOST_AUTO_TEST_CASE(dirMM_Sphere_test)
{

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
  Dir<double> dir(alpha,&rndGen); 
  DirMM<double> dirGMM_sp(dir,niwSp);
  
  uint32_t N=20;
  uint32_t K=2;
  MatrixXd x(3,N);
  sampleClustersOnSphere<double>(x, K);
  dirGMM_sp.initialize(x);
  cout<<"------ sampling -- NIW sphere"<<endl;
  cout<<dirGMM_sp.labels().transpose()<<endl;
  for(uint32_t t=0; t<10; ++t)
  {
    dirGMM_sp.sampleIndicators();
    dirGMM_sp.sampleParameters();
    cout<<dirGMM_sp.labels().transpose()
      <<" logJoint="<<dirGMM_sp.logJoint()<<endl;
  }
  MatrixXu inds = dirGMM_sp.mostLikelyInds(5);
  cout<<"most likely indices"<<endl;
  cout<<inds<<endl;
  cout<<"----------------------------------------"<<endl;
};

typedef double myFlt;

BOOST_AUTO_TEST_CASE(dirMMcld_Sphere_test)
{

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

  Dir<myFlt> dir(alpha,&rndGen); 
  DirMMcld<NiwSphere<myFlt>,myFlt> dirGMM_sp(dir,niwSp);

//  DirMM<myFlt> dirGMM_cpu(dir,niwSp2);

  shared_ptr<ClSphereGpu<myFlt> > clsp(
      new ClSphereGpu<myFlt>(sx, spVectorXu(new VectorXu(N)),&rndGen,K));

  Matrix<myFlt,Dynamic,1> mu(D);
  mu<<0.0,0.0,1.0;
  Matrix<myFlt,Dynamic,Dynamic> Sigma = Matrix<myFlt,Dynamic,Dynamic>::Identity(D,D);

  dirGMM_sp.initialize(clsp);
//  dirGMM_cpu.initialize(*sx);
  cout<<"------ sampling -- NIW sphere"<<endl;
  cout<<counts(dirGMM_sp.labels(),K).transpose()<<endl;
  Timer t;
  for(uint32_t i=0; i<5; ++i)
  {
//    t.tic();
//    dirGMM_cpu.sampleIndicators();
//    dirGMM_cpu.sampleParameters();
//    cout<<dirGMM_cpu.labels().transpose()<<endl;
//    t.toctic(" -----------------CPU------------------- fullIteration");
    t.tic();
    dirGMM_sp.sampleIndicators();
    dirGMM_sp.sampleParameters();
    cout<<dirGMM_sp.z().transpose()<<endl;
    cout<<dirGMM_sp.counts().transpose()<<endl;
    cout<<dirGMM_sp.means()<<endl;
    t.toctic(" -----------------GPU------------------- fullIteration");
//        <<" logJoint="<<dirGMM_sp.logJoint()<<endl;
  }
  cout<<"true mus: "<<endl;
  cout <<mus<<endl;
};


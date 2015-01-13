/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */

#include <iostream>
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE distributions test
#include <boost/test/unit_test.hpp>

#include "sphere.hpp"
#include "clSphereGpu.hpp"
#include "karcherMean.hpp"
#include "normalSphere.hpp"
#include "timer.hpp"

using std::cout;
using std::endl;

BOOST_AUTO_TEST_CASE( sphere_test)
{
  cout<<"----------------------- sphere ----------------------"<<endl;
  
  boost::mt19937 rndGen(1);
  Sphere<double> Sd(3);
  
  cout<<"sampled point: "<<Sd.sampleUnif(&rndGen).transpose()<<endl;

  VectorXd a(3);
  VectorXd b(3);
  a << 0.0,0.0,1.0;
  b << 0.0,0.0,1.0;
  MatrixXd bRa = rotationFromAtoB<double>(a,b);
  MatrixXd aRb = rotationFromAtoB<double>(b,a);
  BOOST_CHECK((bRa.array() == aRb.transpose().array()).all());
  BOOST_CHECK((bRa.array() == MatrixXd::Identity(3,3).array()).all());
  
  a << 0.0,0.0,1.0;
  b << 0.0,1.0,0.0;
  bRa = rotationFromAtoB<double>(a,b);
  aRb = rotationFromAtoB<double>(b,a);
  MatrixXd bRaSoll(3,3);
  bRaSoll << 1.0,0.0,0.0,
           0.0,0.0,1.0,
           0.0,-1.0,0.0;
  BOOST_CHECK((bRa.array() == aRb.transpose().array()).all());
//  cout<<bRa<<endl;
//  cout<<bRaSoll<<endl;
  BOOST_CHECK(((bRa - bRaSoll).array() < 1e-6).all());

  
  VectorXd p = Sd.sampleUnif(&rndGen);
  VectorXd q = Sd.sampleUnif(&rndGen);
  cout<< "q             : " << q.transpose()<< endl;
  cout<< "q in TpS north: " << Sd.Log_p_north(p,q).transpose()<< endl;
  cout<< "q (Log->Exp)  : " << Sd.Exp_p(p,Sd.Log_p(p,q)).transpose()<< endl;

  Sphere<double> Sd10(10);
  p = Sd10.sampleUnif(&rndGen);
  q = Sd10.sampleUnif(&rndGen);
  cout<< "x in TpS: " << Sd10.Log_p_north(p,q).transpose()<< endl;
  
  uint32_t D = 3;
  uint32_t K = 5;
  uint32_t N = 20;
  boost::shared_ptr<Matrix<double,Dynamic,Dynamic> > sq(
      new Matrix<double,Dynamic,Dynamic>(D,N));
  Matrix<double,Dynamic,Dynamic> xMu(D,N);
  Matrix<double,Dynamic,Dynamic> x(D-1,N);
  Matrix<double,Dynamic,Dynamic> p0s(D,K);
  for (uint32_t k=0; k<K; ++k)
     p0s.col(k) =  Sd.sampleUnif(&rndGen);
  VectorXu z(N);
  Matrix<double,Dynamic,Dynamic> mus = sampleClustersOnSphere<double>(*sq, z, K);

  Matrix<double,Dynamic,Dynamic> muEst = karcherMeanMultiple(

  cout<<"muTrue="<<endl<<mus<<endl;
  cout<<"muEst ="<<endl<<muEst<<endl;
  cout<<"muInit ="<<endl<<p0s<<endl;

  Sphere<double> S(D);
  S.rotate_p2north(muEst,xMu, x, z, K);
  
  for(uint32_t i=0; i<N; ++i)
    BOOST_CHECK(
        (x.col(i) - S.Log_p_north(muEst.col(z(i)),sq->col(i))).norm() < 1e-6);
  
 
}

typedef float myFlt;

BOOST_AUTO_TEST_CASE( sphereGpu_test)
{
  cout<<"----------------------- sphere Gpu ----------------------"<<endl;
  uint32_t D = 3;
  uint32_t K = 6;
  uint32_t N = 20;
  boost::mt19937 rndGen(1112);
  Sphere<myFlt> Sd(D);
  
  spVectorXu sz(new VectorXu(N));
  sz->topRows(N/2) = VectorXu::Zero(N/2);
  sz->bottomRows(N/2) = VectorXu::Ones(N/2);
  boost::shared_ptr<Matrix<myFlt,Dynamic,Dynamic> > sq(
      new Matrix<myFlt,Dynamic,Dynamic>(D,N));

  Matrix<myFlt,Dynamic,Dynamic> mus = sampleClustersOnSphere<myFlt>(*sq, 2);
//  for (uint32_t i=0; i<N; ++i) 
//  {
//    sq->col(i) = Sd.sampleUnif(&rndGen);
//    if(i<N/2)
//      (*sz)(i) = 0;
//    else
//      (*sz)(i) = 1;
//  }
  Matrix<myFlt,Dynamic,Dynamic> ps(D,K);
  for (uint32_t k=0; k<K; ++k)
     ps.col(k) =  Sd.sampleUnif(&rndGen);
//  ps.leftCols<2>() << 0.0,0.0,
//        1.0,-1.0,
//        0.0,0.0;
  Matrix<myFlt,Dynamic,1> w(N/2);
  w.setOnes(N/2);
  cout<<" karcher means on CPU"<<endl;
  cout<< karcherMeanWeighted<myFlt>(ps.col(0), sq->leftCols(N/2), w, 20) <<endl;
  cout<< karcherMeanWeighted<myFlt>(ps.col(1), sq->rightCols(N/2), w, 20) <<endl;
  
//  assert(false);

  ClSphereGpu<myFlt> sphere(sq,sz,K);
  //sphere.updateClusters(ps,z);
  Matrix<myFlt,Dynamic,Dynamic> karcherMeans = sphere.karcherMeans(ps);
  cout<< karcherMeans <<endl;
  cout<< "true centers"<<endl << mus <<endl;
  BOOST_CHECK(((karcherMeans.leftCols(2) - mus).array().abs() < 1e-2).all());
  
  Timer t;
  Matrix<myFlt,Dynamic,Dynamic> x = sphere.Log_p_north(ps);
  t.toctic("Log_p_north GPU");
  BOOST_CHECK(x.rows() == static_cast<myFlt>(D-1));
  BOOST_CHECK(x.cols() == static_cast<myFlt>(N));

  for(uint32_t i=0; i<N; ++i)
  {
    BOOST_CHECK(x.col(i).norm() < 3.141592653589793);
    BOOST_CHECK(fabs(x.col(i).norm()
          -acosf(ps.col((*sz)(i)).transpose()*sq->col(i)) ) < 1e-4); 
    myFlt err = fabs(x.col(i).norm() 
        -acosf(ps.col((*sz)(i)).transpose()*sq->col(i)));
    if(err >= 1e-4)
      cout<< err<<endl;
  }

  //cout<<x.transpose()<<endl;

  Matrix<myFlt,Dynamic,Dynamic> xx(D-1,N);
  t.tic();
  xx.leftCols(N/2)  = Sd.Log_p_north(ps.col(0),sq->leftCols(N/2));
  xx.rightCols(N/2) = Sd.Log_p_north(ps.col(1),sq->rightCols(N/2));
  t.toctic("Log_p_north CPU");
  
  //cout<< xx.transpose()<<endl;
  BOOST_CHECK( ((x-xx).array().abs() < 1.0e-4).all());

  cout<<"sufficient statistics test --------------------"<<endl;
  sphere.relinearize(karcherMeans);
  sphere.computeSufficientStatistics();
  for (uint32_t k=0; k<K; ++k)
    cout<<"S_"<<k<<endl<<sphere.S(k)<<endl;

  cout<<"pdf test --------------------------------------"<<endl;
 
  Matrix<myFlt,Dynamic,1> pi(K);
  pi.setOnes(); pi /= K;
  SamplerGpu<myFlt> sampler(N,K,&rndGen);
  vector<Matrix<myFlt,Dynamic,Dynamic> > Sigmas(K);
  for (uint32_t k=0; k<K; ++k)
    if(sphere.counts()(k) > 0)
      Sigmas[k] = sphere.S(k)/sphere.counts()(k);
    else
      Sigmas[k] = Matrix<myFlt,Dynamic,Dynamic>::Identity(D-1,D-1)*0.01;
  Matrix<myFlt,Dynamic,1> logNormalizers(K);
  for(uint32_t k=0; k<K; ++k)
  {
    NormalSphere<myFlt> normS(karcherMeans.col(k),Sigmas[k],&rndGen); 
    logNormalizers(k) = -0.5*normS.logDetSigma();
  }

  sphere.sampleGMMpdf(pi, Sigmas , logNormalizers, &sampler);
  Matrix<myFlt,Dynamic,Dynamic> pdfs = sphere.pdfs();

  for(uint32_t i=0; i<N; ++i)
  {
    Matrix<myFlt,Dynamic,1> logPdf(K);
    for(uint32_t k=0; k<K; ++k)
    {
      NormalSphere<myFlt> normS(karcherMeans.col(k),Sigmas[k],&rndGen);
      logPdf(k) = 0.5*(LOG_2PI*D+normS.logDetSigma()) + normS.logPdf(sq->col(i));; 
      //log(pi(k)) + normS.logPdf(sq->col(i));
//      logPdf(k) = normS.normal_.; //log(pi(k)) + normS.logPdf(sq->col(i));
    }
    Matrix<myFlt,Dynamic,1> A = logPdf;
    myFlt maxLog = logPdf.maxCoeff();
    logPdf = logPdf.array() - log((logPdf.array() - maxLog).exp().matrix().sum())
      - maxLog;
//    cout<<pdfs(i,0)<<" vs "<< exp(logPdf(0))<<endl;
    cout<<"--------------------------------"<<endl;
    cout<<pdfs.row(i)<<endl;
//    cout<<logPdf.array().exp().matrix().transpose()<<endl;
//    cout<<"-"<<endl;
//    cout<<(pdfs.row(i)-logPdf.array().exp().matrix().transpose())<<endl;
//    cout<<"log"<<endl;
//    cout<<logPdf.transpose()<<endl;

//    cout<<pi.array().log().matrix().transpose()<<endl; // DONE: same!
    cout<<(A.array()).matrix().transpose()<<endl;
  }
};


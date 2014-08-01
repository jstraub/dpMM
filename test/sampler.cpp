
#include <iostream>
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE sampler test
#include <boost/test/unit_test.hpp>

#include "sampler.hpp"
#include "distribution.hpp"
#include "timer.hpp"

using namespace Eigen;

using std::cout;
using std::endl;

#define N 640*480
#define M 10

BOOST_AUTO_TEST_CASE( sampler_cpu_test)
{
  cout<<"----------------- sampler CPU float (N="<<N<<")------------------"<<endl;
  Sampler<float> s;
  //cout<<s.sampleUnif(10).transpose()<<endl;

  Timer t;
  VectorXf u = s.sampleUnif(N);
  t.toctic("unif");
  BOOST_CHECK((u.array()<=1.0).all());
  BOOST_CHECK((u.array()>=.0).all());

  MatrixXf pdfs(4,3);
  pdfs << 0.5,0.5,0.0,
       0.25,0.25,0.5,
       0.33,0.33,0.34,
       0.9,0.05,0.05;
  cout<<s.sampleDiscPdf(pdfs).transpose()<<endl;

  MatrixXf logPdfs(4,3);
  logPdfs << -100,-20,-10,
    -100,-10,-30,
    -10,-20,-100,
    -100,-20,-10;
  cout<<s.sampleDiscLogPdfUnNormalized(logPdfs).transpose()<<endl;

  pdfs.setOnes(N,M);
  pdfs *= 1.0/M;
  
  t.tic();
  VectorXu z = s.sampleDiscPdf(pdfs);
  t.toctic("discPdf");
  BOOST_CHECK((z.array()<=M).all());
  BOOST_CHECK((z.array()>=0).all());

  cout<<counts(z,M).transpose()<<endl;
}

BOOST_AUTO_TEST_CASE( sampler_cpu_double_test)
{
  cout<<"----------------- sampler CPU double (N="<<N<<")------------------"<<endl;
  Sampler<double> s;
  //cout<<s.sampleUnif(10).transpose()<<endl;

  Timer t;
  VectorXd u = s.sampleUnif(N);
  t.toctic("unif");
  BOOST_CHECK((u.array()<=1.0).all());
  BOOST_CHECK((u.array()>=.0).all());

  MatrixXd pdfs(4,3);
  pdfs << 0.5,0.5,0.0,
       0.25,0.25,0.5,
       0.33,0.33,0.34,
       0.9,0.05,0.05;
  cout<<s.sampleDiscPdf(pdfs).transpose()<<endl;

  pdfs.setOnes(N,M);
  pdfs *= 1.0/M;
  
  t.tic();
  VectorXu z = s.sampleDiscPdf(pdfs);
  t.toctic("discPdf");
  BOOST_CHECK((z.array()<=M).all());
  BOOST_CHECK((z.array()>=0).all());

  cout<<counts(z,M).transpose()<<endl;
}


BOOST_AUTO_TEST_CASE( sampler_gpu_test)
{
  cout<<"----------------- sampler GPU float (N="<<N<<")------------------"<<endl;
  SamplerGpu<float> s(N,M);
  //cout<<s.sampleUnif().transpose()<<endl;

  Timer t;
  VectorXf u = s.sampleUnif();
  t.toctic("unif");
  BOOST_CHECK((u.array()<=1.0f).all());
  BOOST_CHECK((u.array()>=.0f).all());

  MatrixXf pdfs(N,M);
  pdfs.setOnes(N,M);
  pdfs *= 1.0/M;

  //cout<<s.sampleDiscPdf(pdfs).transpose()<<endl;

  t.tic();
  VectorXu z = s.sampleDiscPdf(pdfs);
  t.toctic("discPdf");
  BOOST_CHECK((z.array()<=M).all());
  BOOST_CHECK((z.array()>=0).all());
  
  cout<<counts(z,M).transpose()<<endl;

  pdfs.setZero(N,M);
  pdfs.col(3).setOnes();
//  cout<<pdfs<<endl;

  t.tic();
  z = s.sampleDiscPdf(pdfs);
  t.toctic("discPdf");
  cout<<counts(z,M).transpose()<<endl;
}

BOOST_AUTO_TEST_CASE( sampler_gpu_double_test)
{
  cout<<"----------------- sampler GPU double (N="<<N<<")------------------"<<endl;
  SamplerGpu<double> s(N,M);
  //cout<<s.sampleUnif().transpose()<<endl;

  Timer t;
  VectorXd u = s.sampleUnif();
  t.toctic("unif");
  BOOST_CHECK((u.array()<=1.0f).all());
  BOOST_CHECK((u.array()>=.0f).all());

  MatrixXd pdfs(N,M);
  pdfs.setOnes(N,M);
  pdfs *= 1.0/M;

  //cout<<s.sampleDiscPdf(pdfs).transpose()<<endl;

  t.tic();
  VectorXu z = s.sampleDiscPdf(pdfs);
  t.toctic("discPdf");
  BOOST_CHECK((z.array()<=M).all());
  BOOST_CHECK((z.array()>=0).all());
  
  cout<<counts(z,M).transpose()<<endl;

  pdfs.setZero(N,M);
  pdfs.col(3).setOnes();
//  cout<<pdfs<<endl;

  t.tic();
  z = s.sampleDiscPdf(pdfs);
  t.toctic("discPdf");
  cout<<counts(z,M).transpose()<<endl;
}

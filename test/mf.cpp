/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */

#include <iostream>
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE mf test
#include <boost/test/unit_test.hpp>

#include <mmf/mfPrior.hpp>
#include <mmf/mf.hpp>

BOOST_AUTO_TEST_CASE(mf_test)
{
  boost::mt19937 rndGen(1);
  double nu = 4;
  MatrixXd Delta = MatrixXd::Identity(2,2);
  Delta *= pow(5.0*M_PI/180.,2)*nu;
  IW<double> iw0(Delta,nu,&rndGen);
  std::vector<shared_ptr<BaseMeasure<double> > >thetas;
  for(uint32_t k=0; k<6; ++k)
    thetas.push_back(shared_ptr<IwTangent<double> >(
          new IwTangent<double>(iw0,&rndGen)));
  VectorXd alpha = VectorXd::Ones(6);
  Dir<Cat<double>, double> dir(alpha,&rndGen);
  DirMM<double> dirMM(dir,thetas);
  MfPrior<double> mfPrior(dirMM,10);

  MatrixXd Sigma = MatrixXd::Identity(2,2);
  Sigma *= pow(1.0*M_PI/180.,2);
  uint32_t N = 100;
  MatrixXd x(3,N*6);
  MatrixXd R(3,3);
  double theta = 30.*M_PI/180.;
  R<<cos(theta), -sin(theta), 0.,
     sin(theta),cos(theta) ,0.,
     0,0,1.;
  cout<<R<<endl;
  cout<<" ..................... "<<endl;
  for(uint32_t k=0; k<6; ++k)
  {
    NormalSphere<double> g(R*mfPrior.M().col(k),Sigma,&rndGen);
    for(uint32_t i =0; i< N; ++i)
      x.col(i+N*k) = g.sample(); 
  }
  VectorXu z = VectorXu::Zero(N*6);
  MF<double> mf = mfPrior.posteriorSample(x,z,0);

  mf.print();

  for(uint32_t i =0; i< N*6; i+=N)
  {
    cout<<mf.logPdf(x.col(i))<<endl;
  }
  MF<double> mf2(mf);
  mf2.print();
};

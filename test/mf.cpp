/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */

#include <iostream>
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE mf test
#include <boost/test/unit_test.hpp>

#include <mmf/mfPrior.hpp>

BOOST_AUTO_TEST_CASE(mf_test)
{
  boost::mt19937 rndGen(1);
  MatrixXd Delta = MatrixXd::Identity(2,2);
  Delta *= pow(15.0*M_PI/180.,2);
  IW<double> iw0(Delta,4,&rndGen);
  std::vector<shared_ptr<IwTangent<double> > >thetas;
  for(uint32_t k=0; k<6; ++k)
    thetas.push_back(shared_ptr<IwTangent<double> >(
          new IwTangent<double>(iw0,&rndGen)));
  VectorXd alpha = VectorXd::Ones(6);
  Dir<Cat<double>, double> dir(alpha,&rndGen);
  MfPrior<double> mfPrior(dir,thetas,10);


  VectorXd x(3,6);
  x << 1.0,-1.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 1.0,-1.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 1.0,-1.0;

  VectorXu z = VectorXu::Zero(6);
  mfPrior.posterior(x,z,0);
  
};

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

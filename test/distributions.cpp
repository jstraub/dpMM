/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>, Randi Cabezas <rcabezas@csail.mit.edu>                    
 * Licensed under the MIT license. See the license file LICENSE.
 */

#include <iostream>
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE distributions test
#include <boost/test/unit_test.hpp>

#include "dir.hpp"
#include "cat.hpp"
#include "niw.hpp"
#include "iw.hpp"
#include "normal.hpp"

#include <omp.h>

//#include "matrix.h"
//#include "mex.h"
//#include "helperMEX.h"
//#include "debugMEX.h"

#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"
#include "gsl/gsl_permutation.h"
#include "gsl/gsl_cdf.h"

#include "dpmmSubclusters/common.h"

#include "dpmmSubclusters/niw_sampled.h"
//#include "dpmmSubclusters/cluster_sampledT.cpp"
#include "dpmmSubclusters/linkedList.cpp"

//#include "dpmmSubclusters/reduction_array.h"
//#include "dpmmSubclusters/reduction_array2.h"
#include "dpmmSubclusters/linear_algebra.h"
//#include "dpmmSubclusters/sample_categorical.h"
//#include "dpmmSubclusters/niw_sampled.h"

using std::cout;
using std::endl;

BOOST_AUTO_TEST_CASE( dir_test)
{
  cout<<"----------------------- dir ----------------------"<<endl;
  VectorXd alpha(3);
  alpha << 1.0,100.0,1.0;

  boost::mt19937 rndGen(1);
  Dir<Cat<double>,double> dir(alpha,&rndGen);
  VectorXd piPdf = dir.sample().pdf();
  
  BOOST_CHECK_EQUAL(piPdf.size(),alpha.size());

  cout<<"-- sampling a bit"<<endl;
  cout<<"alpha="<<alpha.transpose()<<endl;
  for(uint32_t t=0; t<10; ++t)
    cout<<"piPdf="<<dir.sample().pdf().transpose()<<endl;
  
  boost::mt19937 rndGen2(1);
  Dir<Catd,double> dir2(alpha,&rndGen2);
  VectorXd piPdf2 = dir2.sample().pdf();
  BOOST_CHECK_EQUAL(piPdf,piPdf2);

  Dir<Catd,double> dir3(dir);
  VectorXd piPdf3 = dir3.sample().pdf();
  BOOST_CHECK_EQUAL(piPdf.size(),piPdf3.size());

  VectorXf alphaf = alpha.cast<float>();
  Dir<Catf,float> dirf(alphaf,&rndGen);
}

BOOST_AUTO_TEST_CASE( cat_test)
{
  cout<<"----------------------- cat ----------------------"<<endl;
  VectorXd pdf(3);
  pdf << 0.5,0.25,0.25;

  boost::mt19937 rndGen(1);
  Cat<double> cat(pdf,&rndGen);
  double z1 = cat.sample();
  boost::mt19937 rndGen2(1);
  Cat<double> cat2(pdf,&rndGen2);
  double z2 = cat2.sample();
  BOOST_CHECK_EQUAL(z1,z2);

  Cat<double> cat3(cat);
  double z3 = cat3.sample();

  BOOST_CHECK_EQUAL(cat.pdf_.size(), cat3.pdf_.size());
  

  VectorXu z(1000);
  cat.sample(z);
  cout<<"-- sampling a bit"<<endl;
  cout<<"pdf="<<pdf.transpose()<<endl;
  cout<<"z="<<z.transpose()<<endl;
  
  VectorXd pdfEmp = Cat<double>(z,&rndGen).pdf();
  BOOST_CHECK_EQUAL(pdfEmp.size(), cat.pdf().size());
  cout<<"pdf from counts="<<pdfEmp.transpose()<<endl;
  
  VectorXf pdff = pdf.cast<float>();
  Cat<float> catf(pdff,&rndGen);
}

BOOST_AUTO_TEST_CASE(iw_test)
{
  cout<<"----------------------- iw ----------------------"<<endl;
  
  MatrixXd Delta(3,3);
  Delta << 1.0,0.0,0.0,
        0.0,1.0,0.0,
        0.0,0.0,1.0;
  double nu = 100.0;

  boost::mt19937 rndGen(1);
  IW<double> iw(Delta,nu,&rndGen);

  for(uint32_t t=0; t<10; ++t)
  {
    MatrixXd Sigma = iw.sample();
    cout<<"Delta=\n"<<iw.Delta_<<endl;
    cout<<"Sigma=\n"<<Sigma<<endl;
    cout<<"logPdf="<<iw.logPdf(Sigma)<<endl;
  }

}



BOOST_AUTO_TEST_CASE( gauss_test)
{
  cout<<"----------------------- gauss ----------------------"<<endl;
  
  MatrixXd Sigma(3,3);
  Sigma << 1.0,0.0,0.0,
        0.0,1.0,0.0,
        0.0,0.0,1.0;
  VectorXd mu(3);
  mu << 1.0,1.0,1.0;

  boost::mt19937 rndGen(1);
  Normal<double> normal(mu,Sigma,&rndGen);

  VectorXd x(3);
  x << 1.0,1.0,1.0;
  cout << normal.logPdf(x) <<endl;
  x << 1.0,1.0,0.0;
  cout << normal.logPdf(x) <<endl;
  double a = normal.logPdf(x);
  x << 0.0,1.0,1.0;
  BOOST_CHECK_EQUAL(a,normal.logPdf(x));
  cout << normal.logPdf(x) <<endl;
  x << 0.0,0.0,0.0;
  cout << normal.logPdf(x) <<endl;
  BOOST_CHECK(a>normal.logPdf(x));


  Sigma << 5.0,0.0,0.0,
        0.0,1.0,0.0,
        0.0,0.0,0.1;
  mu << 1.0,1.0,0.0;
  Normal<double> normalA(mu,Sigma,&rndGen);
  for (uint32_t i=0; i<100; ++i)
  {
    x = normalA.sample();
    cout<<"x="<<x.transpose()<<" logPdf="<<normalA.logPdf(x)<<endl;
  }

  cout<<"small variance and far away data"<<endl;
  MatrixXd Sigma2D(2,2);
  VectorXd mu2D(2);
  VectorXd x2D(2);
  Sigma2D << 1.0e-12,0.0,
        0.0,1.0e-12;
  mu2D << 0.0,0.0;
  x2D << M_PI*0.5,0.0; // 90 degree away
  Normal<double> normalS(mu2D,Sigma2D,&rndGen);
  cout<<" logPDf "<<normalS.logPdf(x2D)<<endl;

}

//BOOST_AUTO_TEST_CASE( niw_test) { cout<<"----------------------- niw
//----------------------"<<endl;
//  
//  uint32_t D =3;
//  MatrixXd Delta(D,D);
//  Delta << 1.0,0.0,0.0,
//        0.0,1.0,0.0,
//        0.0,0.0,1.0;
//  VectorXd theta(D);
//  theta << 1.0,1.0,1.0;
//  double nu = 100.0;
//  double kappa = 100.0;
//
//  boost::mt19937 rndGen(1);
//  NIW<double> niw(Delta,theta,nu,kappa,&rndGen);
//
//  for(uint32_t t=0; t<10; ++t)
//  {
//    Normal<double> Norm = niw.sample();
//	Norm.print(); 
//    cout<<"logPdf="<<niw.logPdf(Norm)<<endl;
//  }
//  cout<<" comparing NIW against Jasons implementation ------"<<endl;
//  niw.print();
//
//  uint32_t N=80;
//  uint32_t K=1;
//  VectorXu z(N);
//  shared_ptr<MatrixXd> spx(new MatrixXd(D,N));
//  cout<<"true mus:"<<endl<<sampleClusters<double>(*spx, z, K)<<endl;
//
//  double count = N;
//  MatrixXd Outer(D,D); Outer.setZero();
//  VectorXd sum(D); sum.setZero();
//  for(uint32_t i=0; i<N; ++i)
//  {
//    sum += spx->col(i);
//    Outer += spx->col(i)*spx->col(i).transpose();
//  }
//  cout<<"Outer"<<endl<<Outer<<endl; 
//  cout<<"sum"<<endl<<sum<<endl; 
//  cout<<"1/N*sum*sum.T"<<endl<<sum*sum.transpose()/count<<endl; 
//  cout<<"sum*sum.T"<<endl<<sum*sum.transpose()<<endl; 
//  MatrixXd Scatter = Outer - sum*sum.transpose() / count;
//  VectorXd mean = sum/count;
//  cout<<"Scatter"<<endl<<Scatter<<endl; 
//
//  niw.scatter() = Scatter;
//  niw.mean() = mean;
//  niw.count() = count;
//
//  cout<<"--- Julians posterior:"<<endl;
//  niw.posterior().print();
//
//  cout<<"--- Julians posterior direct:"<<endl;
//  niw.posterior(*spx,z,0).print();
//
//
//  MatrixXd DeltaOverNu(Delta);
//  DeltaOverNu /= nu;
//  VectorXd thetaOverKappa(theta);
////  thetaOverKappa /= kappa;
//  niw_sampled niwJason(3,kappa,nu,thetaOverKappa.data(),DeltaOverNu.data());
//  niwJason.set_stats(count,sum.data(),Outer.data());
//  niwJason.update_posteriors();
// 
//  Map<MatrixXd> DeltaPost(niwJason.Delta,D,D);
//  Map<VectorXd> thetaPost(niwJason.theta,D);
//  cout<<"--- Jason posterior:"<<endl;
//  cout<<"nu="<<niwJason.nu<<" kappa="<<niwJason.kappa<<endl; 
//  cout<<"delta"<<endl<<DeltaPost<<endl;
//  cout<<"theta "<<thetaPost.transpose()<<endl;
//  cout<<"--- Jasons adapted to julians:"<<endl;
//  cout<<"delta"<<endl<<DeltaPost*niwJason.nu<<endl;
//  cout<<"theta "<<thetaPost.transpose()*niwJason.kappa<<endl;
//  
//  cout<<" ----- marginal probability of data under NIW"<<endl;
//  cout<<" Julian: "<<niw.logPdfMarginalized()<<endl;
//  cout<<" Jason:  "<<niwJason.data_loglikelihood_marginalized()<<endl;
//  
//  BOOST_CHECK(fabs(niw.logPdfMarginalized() - niwJason.data_loglikelihood_marginalized())<1e-3);
//
//  cout<<"------------------ merge test -----------------------"<<endl;
//
//  NIW<double> niwB(Delta,theta,nu,kappa,&rndGen);
//
//  double countB = 1000;
//  MatrixXd OuterB(D,D);
//  OuterB<< 10,0,0,
//          0,100,0,
//          0,0,10;
//  VectorXd sumB(D);
//    sumB << 1,10,0;
//  MatrixXd ScatterB = OuterB ;//- count*sum*sum.transpose();
//  VectorXd meanB = sumB/countB;
//  cout<<"Scatter"<<endl<<ScatterB<<endl; 
//
//  niwB.scatter() = ScatterB;
//  niwB.mean() = meanB;
//  niwB.count() = countB;
//
//  NIW<double>* niwBmerged = niwB.merge(niw);
//  cout<<"--- Julians posterior after merge:"<<endl;
//  niwBmerged->posterior().print();
//
//  MatrixXd DeltaOverNuB(Delta);
//  DeltaOverNuB /= nu;
//  VectorXd thetaOverKappaB(theta);
////  thetaOverKappa /= kappa;
//  niw_sampled niwJasonB(3,kappa,nu,thetaOverKappaB.data(),DeltaOverNuB.data());
//  niwJasonB.set_stats(countB,sumB.data(),OuterB.data());
//  niwJasonB.update_posteriors();
//  Map<MatrixXd> DeltaPostB(niwJasonB.Delta,D,D);
//  Map<VectorXd> thetaPostB(niwJasonB.theta,D);
//
//  niwJasonB.merge_with(niwJason, false);
////  niwJasonB.update_posteriors();
//  cout<<"--- Jason posterior after merge:"<<endl;
//  cout<<"nu="<<niwJasonB.nu<<" kappa="<<niwJasonB.kappa<<endl; 
//  cout<<"delta"<<endl<<DeltaPostB<<endl;
//  cout<<"theta "<<thetaPostB.transpose()<<endl;
//  cout<<"--- Jasons adapted to julians:"<<endl;
//  cout<<"delta"<<endl<<DeltaPostB*niwJasonB.nu<<endl;
//  cout<<"theta "<<thetaPostB.transpose()*niwJasonB.kappa<<endl;
//
//  cout<<" ------------------- sampling ----------------------------"<<endl;
//  for(uint32_t t=0; t<5; ++t)
//  {
//    cout<<"Jason:"<<endl;
//    niwJasonB.sample();
//    Map<MatrixXd> jasonCov(niwJasonB.param.cov,D,D);
//    Map<VectorXd> jasonMean(niwJasonB.param.mean,D);
//    cout<<jasonMean.transpose()<<endl;
//    cout<<jasonCov<<endl;
//    cout<<"Julian:"<<endl;
//    Normal<double> normB = niwB.posterior().sample();
//    normB.print();
//
//    //cout<<normB.mu_.transpose()<<endl;
//    //cout<<normB.Sigma_<<endl;
//  }
//}

/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */

#include <iostream>
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE clData test
#include <boost/test/unit_test.hpp>

#include "clSphereGpu.hpp"
#include "timer.hpp"
#include "clData.hpp"

using std::cout;
using std::endl;


BOOST_AUTO_TEST_CASE( clData_test)
{
  cout<<"----------------------- clData ----------------------"<<endl;
  uint32_t D = 3;
  uint32_t K = 2;
  uint32_t N = 1000;
  
  spVectorXu z(new VectorXu(N));
  spMatrixXf x(new MatrixXf(D,N));
  for (uint32_t i=0; i<N; ++i) 
  {
    if(i<N/2){
      (*z)(i) = 0;
      x->col(i).setZero();
    }else{
      (*z)(i) = 1;
      x->col(i).setOnes();
    }
  }

  ClDataf data(x,z,K);
  data.update(K);

  cout<<"Ns="<<endl<<data.counts()<<endl;
  cout<<"means="<<endl<<data.means()<<endl;
  cout<<"scatters="<<endl;
  for(uint32_t k=0; k<K; ++k)
    cout<<data.scatters()[k]<<endl;
}

BOOST_AUTO_TEST_CASE( clSphereGpu_test)
{
  cout<<"----------------------- clSphereGpu ----------------------"<<endl;
  uint32_t D = 3;
  uint32_t K = 6;
  uint32_t N = 10000;
  
  spVectorXu z(new VectorXu(N));
  spMatrixXf x(new MatrixXf(D,N));
  boost::mt19937 rndGen(1112);

  VectorXd mu(D);
  mu<<0.0,0.0,1.0;
  MatrixXd Sigma = MatrixXd::Identity(D,D);
  Normald gauss(mu,Sigma,&rndGen);
  for (uint32_t i=0; i<N; ++i) 
  {
    if(i<N/2){
      (*z)(i) = 5;
      x->col(i) = gauss.sample().cast<float>();
      x->col(i) /= x->col(i).norm();
    }else{
      (*z)(i) = 1;
      x->col(i) = gauss.sample().cast<float>();
      x->col(i) /= -1*x->col(i).norm();
    }
//    cout << x->col(i) <<endl;
  }

  ClSphereGpu<float> data(x,z,K);
  data.update(K);

  cout<<"Ns="<<endl<<data.counts()<<endl;
  cout<<"means="<<endl<<data.means()<<endl;
  cout<<"scatters="<<endl;
  for(uint32_t k=0; k<K; ++k)
    cout<<data.scatters()[k]<<endl;
}

//BOOST_AUTO_TEST_CASE( sphereGpu_test)
//{
//  cout<<"----------------------- sphere Gpu ----------------------"<<endl;
//  uint32_t D = 3;
//  uint32_t K = 2;
//  uint32_t N = 100000;
//  boost::mt19937 rndGen(1112);
//  Sphere<> Sd;
//  
//  VectorXu z(N);
//  MatrixXf q(D,N);
//  for (uint32_t i=0; i<N; ++i) 
//  {
//    q.col(i) = Sd.sampleUnif(D,&rndGen).cast<float>();
//    if(i<N/2)
//      z(i) = 0;
//    else
//      z(i) = 1;
//  }
//  MatrixXf ps(D,2);
//  ps << 0.0,0.0,
//        1.0,-1.0,
//        0.0,0.0;
//
//  ClSphereGpu sphere(q,K);
//  //sphere.updateClusters(ps,z);
//  
//  Timer t;
//  MatrixXf x = sphere.Log_p_north(ps,z);
//  t.toctic("Log_p_north GPU");
//  BOOST_CHECK(x.rows() == D-1);
//  BOOST_CHECK(x.cols() == N);
//
//  for(uint32_t i=0; i<N; ++i)
//  {
//    BOOST_CHECK(x.col(i).norm() < 3.141592653589793);
//    BOOST_CHECK(fabs(x.col(i).norm() - acosf(ps.col(z(i)).transpose()*q.col(i)) ) < 1e-4); 
//    float err = fabs(x.col(i).norm() - acosf(ps.col(z(i)).transpose()*q.col(i)));
//    if(err >= 1e-4)
//      cout<< err<<endl;
//  }
//
//  //cout<<x.transpose()<<endl;
//
//  MatrixXf xx(D-1,N);
//  t.tic();
//  xx.leftCols(N/2)  = Sd.Log_p_north(ps.col(0).cast<double>(),
//    q.leftCols(N/2).cast<double>()).cast<float>();
//  xx.rightCols(N/2) = Sd.Log_p_north(ps.col(1).cast<double>(),
//    q.rightCols(N/2).cast<double>()).cast<float>();
//  t.toctic("Log_p_north CPU");
//  
//  //cout<< xx.transpose()<<endl;
//
//  BOOST_CHECK( ((x-xx).array().abs() < 1.0e-4).all());
//
//}

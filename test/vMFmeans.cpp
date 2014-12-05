

#include <iostream>
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE spkm test
#include <boost/test/unit_test.hpp>

#include <boost/shared_ptr.hpp>

#include <stdint.h>

#include "normalSphere.hpp"
#include "sphericalKMeans.hpp"
#include "dpvMFmeans.hpp"
#include "ddpvMFmeans.hpp"
#include "ddpvMFmeansCUDA.hpp"
#include "timer.hpp"

using namespace Eigen;
using std::cout;
using std::endl;

BOOST_AUTO_TEST_CASE(spkm_test)
{
  boost::mt19937 rndGen(91);
  
  //uint32_t N=100000;
  uint32_t N=19;
  uint32_t D=3;
  uint32_t K=2;
  boost::shared_ptr<MatrixXd> spxOrig(new MatrixXd(D,N));
  sampleClustersOnSphere<double>(*spxOrig, K);
  boost::shared_ptr<MatrixXd> spx(new MatrixXd(D,N));
  (*spx) = (*spxOrig);

  uint32_t T=10;
  cout<<" -------------------- spkm ----------------------"<<endl;
  SphericalKMeans<double> spkm(spx,K,&rndGen);
  for(uint32_t t=0; t<T; ++t)
  {
    spkm.updateCenters();
    spkm.updateLabels();
    if (N<20) cout<<spkm.z().transpose()<<" "<<spkm.avgIntraClusterDeviation()<<endl;
//    cout<<spkm.centroids()<<endl;
  }

  double lambda = cos(45.0*M_PI/180.0)-1;
  double beta = 0.5 ; //cos(5.0*M_PI/180.0);
  double Q = -0.01 ;//-cos(1.0*M_PI/180.0);
  cout<<" -------------------- DDP-vMF-means ----------------------"<<endl;
  DDPvMFMeans<double> ddpvmfmeans(spx,lambda,beta,Q,&rndGen);

  double dAng = 5.0*M_PI/180.0;
  MatrixXd dR = MatrixXd::Zero(3,3);
  dR << cos(dAng), sin(dAng), 0,
       -sin(dAng), cos(dAng), 0,
       0         , 0        , 1;

  MatrixXd means = spkm.centroids();
  Timer t0,t1;
  for(uint32_t t=0; t<15; ++t)
  {
    cout<<" -- t = "<<t<<endl;

    if(t==7 || t == 12)
    {
      boost::shared_ptr<MatrixXd> spx3(new MatrixXd(D,N*2));
      //      boost::shared_ptr<MatrixXd> spxTmp(new MatrixXd(D,N));
      //      sampleClustersOnSphere<double>(*spxTmp, 1);
      spx3->leftCols(N) = *spx;
      spx3->rightCols(N) = - (*spx);
      spx = spx3;
    }

    if(t==10)
    {
      boost::shared_ptr<MatrixXd> spx3(new MatrixXd(D,N));
      //      boost::shared_ptr<MatrixXd> spxTmp(new MatrixXd(D,N));
      //      sampleClustersOnSphere<double>(*spxTmp, 1);
      *spx3 = spx->leftCols(N);
      spx = spx3;
    }

    if(t>=3)
    {
      *spx = dR * (*spx);

      means = dR*means;
      cout<<"new means:"<<endl
        <<means.transpose()<<endl
        <<" ----------------------------- "<<endl;
    }


      t1.tic();
    ddpvmfmeans.nextTimeStep(spx); // feed in new data (here just the same
      t1.toctic("nextTimestep");
    for(uint32_t i=0; i<10; ++i)
    { // run clustering till "converence"
//      cout<<"========================= label udaptes ========================="<<endl;
      t1.tic();
//      ddpvmfmeans.updateLabelsParallel();
      ddpvmfmeans.updateLabels();
//      cout<<ddpvmfmeans.z().transpose()<<" "
//        <<ddpvmfmeans.avgIntraClusterDeviation()<<endl;
//      cout<<"========================= center udaptes ========================="<<endl;
      t1.toctic("updateLabels");
      ddpvmfmeans.updateCenters();
      t1.toctic("updateCenters");
      if(N<20)
        cout<<ddpvmfmeans.z().transpose()<<" "
          <<ddpvmfmeans.avgIntraClusterDeviation()<<endl;
      //    cout<<spkm.centroids()<<endl;
    }
      t1.tic();
    ddpvmfmeans.updateState(); // update the state internally
      t1.toctic("updateState");
  }
  t0.toctic(" -----------  DDP-vMF-means ");


  if(false)
  {
  (*spx) = (*spxOrig); // reset spx
  means = spkm.centroids();
  cout<<" -------------------- DDP-vMF-means CUDA ----------------------"<<endl;
  DDPvMFMeansCUDA<double> ddpvmfmeansCUDA(spx,lambda,beta,Q,&rndGen);

  t0.tic();
  for(uint32_t t=0; t<10; ++t)
  {
    cout<<" -- t = "<<t<<endl;

    if(t==7)
    {
      boost::shared_ptr<MatrixXd> spx3(new MatrixXd(D,N*2));
      //      boost::shared_ptr<MatrixXd> spxTmp(new MatrixXd(D,N));
      //      sampleClustersOnSphere<double>(*spxTmp, 1);
      spx3->leftCols(N) = *spx;
      spx3->rightCols(N) = - (*spx);
      spx = spx3;
    }

    if(t>=3)
    {
      *spx = dR * (*spx);
      means = dR*means;
      cout<<"new means:"<<endl
        <<means.transpose()<<endl
        <<" ----------------------------- "<<endl;
    }

    t1.tic();
    ddpvmfmeansCUDA.nextTimeStep(spx); // feed in new data (here just the same
    t1.toctic("nextStep -----------------------------------");
    for(uint32_t i=0; i<10; ++i)
    { // run clustering till "converence"
      t1.tic();
//      ddpvmfmeansCUDA.updateLabels();
      ddpvmfmeansCUDA.updateLabelsParallel();
      t1.toctic("updateLabels");
//      cout<<ddpvmfmeansCUDA.z().transpose()<<" "
//        <<ddpvmfmeansCUDA.avgIntraClusterDeviation()<<endl;
      ddpvmfmeansCUDA.updateCenters();
      t1.toctic("updateCenters");
      if(N<20)
        cout<<ddpvmfmeansCUDA.z().transpose()<<" "
          <<ddpvmfmeansCUDA.avgIntraClusterDeviation()<<endl;
      //    cout<<spkm.centroids()<<endl;
    }

    t1.tic();
    ddpvmfmeansCUDA.updateState(); // update the state internally
    t1.toctic("updateState");
  }
  t0.toctic(" -----------  DDP-vMF-means CUDA ");
  }
}


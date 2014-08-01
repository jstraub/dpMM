
#include <iostream>
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE spkm test
#include <boost/test/unit_test.hpp>

#include <boost/shared_ptr.hpp>

#include "kmeans.hpp"
#include "sphericalKMeans.hpp"
#include "normalSphere.hpp"

using namespace Eigen;
using std::cout;
using std::endl;

BOOST_AUTO_TEST_CASE(spkm_test)
{
  boost::mt19937 rndGen(91);
  
  uint32_t N=20;
  uint32_t D=3;
  uint32_t K=2;
  shared_ptr<MatrixXd> spx(new MatrixXd(D,N));
  sampleClustersOnSphere<double>(*spx, K);


  uint32_t T=10;
  cout<<" -------------------- spkm ----------------------"<<endl;
  SphericalKMeans<double> spkm(spx,K,&rndGen);
  for(uint32_t t=0; t<T; ++t)
  {
    spkm.updateLabels();
    cout<<spkm.z().transpose()<<" "<<spkm.avgIntraClusterDeviation()<<endl;
    spkm.updateCenters();
//    cout<<spkm.centroids()<<endl;
  }
  MatrixXd deviates;
  MatrixXu inds = spkm.mostLikelyInds(10,deviates);
  cout<<"most likely indices"<<endl;
  cout<<inds<<endl;

  cout<<" ---------------- spkm_karch -------------------"<<endl;
  boost::mt19937 rndGen2(91);
  SphericalKMeansKarcher<double> spkmKarch(spx,K,&rndGen2);
 
  for(uint32_t t=0; t<T; ++t)
  {
    spkmKarch.updateLabels();
    cout<<spkmKarch.z().transpose()<<" "<<spkmKarch.avgIntraClusterDeviation()<<endl;
    spkmKarch.updateCenters();
//    cout<<spkmKarch.centroids()<<endl;
  }
  inds = spkmKarch.mostLikelyInds(10,deviates);
  cout<<"most likely indices"<<endl;
  cout<<inds<<endl;

  cout<<" ---------------- kmeans -------------------"<<endl;
  boost::mt19937 rndGen3(91);
  KMeans<double> kmeans(spx,K,&rndGen3);
 
  for(uint32_t t=0; t<T; ++t)
  {
    kmeans.updateLabels();
    cout<<kmeans.z().transpose()<<" "<<kmeans.avgIntraClusterDeviation()<<endl;
    kmeans.updateCenters();
//    cout<<kmeans.centroids()<<endl;
  }
  inds = kmeans.mostLikelyInds(10,deviates);
  cout<<"most likely indices"<<endl;
  cout<<inds<<endl;
}

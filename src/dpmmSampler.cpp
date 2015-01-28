/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */

#include <iostream>
#include <fstream>
#include <string>
#include <boost/program_options.hpp>

#include "dpSubclusterMM.hpp"
#include "dpmmSampler.hpp"
#include "dirMM.hpp"
#include "dirMMcld.hpp"
#include "clSphereGpu.hpp"
#include "niwSphere.hpp"
#include "niwSphereFull.hpp"
//#include "niwTangent.hpp"
#include "niwBaseMeasure.hpp"
#include "unifSphere.hpp"
#include "sphericalKMeans.hpp"
#include "kmeans.hpp"
#include "crpMM.hpp"
#include "vmfBaseMeasure.hpp"
//#include "dpvMFmeans.hpp"
#include "timer.hpp"

using namespace Eigen;
using std::string; 
using std::ofstream; 
using std::ifstream; 
namespace po = boost::program_options;

typedef double flt;

int main(int argc, char **argv)
{

  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "produce help message")
    ("seed", po::value<int>(), "seed for random number generator")
    ("N,N", po::value<int>(), "number of input datapoints")
    ("D,D", po::value<int>(), "number of dimensions of the data")
    ("T,T", po::value<int>(), "number of iterations")
    ("alpha,a", po::value< vector<double> >()->multitoken(), 
      "alpha parameter of the DP (if single value assumes all alpha_i are the "
      "same")
    ("K,K", po::value<int>(), "number of initial clusters ")
    ("nopropose,n", "flag to disable the propsal of splits and merges")
    ("base", po::value<string>(), 
      "which base measure to use (NIW, DpNiw, DpNiwSphereFull, "
      " DpNiwSphere, NiwSphere, DirNiwSphereFull"
      " NiwSphereUnifNoise, spkm, spkmKarcher, kmeans, CrpvMF right now)")
    ("params,p", po::value< vector<double> >()->multitoken(), 
      "parameters of the base measure")
    ("brief", po::value< vector<double> >()->multitoken(), 
      "brief parameters of the base measure (ie Delta = delta*I; "
      "theta=t*ones(D)")
    ("input,i", po::value<string>(), 
      "path to input dataset .csv file (rows: dimensions; cols: different "
      "datapoints)")
    ("output,o", po::value<string>(), 
      "path to output labels .csv file (rows: time; cols: different "
      "datapoints)")
    ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);    

  if (vm.count("help")) {
    cout << desc << "\n";
    return 1;
  }

  uint64_t seed = time(0);
  if(vm.count("seed"))
    seed = static_cast<uint64_t>(vm["seed"].as<int>());
  boost::mt19937 rndGen(seed);
  uint32_t K=5;
  if (vm.count("K")) K = vm["K"].as<int>();
  // number of iterations
  uint32_t T=100;
  if (vm.count("T")) T = vm["T"].as<int>();
  uint32_t N=100;
  if (vm.count("N")) N = vm["N"].as<int>();
  uint32_t D=2;
  if (vm.count("D")) D = vm["D"].as<int>();
  cout << "T="<<T<<endl;
  bool proposeSplitMerge = true;
  if (vm.count("nopropose")) proposeSplitMerge = false; 
  if(proposeSplitMerge)
    cout<<"Propose splits and merges"<<endl;
  else
    cout<<"DO not propose splits or merges"<<endl;

  // DP alpha parameter
  VectorXd alpha;
  if (vm.count("alpha"))
  {
    vector<double> params = vm["alpha"].as< vector<double> >();
    if(params.size()==1)
    {
      alpha.setOnes(K);
      alpha *= params[0];
    }else{
      cout<<"Warning: alpha.size = "<<params.size()<<" K0 = "<<K<<endl;
      alpha.setOnes(params.size());
      for (uint32_t k=0; k<params.size(); ++k)
        alpha(k) = params[k];
    }
  }
  cout << "alpha="<<alpha.transpose()<<endl;

  shared_ptr<MatrixXd> spx(new MatrixXd(D,N));
  MatrixXd& x(*spx);
  string pathIn ="";
  if(vm.count("input")) pathIn = vm["input"].as<string>();
  if (!pathIn.compare(""))
  {
    for(uint32_t i=0; i<N; ++i)
      if(i<N/2)
      {
        x.col(i) << VectorXd::Zero(D);
      }else{
        x.col(i) << 2.0*VectorXd::Ones(D);
      }
  }else{
    cout<<"loading data from "<<pathIn<<endl;
    ifstream fin(pathIn.data(),ifstream::in);
    for (uint32_t j=0; j<D; ++j)
      for (uint32_t i=0; i<N; ++i)
        fin>>x(j,i);
    //cout<<x<<endl;
  }

  // which base distribution
  string base = "NIW";
  if(vm.count("base")) base = vm["base"].as<string>();

  if(base.compare("DpNiw")){
    // normalize to unit length
    int err = 0;
#pragma omp parallel for
    for (uint32_t i=0; i<N; ++i)
      if(fabs(x.col(i).norm() - 1.0) > 1e-1)
      {
        err++;
        cout<<x.col(i).norm() <<endl;
      }else
        x.col(i) /= x.col(i).norm();
    if(err>0) return 0;
  }

  
  DpMM<double> *dpmm=NULL;
  DpMM<double> *dpmmf=NULL;
  Clusterer<double> *spkm = NULL;
  if(!base.compare("NIW"))
  {
    MatrixXd Delta(D,D);
    VectorXd theta(D);
    double nu = 10.0;
    double kappa = 10.0;
//    Delta << nu,0.0,0.0,nu;
//    theta << 0.0,0.0;
    if(vm.count("params"))
    {
      vector<double> params = vm["params"].as< vector<double> >();
      cout<<"params length="<<params.size()<<" D="<<D<<endl;
      nu = params[0];
      kappa = params[1];
      for(uint32_t i=0; i<D; ++i)
        theta(i) = params[2+i];
      for(uint32_t i=0; i<D; ++i)
        for(uint32_t j=0; j<D; ++j)
          Delta(i,j) = params[2+D+i+D*j];
      cout <<"nu="<<nu<<endl;
      cout <<"kappa="<<kappa<<endl;
      cout <<"theta="<<theta<<endl;
      cout <<"Delta="<<Delta<<endl;
    }
    NIWd niw(Delta, theta, nu, kappa, &rndGen);
    dpmm = new DpStickMM<NIWd>(alpha(0), niw);

  }else if(!base.compare("DpNiw")){
    MatrixXd Delta(D,D);
    VectorXd theta(D);
    double nu = 10.0;
    double kappa = 10.0;
//    Delta << nu,0.0,0.0,nu;
//    theta << 0.0,0.0;
    if(vm.count("params"))
    {
      vector<double> params = vm["params"].as< vector<double> >();
      cout<<"params length="<<params.size()<<" D="<<D<<endl;
      nu = params[0];
      kappa = params[1];
      for(uint32_t i=0; i<D; ++i)
        theta(i) = params[2+i];
      for(uint32_t i=0; i<D; ++i)
        for(uint32_t j=0; j<D; ++j)
          Delta(i,j) = params[2+D+i+D*j];
      cout <<"nu="<<nu<<endl;
      cout <<"kappa="<<kappa<<endl;
      cout <<"theta="<<theta<<endl;
      cout <<"Delta="<<Delta<<endl;
    }
    NIWd niw(Delta, theta, nu, kappa, &rndGen);

    shared_ptr<NiwSampledd> niwSampl(new NiwSampledd(niw));
    shared_ptr<LrCluster<NiwSampledd,double> > lrTheta(new 
        LrCluster<NiwSampledd,double>(niwSampl,1.0,&rndGen));
    dpmm = new DpSubclusterMM<NiwSampledd,double>(alpha(0), lrTheta, K, &rndGen);

  }else if(!base.compare("NiwSphere")){
    cout<<"D="<<D<<endl;
    MatrixXd Delta(D-1,D-1);
    double nu = 10.0;
    //    Delta << nu,0.0,0.0,nu;
    //    theta << 0.0,0.0;
    if(vm.count("params"))
    {
      vector<double> params = vm["params"].as< vector<double> >();
      cout<<"params length="<<params.size()<<" D="<<D<<endl;
      nu = params[0];
#pragma omp parallel for
      for(uint32_t i=0; i<D-1; ++i)
        for(uint32_t j=0; j<D-1; ++j)
          Delta(i,j) = params[1+i+(D-1)*j];
      if (D<30){
        cout <<"nu="<<nu<<endl;
        cout <<"Delta="<<Delta<<endl;
      }
    }else if(vm.count("brief"))
    {
      vector<double> params = vm["brief"].as< vector<double> >();
      nu = params[0];
      Delta = params[1]*MatrixXd::Identity(D-1,D-1);
      cout <<"nu="<<nu<<endl;
      cout <<"Delta="<<params[1]<<"I_"<<(D-1)<<endl;
      if (D<30){
        cout <<"nu="<<nu<<endl;
        cout <<"Delta="<<Delta<<endl;
      }
    }
    IWd iw(Delta, nu, &rndGen);
    shared_ptr<NiwSphered> niwSp (new NiwSphered(iw, &rndGen));
//    alphaV[K-1] *=0.1;
    DirCatd dir(alpha,&rndGen); 
    dpmm = new DirMM<double>(dir, niwSp, K);

  }else if(!base.compare("DpNiwSphere")){
    cout<<"D="<<D<<endl;
    MatrixXd Delta(D-1,D-1);
    double nu = 10.0;
    //    Delta << nu,0.0,0.0,nu;
    //    theta << 0.0,0.0;
    if(vm.count("params"))
    {
      vector<double> params = vm["params"].as< vector<double> >();
      cout<<"params length="<<params.size()<<" D="<<D<<endl;
      nu = params[0];
#pragma omp parallel for
      for(uint32_t i=0; i<D-1; ++i)
        for(uint32_t j=0; j<D-1; ++j)
          Delta(i,j) = params[1+i+(D-1)*j];
      if (D<30){
        cout <<"nu="<<nu<<endl;
        cout <<"Delta="<<Delta<<endl;
      }
    }else if(vm.count("brief"))
    {
      vector<double> params = vm["brief"].as< vector<double> >();
      nu = params[0];
      Delta = params[1]*MatrixXd::Identity(D-1,D-1);
      cout <<"nu="<<nu<<endl;
      cout <<"Delta="<<params[1]<<"I_"<<(D-1)<<endl;
      if (D<30){
        cout <<"nu="<<nu<<endl;
        cout <<"Delta="<<Delta<<endl;
      }
    }
    IWd iw(Delta, nu, &rndGen);
    shared_ptr<NiwSphered> theta (new NiwSphered(iw, &rndGen));
    shared_ptr<LrCluster<NiwSphered,double> > lrTheta(new 
      LrCluster<NiwSphered,double>(theta,1.0,&rndGen));
    dpmm = new DpSubclusterMM<NiwSphered,double>(alpha(0), lrTheta, K, &rndGen);

  }else if(!base.compare("DpNiwSphereFull")){
    cout<<"D="<<D<<endl;
    MatrixXd Delta(D-1,D-1);
    double nu = 10.0;
    //    Delta << nu,0.0,0.0,nu;
    //    theta << 0.0,0.0;
    if(vm.count("params"))
    {
      vector<double> params = vm["params"].as< vector<double> >();
      cout<<"params length="<<params.size()<<" D="<<D<<endl;
      nu = params[0];
#pragma omp parallel for
      for(uint32_t i=0; i<D-1; ++i)
        for(uint32_t j=0; j<D-1; ++j)
          Delta(i,j) = params[1+i+(D-1)*j];
      if (D<30){
        cout <<"nu="<<nu<<endl;
        cout <<"Delta="<<Delta<<endl;
      }
    }else if(vm.count("brief"))
    {
      vector<double> params = vm["brief"].as< vector<double> >();
      nu = params[0];
      Delta = params[1]*MatrixXd::Identity(D-1,D-1);
      cout <<"nu="<<nu<<endl;
      cout <<"Delta="<<params[1]<<"I_"<<(D-1)<<endl;
      if (D<30){
        cout <<"nu="<<nu<<endl;
        cout <<"Delta="<<Delta<<endl;
      }
    }
    IWd iw(Delta, nu, &rndGen);
    shared_ptr<NiwSphereFulld> theta (new NiwSphereFulld(iw,&rndGen));
    shared_ptr<LrCluster<NiwSphereFulld,double> > lrTheta(new 
      LrCluster<NiwSphereFulld,double>(theta,1.0,&rndGen));
    dpmm = new DpSubclusterMM<NiwSphereFulld,double>(alpha(0), lrTheta, 
        K, &rndGen);

  }else if(!base.compare("DirNiwSphereFull")){
    cout<<"D="<<D<<endl;
    MatrixXd Delta(D-1,D-1);
    double nu = 10.0;
    //    Delta << nu,0.0,0.0,nu;
    //    theta << 0.0,0.0;
    if(vm.count("params"))
    {
      vector<double> params = vm["params"].as< vector<double> >();
      cout<<"params length="<<params.size()<<" D="<<D<<endl;
      nu = params[0];
#pragma omp parallel for
      for(uint32_t i=0; i<D-1; ++i)
        for(uint32_t j=0; j<D-1; ++j)
          Delta(i,j) = params[1+i+(D-1)*j];
      if (D<30){
        cout <<"nu="<<nu<<endl;
        cout <<"Delta="<<Delta<<endl;
      }
    }else if(vm.count("brief"))
    {
      vector<double> params = vm["brief"].as< vector<double> >();
      nu = params[0];
      Delta = params[1]*MatrixXd::Identity(D-1,D-1);
      cout <<"nu="<<nu<<endl;
      cout <<"Delta="<<params[1]<<"I_"<<(D-1)<<endl;
      if (D<30){
        cout <<"nu="<<nu<<endl;
        cout <<"Delta="<<Delta<<endl;
      }
    }
    DirCatd dir(alpha,&rndGen); 
    IWd iw(Delta, nu, &rndGen);
    shared_ptr<NiwSphereFulld> theta (new NiwSphereFulld(iw,&rndGen));
    dpmm = new DirMM<double>(dir, theta, K);

//  }else if(!base.compare("DpNiwTangent")){
//    cout<<"D="<<D<<endl;
//    MatrixXd Delta(D-1,D-1);
//    double nu = 10.0;
//    //    Delta << nu,0.0,0.0,nu;
//    //    theta << 0.0,0.0;
//    if(vm.count("params"))
//    {
//      vector<double> params = vm["params"].as< vector<double> >();
//      cout<<"params length="<<params.size()<<" D="<<D<<endl;
//      nu = params[0];
//#pragma omp parallel for
//      for(uint32_t i=0; i<D-1; ++i)
//        for(uint32_t j=0; j<D-1; ++j)
//          Delta(i,j) = params[1+i+(D-1)*j];
//      if (D<30){
//        cout <<"nu="<<nu<<endl;
//        cout <<"Delta="<<Delta<<endl;
//      }
//    }else if(vm.count("brief"))
//    {
//      vector<double> params = vm["brief"].as< vector<double> >();
//      nu = params[0];
//      Delta = params[1]*MatrixXd::Identity(D-1,D-1);
//      cout <<"nu="<<nu<<endl;
//      cout <<"Delta="<<params[1]<<"I_"<<(D-1)<<endl;
//      if (D<30){
//        cout <<"nu="<<nu<<endl;
//        cout <<"Delta="<<Delta<<endl;
//      }
//    }
//    VectorXd mu(D-1); mu.setZero();
//    double kappa = .01;
//    cout<<"theta = "<<mu.transpose()<<endl;
//    cout<<"kappa = "<<kappa<<endl;
//    NIWd niw(Delta, mu, nu, kappa, &rndGen);
//    shared_ptr<NiwTangentd> theta (new NiwTangentd(niw,&rndGen));
//    shared_ptr<LrCluster<NiwTangentd,double> > lrTheta(new 
//      LrCluster<NiwTangentd,double>(theta,1.0,&rndGen));
//    dpmm = new DpSubclusterMM<NiwTangentd,double>(alpha(0), lrTheta, 
//        K, &rndGen);
//
  }else if(!base.compare("NiwSphereUnifNoise")){

    MatrixXd Delta(D-1,D-1);
    double nu = 10.0;
    //    Delta << nu,0.0,0.0,nu;
    //    theta << 0.0,0.0;
    if(vm.count("params"))
    {
      vector<double> params = vm["params"].as< vector<double> >();
      cout<<"params length="<<params.size()<<" D="<<D<<endl;
      nu = params[0];
#pragma omp parallel for
      for(uint32_t i=0; i<D-1; ++i)
        for(uint32_t j=0; j<D-1; ++j)
          Delta(i,j) = params[1+i+(D-1)*j];
      if (D<30){
        cout <<"nu="<<nu<<endl;
        cout <<"Delta="<<Delta<<endl;
      }
    }else if(vm.count("brief"))
    {
      vector<double> params = vm["brief"].as< vector<double> >();
      nu = params[0];
      Delta = params[1]*MatrixXd::Identity(D-1,D-1);
      cout <<"nu="<<nu<<endl;
      cout <<"Delta="<<params[1]<<" I_"<<(D-1)<<endl;
      if (D<30){
        cout <<"nu="<<nu<<endl;
        cout <<"Delta="<<Delta<<endl;
      }
    }

    IWd iw(Delta, nu, &rndGen);
    vector<shared_ptr<BaseMeasure<double> > > thetas;
    for (uint32_t k=0; k<K-1; ++k)
      thetas.push_back(shared_ptr<BaseMeasure<double> >(
            new NiwSphered(iw, &rndGen)));
    thetas.push_back(boost::shared_ptr<BaseMeasure<double> >(new UnifSphered(D)));

    DirCatd dir(alpha,&rndGen); 
    dpmm = new DirMM<double>(dir, thetas);

  }else if(!base.compare("NiwSphereCuda")){
    cout<<"D="<<D<<endl;
    MatrixXd Delta(D-1,D-1);
    double nu = 10.0;
    //    Delta << nu,0.0,0.0,nu;
    //    theta << 0.0,0.0;
    if(vm.count("params"))
    {
      vector<double> params = vm["params"].as< vector<double> >();
      cout<<"params length="<<params.size()<<" D="<<D<<endl;
      nu = params[0];
#pragma omp parallel for
      for(uint32_t i=0; i<D-1; ++i)
        for(uint32_t j=0; j<D-1; ++j)
          Delta(i,j) = params[1+i+(D-1)*j];
      if (D<30){
        cout <<"nu="<<nu<<endl;
        cout <<"Delta="<<Delta<<endl;
      }
    }else if(vm.count("brief"))
    {
      vector<double> params = vm["brief"].as< vector<double> >();
      nu = params[0];
      Delta = params[1]*MatrixXd::Identity(D-1,D-1);
      cout <<"nu="<<nu<<endl;
      cout <<"Delta="<<params[1]<<"I_"<<(D-1)<<endl;
      if (D<30){
        cout <<"nu="<<nu<<endl;
        cout <<"Delta="<<Delta<<endl;
      }
    }
    IWd iw(Delta, nu, &rndGen);
    shared_ptr<NiwSphered> niwSp (new NiwSphered(iw, &rndGen));
//    alphaV[K-1] *=0.1;
    DirCatd dir(alpha,&rndGen); 
    dpmmf = new DirMMcld<NiwSphered,double>(dir, niwSp);
//  }else if(!base.compare("DPvMFmeans")){
//    double lambda = cos(5.0*M_PI/180.0);
//    if(vm.count("params"))
//    {
//      vector<double> params = vm["params"].as< vector<double> >();
//      cout<<"params length="<<params.size()<<endl;
//      lambda = params[0];
//    }
//    spkm = new DPvMFMeans<double>(spx, K, lambda, &rndGen);
  }else if(!base.compare("CrpvMF")){
    cout<<"D="<<D<<endl;

    double a0 = 2.0;
    double b0 = 1.7;
    double t0 = 0.01;
    VectorXd m0(D);

    if(vm.count("params"))
    {
      vector<double> params = vm["params"].as< vector<double> >();
      cout<<"params length="<<params.size()<<" D="<<D<<endl;
      for(uint32_t i=0; i<D; ++i)
        m0(i) = params[i];
      t0 = params[D];
      a0 = params[D+1];
      b0 = params[D+2];
    }
    vMFpriorFull<double> vMFprior(m0,t0,a0,b0,&rndGen);
    shared_ptr<vMFbase<double> > vMFsampled( new vMFbase<double>(vMFprior));
    dpmm = new CrpMM<double>(alpha(0),vMFsampled,K,&rndGen);

  }else if(!base.compare("spkm")){
    spkm = new SphericalKMeans<double>(spx, K, &rndGen);
  }else if(!base.compare("spkmKarcher")){
    spkm = new SphericalKMeansKarcher<double>(spx, K, &rndGen);
  }else if(!base.compare("kmeans")){
    spkm = new KMeans<double>(spx, K, &rndGen);
  }else{
    cout<<"base "<<base<<" not supported"<<endl;
    return 1;
  }

  string pathOut ="./labels.csv";
  if(vm.count("output")) 
    pathOut = vm["output"].as<string>();
  cout<<"output to "<<pathOut<<endl;

  // TODO stupid duplication
  if(!base.compare("NiwSphereCuda"))
  {
    assert(dpmmf!=NULL);
//    MatrixXf x_f = x.cast<float>();
    shared_ptr<ClSphereGpud> cld(new ClSphereGpud(spx,
          spVectorXu(new VectorXu(N)),K));
    dpmmf->initialize(cld);

    ofstream fout(pathOut.data(),ofstream::out);
    ofstream foutJointLike((pathOut+"_jointLikelihood.csv").data(),ofstream::out);
    for (uint32_t t=0; t<T; ++t)
    {
      cout<<"------------ t="<<t<<" -------------"<<endl;
      dpmmf->sampleLabels();

      const VectorXu& z = dpmmf->getLabels().transpose();
      cout<<"   K="<<dpmm->getK();
      cout<<"  counts=   "<<counts<double,uint32_t>(z,dpmmf->getK()).transpose()<<endl;
//      cout<<"  logJoint= "<<dpmmf->logJoint()<<endl;
      for (uint32_t i=0; i<z.size()-1; ++i) 
        fout<<z(i)<<" ";
      fout<<z(z.size()-1)<<endl;
      foutJointLike<<dpmmf->logJoint()<<endl;

      dpmmf->sampleParameters();
    }
    fout.close();
    foutJointLike.close();

    MatrixXd logLikes;
    MatrixXu inds = dpmmf->mostLikelyInds(10,logLikes);
    cout<<"most likely indices"<<endl;
    cout<<inds<<endl;
    cout<<"----------------------------------------"<<endl;

    fout.open((pathOut+"mlInds.csv").data(),ofstream::out);
    fout<<inds<<endl;
    fout.close();
    fout.open((pathOut+"mlLogLikes.csv").data(),ofstream::out);
    fout<<logLikes<<endl;
    fout.close();

  }else if(!base.compare("spkm") || !base.compare("spkmKarcher") 
      || !base.compare("kmeans"))
  {
    ofstream fout(pathOut.data(),ofstream::out);
    ofstream foutJointLike((pathOut+"_jointLikelihood.csv").data(),ofstream::out);
    Timer watch;
    for (uint32_t t=0; t<T; ++t)
    {
      cout<<"------------ t="<<t<<" -------------"<<endl;
      watch.tic();
      spkm->updateCenters();
      watch.toctic("-- updateCenters");

      const VectorXu& z = spkm->z();
      for (uint32_t i=0; i<z.size()-1; ++i) 
        fout<<z(i)<<" ";
      fout<<z(z.size()-1)<<endl;
      double deviation = spkm->avgIntraClusterDeviation();

      cout<<"   K="<<spkm->getK();
      cout<<"  counts=   "<<counts<double,uint32_t>(z,spkm->getK()).transpose();
      cout<<" avg deviation "<<deviation<<endl;
      foutJointLike<<deviation<<endl;

      watch.tic();
      spkm->updateLabels();
      watch.toctic("-- updateLabels");
    }
    fout.close();

    MatrixXd deviates;
    MatrixXu inds = spkm->mostLikelyInds(10,deviates);
    cout<<"most likely indices"<<endl;
    cout<<inds<<endl;
    cout<<"----------------------------------------"<<endl;

    fout.open((pathOut+"mlInds.csv").data(),ofstream::out);
    fout<<inds<<endl;
    fout.close();
    fout.open((pathOut+"mlLogLikes.csv").data(),ofstream::out);
    fout<<deviates<<endl;
    fout.close();

    ofstream foutMeans((pathOut+"_means.csv").data(),ofstream::out);
    foutMeans << spkm->centroids()<<endl;
    foutMeans.close();

  }else{
    assert(dpmm!=NULL);
    dpmm->initialize(x);

    ofstream fout(pathOut.data(),ofstream::out);
    ofstream foutJointLike((pathOut+"_jointLikelihood.csv").data(),ofstream::out);
    ofstream foutMeans((pathOut+"_means.csv").data(),ofstream::out);
    ofstream foutCovs((pathOut+"_covs.csv").data(),ofstream::out);

      const VectorXu& z = dpmm->getLabels().transpose();
      for (uint32_t i=0; i<z.size()-1; ++i) 
        fout<<z(i)<<" ";
      fout<<z(z.size()-1)<<endl;
      foutJointLike<<dpmm->logJoint()<<endl;
      dpmm->dump(foutMeans,foutCovs);

    for (uint32_t t=0; t<T; ++t)
    {
      cout<<"------------ t="<<t<<" -------------"<<endl;
      // sample clusters from the base measure to fill up to K clusters 
      // only if we are not proposing splits and merges to make sure that the
      // number of clusters stays the originally intended one
//      if(!proposeSplitMerge)  dpmm->resampleFromBase(K);

//      VectorXd Ns = counts(dpmm->getLabels(),dpmm->getK()*2).transpose();
//      cout<<"-- counts= "<<Ns.transpose()<<" sum="<<Ns.sum()<<endl;
      dpmm->sampleParameters();


      const VectorXu& z = dpmm->getLabels().transpose();
      for (uint32_t i=0; i<z.size()-1; ++i) 
        fout<<z(i)<<" ";
      fout<<z(z.size()-1)<<endl;
      foutJointLike<<dpmm->logJoint()<<endl;
      dpmm->dump(foutMeans,foutCovs);
      
      dpmm->sampleLabels();

      VectorXd Ns = dpmm->getCounts();
      cout<<"--  counts= "<<Ns.transpose()<<" sum="<<Ns.sum()<<endl;
      cout<<"    K="<<dpmm->getK();
      cout<<"    logJoint= "<<dpmm->logJoint()<<endl;

      if(proposeSplitMerge)
      {
        dpmm->proposeMerges();
        //      Ns = counts(dpmm->getLabels(),dpmm->getK()*2).transpose();
        //      cout<<"-- counts= "<<Ns.transpose()<<" sum="<<Ns.sum()<<endl;

        dpmm->proposeSplits();
        //      Ns = counts(dpmm->getLabels(),dpmm->getK()*2).transpose();
        //      cout<<"-- counts= "<<Ns.transpose()<<" sum="<<Ns.sum()<<endl;
      }
    }
    fout.close();
    foutJointLike.close();

    MatrixXd logLikes;
    MatrixXu inds = dpmm->mostLikelyInds(10,logLikes);
    cout<<"most likely indices"<<endl;
    cout<<inds<<endl;
    cout<<"----------------------------------------"<<endl;

    fout.open((pathOut+"mlInds.csv").data(),ofstream::out);
    fout<<inds<<endl;
    fout.close();
    fout.open((pathOut+"mlLogLikes.csv").data(),ofstream::out);
    fout<<logLikes<<endl;
    fout.close();
  }
};


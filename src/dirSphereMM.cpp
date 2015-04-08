/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */

#include <iostream>
#include <fstream>
#include <string>
#include <boost/program_options.hpp>

#include <omp.h>

#include <dpMM/dirMM.hpp>
#include <dpMM/niwSphere.hpp>

using namespace Eigen;
using std::string;
namespace po = boost::program_options;

{

  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "produce help message")
    ("N,N", po::value<int>(), "number of input datapoints")
    ("D,D", po::value<int>(), "number of dimensions of the data")
    ("T,T", po::value<int>(), "iterations")
    ("alpha,a", po::value<double>(), 
      "alpha parameter of the Dir (assumed symmetric)")
    ("base,b", po::value<string>(), 
      "which base measure to use (only NIW right now)")
    ("params,p", po::value< vector<double> >()->multitoken(), 
      "parameters of the base measure")
    ("input,i", po::value<string>(), 
      "path to input dataset .csv file (rows: dimensions; cols: different datapoints)")
    ("output,o", po::value<string>(), 
      "path to output labels .csv file (rows: time; cols: different datapoints)")
    ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);    

  if (vm.count("help")) {
    cout << desc << "\n";
    return 1;
  }
  
#ifndef NDEBUG
  uint32_t nThreads = 1;
#else
  uint32_t nThreads = omp_get_max_threads(); // has to match number of threads on computer
#endif
  // number of iterations
  uint32_t T=1000;
  if (vm.count("T")) T = vm["T"].as<int>();
  uint32_t N=100;
  if (vm.count("N")) N = vm["N"].as<int>();
  uint32_t D=2;
  if (vm.count("D")) D = vm["D"].as<int>();
  cout << "T="<<T<<endl;
  // DP alpha parameter
  double alpha = 0.1;
  if (vm.count("alpha")) alpha = vm["alpha"].as<double>();
  cout << "alpha="<<alpha<<endl;
  // which base distribution
  string base = "NIW";
  if(vm.count("base")) base = vm["base"].as<string>();
  
//  DpMM *dpmm;
//  if(!base.compare("NIW"))
//  {
    MatrixXd Delta(D-1,D-1);
    VectorXd theta(D);
    double nu = D+3.0;
    double kappa = D+3.0;
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
      for(uint32_t i=0; i<D-1; ++i)
        for(uint32_t j=0; j<D-1; ++j)
          Delta(i,j) = params[2+D-1+i+(D-1)*j];
      cout <<"nu="<<nu<<endl;
      cout <<"kappa="<<kappa<<endl;
      cout <<"theta="<<theta<<endl;
      cout <<"Delta="<<Delta<<endl;
    }
    niwSphere_sampled niw(D-1,kappa,nu, theta.data(),Delta.data());
//    
//  }else{
//    cout<<"base "<<base<<" not supported"<<endl;
//    return 1;
//  }
  
  MatrixXd x(D,N);
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
    std::ifstream fin(pathIn.data(),std::ifstream::in);
    for (uint32_t j=0; j<D; ++j)
      for (uint32_t i=0; i<N; ++i)
      {
        fin>>x(j,i);
      }
    //cout<<x<<endl;
  }
  string pathOut ="./labels.csv";
  if(vm.count("output")) 
    pathOut = vm["output"].as<string>();
  cout<<"output to "<<pathOut<<endl;

  VectorXd z(N);
  z.setZero();
  cout<<"z.shape="<<z.size()<<endl;

  DpSubclustersSphereMM *dpmm = new DpSubclustersSphereMM(N,D,x.data(), 
      z.data(), alpha, niw, nThreads,true,true);
  cout<<"-- init"<<endl;
  dpmm->initialize();

  std::ofstream fout(pathOut.data(),std::ofstream::out);
  for (uint32_t t=0; t<T; ++t)
  {
    cout<<"------------ t="<<t<<" -------------"<<endl;
    cout<<"-- sampling params"<<endl;
    dpmm->sample_params();
    cout<<"-- sampling superclusters"<<endl;
    dpmm->sample_superclusters();
    cout<<"-- sampling labels"<<endl;
    dpmm->sample_labels();

    if(t%50 == 0)
    {
      cout<<"-- random splits"<<endl;
      dpmm->propose_random_splits();
    }
    cout<<"-- random merges"<<endl;
    dpmm->propose_random_merges();
    cout<<"-- splits"<<endl;
    dpmm->propose_splits();

    cout<<"   logLike=\t"<<dpmm->joint_loglikelihood()<<endl;
    cout<<"   K=\t"<<dpmm->getK()<<endl;
    cout<<"   Nk=\t"<<dpmm->getNK()<<endl;


//    const VectorXi& z = dpmm->getLabels().transpose();
//    //cout<< "z_"<<t<<"= "<<z.transpose()<<endl;
    for (uint32_t i=0; i<z.size()-1; ++i) 
      fout<<int(floor(z(i)))<<" ";
    fout<<int(floor(z(z.size()-1)))<<endl;
  }
  fout.close();
}

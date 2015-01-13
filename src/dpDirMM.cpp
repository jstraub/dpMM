
#include <iostream>
#include <fstream>
#include <string>
#include <boost/program_options.hpp>

#include "dpSubclusterMM.hpp"
#include "dirBaseMeasure.hpp"

using namespace Eigen;
using std::string; 
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
    ("T,T", po::value<int>(), "iterations")
    ("alpha,a", po::value< vector<double> >()->multitoken(), 
      "alpha parameter of the DP (if single value assumes all alpha_i are the "
      "same")
    ("K,K", po::value<int>(), "number of initial clusters ")
    ("base", po::value<string>(), 
      "which base measure to use (only DpDir, ")
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
  // DP alpha parameter
  VectorXd alpha(K);
  alpha.setOnes(K);
  if (vm.count("alpha"))
  {
    vector<double> params = vm["alpha"].as< vector<double> >();
    if(params.size()==1)
      alpha *= params[0];
    else
      for (uint32_t k=0; k<K; ++k)
        alpha(k) = params[k];
  }
  cout << "alpha="<<alpha.transpose()<<endl;

  shared_ptr<MatrixXd> spx(new MatrixXd(D,N));
  MatrixXd& x(*spx);
  string pathIn ="";
  if(vm.count("input")) pathIn = vm["input"].as<string>();
  if (!pathIn.compare(""))
  {
    for(uint32_t i=0; i<N; ++i)
    {
      if(i<N/3)
      {
        x.col(i) << VectorXd::Zero(D/3), VectorXd::Zero(D/3), VectorXd::Ones(D/3)*100;
      }else if(N/3 <= i && i < 2*N/3){
        x.col(i) << VectorXd::Zero(D/3), VectorXd::Ones(D/3)*100, VectorXd::Zero(D/3);
      }else{
        x.col(i) << VectorXd::Ones(D/3)*100, VectorXd::Zero(D/3), VectorXd::Zero(D/3);
      }
//      x.col(i) /= x.col(i).sum();
    }
    cout<<x<<endl;
  }else{
    cout<<"loading data from "<<pathIn<<endl;
    std::ifstream fin(pathIn.data(),std::ifstream::in);
    for (uint32_t j=0; j<D; ++j)
      for (uint32_t i=0; i<N; ++i)
        fin>>x(j,i);
  }

  // which base distribution
  string base = "DpDir";
  if(vm.count("base")) base = vm["base"].as<string>();

//  if(base.compare("DpDir")){
//    // normalize to unit sum
//    int err = 0;
//#pragma omp parallel for
//    for (uint32_t i=0; i<N; ++i)
//      if(fabs(x.col(i).sum() - 1.0) > 1e-1)
//      {
//        err++;
//        cout<<x.col(i).sum() <<endl;
//      }else
//        x.col(i) /= x.col(i).sum();
//    if(err>0) return 0;
//  }

  
  DpMM<double> *dpmm=NULL;
  if(!base.compare("DpDir"))
  {
  // Dir alpha parameter
    VectorXd gamma(D);
    gamma.setOnes(D);
    if(vm.count("params"))
    {
      vector<double> params = vm["params"].as< vector<double> >();
      cout<<"params length="<<params.size()<<" D="<<D<<endl;
      if(params.size() == 1)
        gamma *= params[0];
      else
        for(uint32_t i=0; i<D; ++i)
          gamma(i) = params[i];
      cout <<"gamma="<<gamma<<endl;
    }
    DirMultd dir(gamma, &rndGen);
    shared_ptr<DirMultSampledd> dirSampl(new DirMultSampledd(dir));
    shared_ptr<LrCluster<DirMultSampledd,double> > lrTheta(new 
        LrCluster<DirMultSampledd,double>(dirSampl,1.0,&rndGen));
    dpmm = new DpSubclusterMM<DirMultSampledd,double>(alpha(0), 
        lrTheta, K, &rndGen);
  }  
  
  
  string pathOut ="./labels.csv";
  if(vm.count("output")) 
    pathOut = vm["output"].as<string>();
  cout<<"output to "<<pathOut<<endl;

  assert(dpmm!=NULL);
  dpmm->initialize(x);

  std::ofstream fout(pathOut.data(),std::ofstream::out);
  std::ofstream foutJointLike((pathOut+"_jointLikelihood.csv").data(),std::ofstream::out);
  std::ofstream foutMeans((pathOut+"_means.csv").data(),std::ofstream::out);
  std::ofstream foutCovs((pathOut+"_covs.csv").data(),std::ofstream::out);

  const VectorXu& z = dpmm->getLabels().transpose();
  for (uint32_t i=0; i<z.size()-1; ++i) 
    fout<<z(i)<<" ";
  fout<<z(z.size()-1)<<endl;
  foutJointLike<<dpmm->logJoint()<<endl;
  dpmm->dump(foutMeans,foutCovs);

  for (uint32_t t=0; t<T; ++t)
  {
    cout<<"------------ t="<<t<<" -------------"<<endl;
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

    dpmm->proposeMerges();
    //      Ns = counts(dpmm->getLabels(),dpmm->getK()*2).transpose();
    //      cout<<"-- counts= "<<Ns.transpose()<<" sum="<<Ns.sum()<<endl;

    dpmm->proposeSplits();
    //      Ns = counts(dpmm->getLabels(),dpmm->getK()*2).transpose();
    //      cout<<"-- counts= "<<Ns.transpose()<<" sum="<<Ns.sum()<<endl;

  }
  fout.close();
  foutJointLike.close();

  MatrixXd logLikes;
  MatrixXu inds = dpmm->mostLikelyInds(10,logLikes);
  cout<<"most likely indices"<<endl;
  cout<<inds<<endl;
  cout<<"----------------------------------------"<<endl;

  fout.open((pathOut+"mlInds.csv").data(),std::ofstream::out);
  fout<<inds<<endl;
  fout.close();
  fout.open((pathOut+"mlLogLikes.csv").data(),std::ofstream::out);
  fout<<logLikes<<endl;
  fout.close();
};



#include <Eigen/Dense>
#include <iostream>
#include <fstream>

#include <boost/program_options.hpp>

#include "normalSphere.hpp"

using namespace Eigen;
using namespace std;

namespace po = boost::program_options;

int main(int argc, char** argv)
{
  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "produce help message")
    ("seed", po::value<int>(), "seed for random number generator")
    ("N,N", po::value<int>(), "number of input datapoints")
    ("D,D", po::value<int>(), "number of dimensions of the data")
    ("K,K", po::value<int>(), "number of initial clusters ")
    ("nu,nu", po::value<double>(), "nu parameter of IW from which "
      "sigmas are sampled")
    ("minAngle,a", po::value<double>(), "min angle between means on sphere")
    ("delta,d", po::value<double>(), "delta of NIW")
    ("output,o", po::value<string>(), 
      "path to output labels and data .csv file (rows: time; cols: different "
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
  uint32_t N=100;
  if (vm.count("N")) N = vm["N"].as<int>();
  uint32_t D=3;
  if (vm.count("D")) D = vm["D"].as<int>();
  string pathOut ="./rndSphereData";
  if(vm.count("output")) 
    pathOut = vm["output"].as<string>();
  cout<<"output to "<<pathOut<<endl;
  
  double nu = static_cast<double>(D)+2.; // 100 for spherical
  if (vm.count("nu")) nu = vm["nu"].as<double>();
  if(nu < static_cast<double>(D)+1.+1e-8)
    nu = static_cast<double>(D)+1.+1e-8;
  double minAngle = 6.;
  if (vm.count("minAngle")) nu = vm["minAngle"].as<double>(); 
  double delta = 6.;
  if (vm.count("delta")) nu = vm["delta"].as<double>(); 

  MatrixXd Delta(D-1,D-1);
  Delta = MatrixXd::Identity(D-1,D-1);
  Delta *= nu*(delta*PI/180.)*(delta*PI/180.);

//  double nu = D+0.1
//  MatrixXd Delta(D,D);
//  Delta.setIdentity();
//  Delta *= nu * (12.*PI)/180.0;
  MatrixXd x(D,N);
  VectorXu z(N);
  sampleClustersOnSphere<double>(Delta,nu,x,z,K,minAngle);
  for(uint32_t i=0; i<N-1; ++i)
    if(fabs(x.col(i).norm()-1.0) > 1e-2)
    {
      cout<<"@"<<i<<":"<<x.col(i).norm()<<endl;
      cout<<" error in generating data"<<endl;
      return 0;
    }

  ofstream fout;
  fout.open((pathOut+".csv").data(),ofstream::out);
  for(uint32_t d=0; d<D; ++d)
  {
    for(uint32_t i=0; i<N-1; ++i)
      fout<<x(d,i)<<" ";
    fout<<x(d,N-1)<<endl;
  }
  fout.close();
  fout.open((pathOut+"_gt.lbl").data(),ofstream::out);
    for(uint32_t i=0; i<N-1; ++i)
      fout<<z(i)<<" ";
    fout<<z(N-1)<<endl;
  fout.close();

}

#include <iostream>

#include <boost/program_options.hpp>

#include "dirNaiveBayes.hpp"
#include "niwBaseMeasure.hpp"
#include "typedef.h"

namespace po = boost::program_options;

int main(int argc, char **argv){

	
  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "produce help message")
	("K,K", po::value<int>(), "number of initial clusters ")
	("T,T", po::value<int>(), "iterations")
	("N,N", po::value<int>(), "number of input datapoints")
	("M,M", po::value<int>(), "number of documents")
    ("D,D", po::value<int>(), "number of dimensions of the data")
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

	uint32_t K=2;
	if (vm.count("K")) K = vm["K"].as<int>();
	// number of iterations
	uint32_t T=100;
	if (vm.count("T")) T = vm["T"].as<int>();
	uint32_t N=100;
	if (vm.count("N")) N = vm["N"].as<int>();
	uint32_t D=2;
	if (vm.count("D")) D = vm["D"].as<int>();
	vector<uint> Mword; 
	uint32_t M=2;
	if (vm.count("M")) M = vm["M"].as<int>();

	
	vector< Matrix<double, Dynamic, Dynamic> > x;
	x.reserve(N);
	string pathIn ="";
	if(vm.count("input")) pathIn = vm["input"].as<string>();
	if (!pathIn.compare(""))
	{
		uint Ndoc=M;
		uint Nword=int(N/M); 
		for(uint i=0; i<Ndoc; ++i) {
			MatrixXd  xdoc(3,Nword);  
			for(uint w=0; w<Nword; ++w) {
				if(i<Ndoc/2)
					xdoc.col(w) <<  VectorXd::Zero(D);
				else
					xdoc.col(w) <<  2.0*VectorXd::Ones(D);
			}

			x.push_back(xdoc); 
		}
	}else{
		
		MatrixXd data(D,N);
		VectorXu words(M);

		cout<<"loading data from "<<pathIn<<endl;
		ifstream fin(pathIn.data(),ifstream::in);
		for (uint32_t j=0; j<M; ++j)
			fin >> words(j,1); 

		for (uint32_t j=1; j<(D+1); ++j) 
		{
			for (uint32_t i=0; i<N; ++i) 
			{
				fin>>data(j-1,i);
			}
		}
		
		uint count = 0;
		for (uint32_t j=0; j<M; ++j)
		{
			x.push_back(data.middleCols(count,words[j]));
			count+=words[j];
		}
	}


	
	double nu = D+1;
	double kappa = D+1;
	MatrixXd Delta = 0.1*MatrixXd::Identity(D,D);
	Delta *= nu;
	VectorXd theta = VectorXd::Zero(D);
	VectorXd alpha = 10.0*VectorXd::Ones(K);

	boost::mt19937 rndGen(9191);
	NIW<double> niw(Delta,theta,nu,kappa,&rndGen);

	boost::shared_ptr<NiwMarginalized<double> > niwMargBase(
		new NiwMarginalized<double>(niw));


	Dir<Catd,double> dir(alpha,&rndGen); 

  
	//cout<<"------ marginalized ---- NIW "<<endl;
	//DirNaiveBayes<double> naive_marg(dir,niwMargBase);
	//naive_marg.initialize(x);
	//cout<<naive_marg.labels().transpose()<<endl;
	//for(uint32_t t=0; t<30; ++t)
	//{
	//naive_marg.sampleLabels();
	//naive_marg.sampleParameters();
	//cout<<naive_marg.labels().transpose()
		//<<" logJoint="<<naive_marg.logJoint()<<endl;
	//}

	boost::shared_ptr<NiwSampled<double> > niwSampled( new NiwSampled<double>(niw));
	DirNaiveBayes<double> naive_samp(dir,niwSampled);
  
	naive_samp.initialize( (const vector< Matrix<double, Dynamic, Dynamic> >) x );
	naive_samp.inferAll(30,true);

	return(0); 
	
};

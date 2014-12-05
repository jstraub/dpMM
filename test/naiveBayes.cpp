#include <iostream>

#include <boost/program_options.hpp>

#include "dirNaiveBayes.hpp"
#include "niwBaseMeasure.hpp"
#include "typedef.h"
#include "timer.hpp"

namespace po = boost::program_options;

int main(int argc, char **argv){

	
  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "produce help message")
	("K,K", po::value<int>(), "number of initial clusters ")
	("T,T", po::value<int>(), "iterations")
	("v,v", po::value<bool>(), "verbose output")
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

	uint K=2;
	uint T=100;
	uint N=100;
	uint D=2;
	uint M=2;
	uint NumObs = 1; 
	bool verbose = false; 
	vector<uint> Mword; 
	if (vm.count("K")) 
		K = vm["K"].as<int>();
	if (vm.count("T")) 
		T = vm["T"].as<int>();
	if (vm.count("v"))
		verbose = vm["v"].as<bool>();

	string pathIn ="";
	string pathOut ="";
	if(vm.count("input")) 
		pathIn = vm["input"].as<string>();
	if(vm.count("output")) 
		pathOut= vm["output"].as<string>();
	
	vector< Matrix<double, Dynamic, Dynamic> > x;
	x.reserve(N);
	if (!pathIn.compare(""))
	{
		cout<<"making some data up " <<endl;
		uint Ndoc=M;
		uint Nword=int(N/M); 
		for(uint i=0; i<Ndoc; ++i) {
			MatrixXd  xdoc(D,Nword);  
			for(uint w=0; w<Nword; ++w) {
				if(i<Ndoc/2)
					xdoc.col(w) <<  VectorXd::Zero(D);
				else
					xdoc.col(w) <<  2.0*VectorXd::Ones(D);
			}

			x.push_back(xdoc); 
		}
	}else{

		cout<<"loading data from "<<pathIn<<endl;
		ifstream fin(pathIn.data(),ifstream::in);
		
		//read parameters from file (tired of passing them in)
		fin>>NumObs; 
		fin>>N;
		fin>>M;
		fin>>D; 

		MatrixXd data(D,N);
		VectorXu words(M);

		for (uint j=0; j<M; ++j) 
			fin>>words(j); 
		
		for (uint j=1; j<(D+1); ++j) 
			for (uint i=0; i<N; ++i) 
				fin>>data(j-1,i);
		
		uint count = 0;
		for (uint j=0; j<M; ++j)
		{
			x.push_back(data.middleCols(count,words[j]));
			count+=words[j];
		}
		fin.close();
	}

	
	double nu = D+1;
	double kappa = D+1;
	MatrixXd Delta = 0.1*MatrixXd::Identity(D,D);
	Delta *= nu;
	VectorXd theta = VectorXd::Zero(D);
	VectorXd alpha = 10.0*VectorXd::Ones(K);

	boost::mt19937 rndGen(9191);
	NIW<double> niw(Delta,theta,nu,kappa,&rndGen);

	Dir<Catd,double> dir(alpha,&rndGen); 

  
	//cout<<"------ marginalized ---- NIW "<<endl;
	//DirNaiveBayes<double> naive_marg(dir,niwMargBase);
	//naive_marg.initialize(x);
	//cout<<naive_marg.labels().transpose()<<endl;
	//for(uint t=0; t<30; ++t)
	//{
	//naive_marg.sampleLabels();
	//naive_marg.sampleParameters();
	//cout<<naive_marg.labels().transpose()
		//<<" logJoint="<<naive_marg.logJoint()<<endl;
	//}
	Timer tlocal;
	tlocal.tic();

	boost::shared_ptr<NiwSampled<double> > niwSampled( new NiwSampled<double>(niw));
	DirNaiveBayes<double> naive_samp(dir,niwSampled);
  
	cout << "naiveBayesian Clustering:" << endl; 
	cout << "Ndocs=" << M << endl; 
	cout << "Ndata=" << N << endl; 
	cout << "dim=" << D << endl;
	cout << "Num Cluster = " << K << ", (" << T << " iterations)." << endl;

	naive_samp.initialize( (const vector< Matrix<double, Dynamic, Dynamic> >) x );
	naive_samp.inferAll(T,verbose);


	if (pathOut.compare(""))
	{
		ofstream fout(pathOut.data(),ofstream::out);
		
		streambuf *coutbuf = std::cout.rdbuf(); //save old cout buffer
		cout.rdbuf(fout.rdbuf()); //redirect std::cout to fout1 buffer

			naive_samp.dump(fout,fout);

		std::cout.rdbuf(coutbuf); //reset to standard output again

		fout.close();
	}

	tlocal.displayElapsedTimeAuto();
	return(0); 
	
};

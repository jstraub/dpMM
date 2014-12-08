#include <iostream>

#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp> //case insensitve string comparison

#include "dirMultiNaiveBayes.hpp"
#include "niwBaseMeasure.hpp"
#include "niwSphere.hpp"
#include "normalSphere.hpp" //for sampling points on the sphere
#include "iw.hpp"
#include "typedef.h"
#include "timer.hpp"

namespace po = boost::program_options;
using boost::iequals; 

int main(int argc, char **argv){

	
  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "produce help message")
	("K,K", po::value<int>(), "number of initial clusters ")
	("T,T", po::value<int>(), "iterations")
	("v,v", po::value<bool>(), "verbose output")
    ("params,p", po::value<string>(), 
      "path to file containing parameters (see class definition) "
      "datapoints)")
	("nu,n", po::value<std::vector<double> >()->multitoken(), "nu to use for distributions")
	("deltaOffset,d", po::value<std::vector<double> >()->multitoken(), "delta offset to use for distributions")
    ("input,i", po::value<string>(), 
      "path to input dataset .csv file (rows: dimensions; cols: different "
      "datapoints)")
    ("output,o", po::value<string>(), 
      "path to output labels .csv file (rows: time; cols: different "
      "datapoints)")
	("b,b", po::value<std::vector<string> >()->multitoken(), 
	  "base class to use for components (must be the same size as M in the data). " 
	  "valid values: NiwSampled, NiwSphere, [default: NiwSampled].");
    

    po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);    

	if (vm.count("help")) {
		cout << desc << "\n";
		return 1;
	}

	bool verbose = false; 

	uint NumObs = 2; //num observations (number of components of multi-dimention data)
	uint K=2; //num clusters
	uint T=10; //iterations
	uint M=2; //num docs

	vector<uint> N(NumObs, 100) ; //num data points (total)
	vector<uint> D(NumObs, 2);  //dimention of data 
	vector<double> nuIn; //nu
	vector<double> deltaOffsetIn; //delta offset
	
	//use the params provided 
	string paramsF="";
	if (vm.count("params"))
		paramsF = vm["params"].as<string>(); 

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
	if(vm.count("nu"))
		nuIn = vm["nu"].as<vector<double> >();
	if(vm.count("deltaOffset")) 	
		deltaOffsetIn = vm["deltaOffset"].as<vector<double> >();
	
	vector<vector< Matrix<double, Dynamic, Dynamic> > > x;
	x.reserve(NumObs);
	if (!pathIn.compare(""))
	{
		cout<<"making some data up" <<endl;
		for (uint n=0; n<NumObs; ++n) 
		{
			vector<Matrix<double, Dynamic, Dynamic> > temp;
			temp.reserve(M); 
			uint Ndoc=M;
			uint Nword=int(N[n]/M); 
			for(uint i=0; i<Ndoc; ++i) 
			{
				MatrixXd  xdoc(D[n],Nword);  
				//for(uint w=0; w<Nword; ++w) 
				//{
				//	if(i<Ndoc/2)
				//		xdoc.col(w) <<  (NumObs*(n%2))*VectorXd::Ones(D[n]);
				//	else
				//		xdoc.col(w) <<  (NumObs*(n%2)+1)*VectorXd::Ones(D[n]);
				//}

				sampleClustersOnSphere(xdoc,K); 


				temp.push_back(xdoc); 
			}

			x.push_back(temp); 
		}

	}else{
		cout<<"loading data from "<<pathIn<<endl;
		ifstream fin(pathIn.data(),ifstream::in);

		if (!fin.good()) {
			cout << "could not open file...returning" << endl;
			return(-1);
		}

		//read data
		fin>>NumObs; 
		N.clear(); N.reserve(NumObs);
		D.clear(); D.reserve(NumObs);
		
		for(uint32_t n=0; n<NumObs; ++n) 
		{
			vector< Matrix<double, Dynamic, Dynamic> > xiter;
			uint Niter, Diter;  

			fin>>Niter;
			fin>>M;
			fin>>Diter; 

			N.push_back(Niter); 
			D.push_back(Diter); 

			MatrixXd data(Diter,Niter);
			VectorXu words(M);

			for (uint j=0; j<M; ++j) 
				fin>>words(j); 
		
			for (uint j=1; j<(Diter+1); ++j) 
				for (uint i=0; i<Niter; ++i) 
					fin>>data(j-1,i);
		
			uint count = 0;
			for (uint j=0; j<M; ++j)
			{
				xiter.push_back(data.middleCols(count,words[j]));
				count+=words[j];
			}
			x.push_back(xiter);
		}
		fin.close();
	}



	if(uint(nuIn.size())!=NumObs && !nuIn.empty()) { 
		cerr << "Error specified number nu parameters does not equal number of data components" << endl;
		cerr << "#N nu=" << nuIn.size() << ", #data components=" << NumObs <<  "." << endl;
		cerr << "exiting" << endl;
		return(-1); 
	}
	if(uint(deltaOffsetIn.size())!=NumObs && !deltaOffsetIn.empty()) { 
		cerr << "Error specified number delta offset parameters does not equal number of data components" << endl;
		cerr << "#N nu=" << deltaOffsetIn.size() << ", #data components=" << NumObs <<  "." << endl;
		cerr << "exiting" << endl;
		return(-1); 
	}
	
	
	boost::mt19937 rndGen(9191);
	DirMultiNaiveBayes<double> *naive_samp; 
	
	if(paramsF.compare("")){
		//initialize from file 
		ifstream fin(paramsF.data(),ifstream::in);
			naive_samp = new DirMultiNaiveBayes<double>(fin,&rndGen);
			naive_samp->loadData(x); 
		fin.close();
	} else {
		//initialize here 


		vector<string> baseDist; //base distributions for each component
	
		if(vm.count("b")) { 
			baseDist.clear();
			baseDist = vm["b"].as<vector<string> >();

			if(uint(baseDist.size())!=NumObs) { 
				cerr << "Error specified number of base distributions not equal number of data components" << endl;
				cerr << "#bases =" << baseDist.size() << ", #data components=" << NumObs <<  "." << endl;
				cerr << "exiting" << endl;
				return(-1); 
			}

		} else {
			baseDist.clear();
			baseDist = vector<string>(NumObs, "NiwSampled");
		}


		VectorXd alpha = 10.0*VectorXd::Ones(K);

		Dir<Catd,double> dir(alpha,&rndGen); 
		vector<boost::shared_ptr<BaseMeasure<double> > > niwSampled;
		niwSampled.reserve(NumObs);

		//creates thetas  
		for(uint m=0;m<NumObs ; ++m) 
		{
			if(iequals(baseDist[m], "NiwSampled")) { //sampled normal inversed wishart
				double nu;
				if(nuIn.empty()) {
					nu = D[m]+1;
				}else {
					nu = nuIn[m];
				}
				double kappa = D[m]+1;
				
				MatrixXd Delta = MatrixXd::Identity(D[m],D[m]);
				if(deltaOffsetIn.empty()) {
					Delta *= 0.1; 
				} else {
					Delta *= deltaOffsetIn[m]; 
				}
				Delta *= nu;
				
				VectorXd theta = VectorXd::Zero(D[m]);

				NIW<double> niw(Delta,theta,nu,kappa,&rndGen);
				boost::shared_ptr<NiwSampled<double> > tempBase( new NiwSampled<double>(niw));
				niwSampled.push_back(boost::shared_ptr<BaseMeasure<double> >(tempBase));

			} else if(iequals(baseDist[m], "NiwSphere")) {
				//IW is one dimention smaller (makes sense since it's on tangent space)
				//double nu = D[m];
				double nu;
				if(nuIn.empty()) {
					nu = D[m]+1;
				}else {
					nu = nuIn[m];
				}
				double kappa = D[m];
				
				MatrixXd Delta = MatrixXd::Identity(D[m]-1,D[m]-1);
				if(deltaOffsetIn.empty()) {
					Delta *= pow(15.0*M_PI/180.0,2); 
				} else {
					Delta *= deltaOffsetIn[m]; 
				}
				Delta *= nu;
				
				IW<double> iw(Delta,nu,&rndGen);
				boost::shared_ptr<NiwSphere<double> > tempBase( new NiwSphere<double>(iw,&rndGen));
				niwSampled.push_back(boost::shared_ptr<BaseMeasure<double> >(tempBase));

			} else {
				cerr << "error with base distributions (check help) ... returning." << endl;
				return(-1); 
			}
		}

		naive_samp = new DirMultiNaiveBayes<double>(dir,niwSampled);
		naive_samp->initialize( x );
	}

	Timer tlocal;
	tlocal.tic();

	
  
	cout << "multiObsNaiveBayesian Clustering:" << endl; 
	cout << "Ndocs=" << M << endl; 
	cout << "NumComp=" << NumObs << endl; 
	cout << "NumData,Dim= ";
	for(uint n=0; n<NumObs; ++n)
		cout << "[" << N[n] << ", " << D[n] << "]; "; 
	cout << endl; 

	cout << "Num Cluster = " << K << ", (" << T << " iterations)." << endl;



	naive_samp->inferAll(T,verbose);


	if (pathOut.compare(""))
	{
		ofstream fout(pathOut.data(),ofstream::out);
		
		//streambuf *coutbuf = std::cout.rdbuf(); //save old cout buffer
		//cout.rdbuf(fout.rdbuf()); //redirect std::cout to fout1 buffer

			//naive_samp.dump(fout,fout);
			naive_samp->dump_clean(fout);
			//naive_samp.dump_clean();

		//std::cout.rdbuf(coutbuf); //reset to standard output again

		fout.close();
	}

	tlocal.displayElapsedTimeAuto();

	delete naive_samp; 

	return(0); 
	
};
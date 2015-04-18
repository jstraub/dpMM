/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>, Randi Cabezas <rcabezas@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */

#pragma once
#include <iostream>
#include <stdint.h>
#include <vector>
#include <Eigen/Dense>

#include <boost/shared_ptr.hpp>

#include <dpMM/dpMM.hpp>
#include <dpMM/cat.hpp>
#include <dpMM/dir.hpp>
#include <dpMM/niw.hpp>
#include <dpMM/sampler.hpp>
#include <dpMM/basemeasure.hpp>
#include <dpMM/niwBaseMeasure.hpp>
#include <dpMM/niwSphere.hpp>
#include <dpMM/dirBaseMeasure.hpp>
#include <mmf/mfBaseMeasure.hpp>

using namespace Eigen;
using std::cout;
using std::endl;
using boost::shared_ptr;
using std::vector;

template<typename T=double>
class DirMultiNaiveBayes : public DpMM<T>{

public:
  DirMultiNaiveBayes(std::ifstream &in, boost::mt19937 *rng);
  DirMultiNaiveBayes(const Dir<Cat<T>, T>& alpha, const vector<boost::shared_ptr<BaseMeasure<T> > >&thetas);
  DirMultiNaiveBayes(const Dir<Cat<T>, T>& alpha, const vector< vector<boost::shared_ptr<BaseMeasure<T> > > >&thetas);
  virtual ~DirMultiNaiveBayes();

  virtual void loadData(const vector<vector<Matrix<T,Dynamic,Dynamic> > > &x);//does nothing other than load data
  virtual void initialize(const vector<vector< Matrix<T,Dynamic,Dynamic> > >&x);
  virtual void initialize(const vector<vector< Matrix<T,Dynamic,Dynamic> > >&x, VectorXu &z);
  virtual void initialize(const boost::shared_ptr<ClGMMData<T> >&cld)
    {cout<<"not supported"<<endl; assert(false);};

  virtual void sampleLabels();
  virtual void MAPLabel();
  virtual void sampleParameters();

  virtual T logJoint(bool verbose=false);
  virtual const VectorXu& labels(){return z_;};
  virtual const VectorXu& getLabels(){return z_;};
  virtual uint32_t getK() const { return K_;};
  virtual uint32_t getM() const { return M_;};
  virtual uint32_t getN() const { return Nd_;};

//  virtual MatrixXu mostLikelyInds(uint32_t n);
  virtual MatrixXu mostLikelyInds(uint32_t n, Matrix<T,Dynamic,Dynamic>& logLikes);

  Matrix<T,Dynamic,1> getCounts();

  virtual void inferAll(uint32_t nIter, bool verbose=false);

  virtual void dump(std::ofstream& fOutMeans, std::ofstream& fOutCovs);
  virtual void dump_clean(std::ofstream &out);

  virtual vector<boost::shared_ptr<BaseMeasure<T> > > getThetas(uint32_t m) {
	  return(thetas_[m]);
  };
  virtual boost::shared_ptr<BaseMeasure<T> > getThetas(uint32_t m, uint32_t k) {
	  return(thetas_[m][k]);
  };

  virtual vector<T> evalLogLik(const vector<Matrix<T,Dynamic,1> > xnew, const vector<uint32_t> clusterInd,
							   const vector<uint32_t> comp2eval =vector<uint32_t>());

  virtual uint32_t sampleLabels(const vector<Matrix<T,Dynamic,1> > xnew,
								const vector<uint32_t> comp2eval =vector<uint32_t>());
  virtual vector<uint32_t> sampleLabels(const vector<vector<Matrix<T,Dynamic,1> > > xnew,
										const vector<uint32_t> comp2eval =vector<uint32_t>());
  virtual uint32_t MAPLabels(const vector<Matrix<T,Dynamic,1> > xnew,
							 const vector<uint32_t> comp2eval =vector<uint32_t>());
  virtual vector<uint32_t> MAPLabels(const vector<vector<Matrix<T,Dynamic,1> > > xnew,
									 const vector<uint32_t> comp2eval =vector<uint32_t>());
  virtual void updatePDF();

  vector<uint32_t> getLogEvalItersHist() {return logJointIterEval;}
  vector<T> getLogJointHist() {return logJointHist;}

protected:
  virtual T evalLogLik(const vector<Matrix<T,Dynamic,1> > xnew, const uint32_t clusterInd,
					   const vector<uint32_t> comp2eval);
  virtual uint32_t labels_sample_max(const vector<Matrix<T,Dynamic,1> > xnew,
									 const vector<uint32_t> comp2eval, const bool return_MAP_labels=false);

  uint32_t Nd_;
  uint32_t K_; //num cluseters
  uint32_t M_; //num data sources
  Dir<Cat<T>, T> dir_;
  Cat<T> pi_;
#ifdef CUDA
  SamplerGpu<T>* sampler_;
#else
  Sampler<T>* sampler_;
#endif
  virtual void initialize_sampler();
  Matrix<T,Dynamic,Dynamic> pdfs_;
//  Cat cat_;
  vector<vector<boost::shared_ptr<BaseMeasure<T> > > > thetas_;  // theta_[M][K]

  //suffiecient stats
  vector<vector<Matrix<T,Dynamic,Dynamic> > > x_; //x_[M][doc](:,word)
  VectorXu z_;

  virtual void helper_setDims();
  vector<VectorXu> dataDim;

  vector<uint32_t> logJointIterEval;
  vector<T> logJointHist;
};

// --------------------------------------- impl -------------------------------

template<typename T>
void DirMultiNaiveBayes<T>::initialize_sampler() {
	if (sampler_ != NULL) {
		delete sampler_;
		sampler_ = NULL;
	}
	//initialize sampler
	#ifdef CUDA
	  sampler_ = new SamplerGpu<T>(uint32_t(Nd_),K_,dir_.pRndGen_);
	#else
	  sampler_ = new Sampler<T>(dir_.pRndGen_);
	#endif
}

template<typename T>
DirMultiNaiveBayes<T>::DirMultiNaiveBayes(std::ifstream &in, boost::mt19937 *rng) :
 dir_(Matrix<T,2,1>::Ones(),rng), pi_(dir_.sample()),sampler_(NULL)
{
	//initialize the class from the file pointer given
	in >> M_;
	in >> K_;
	in >> Nd_;

	vector<uint32_t> dim;
	vector<baseMeasureType> type;
	Matrix<T,Dynamic,1> alpha(K_), pi(K_);

	for(uint32_t m = 0; m<M_; ++m){
		uint32_t temp;
		in >> temp;
		dim.push_back(temp);
	}

	for(uint32_t m = 0; m<M_; ++m){
		uint32_t temp;
		in >> temp;
		type.push_back(baseMeasureType(temp));
	}

	z_ = VectorXu(Nd_);
	for(uint32_t n=0; n<Nd_; ++n)
		in >> z_(n);	

	for(uint32_t k=0; k<K_; ++k)
		in >> alpha(k);

	for(uint32_t k=0; k<K_; ++k)
		in >> pi(k);

	pdfs_ = Matrix<T,Dynamic,Dynamic>(Nd_,K_);
	//for(uint32_t n=0; n<Nd_*K_; ++n)
	//	in >> pdfs_(n%Nd_, (n-(n%Nd_))/Nd_);
	//	//in >> pdfs_((n-(n%Nd_))/Nd_, n%Nd_);
	////pdfs_ = pdfs_.transpose();

	//get parameters
	for(uint32_t m=0; m<M_; ++m) {
		baseMeasureType typeIter = baseMeasureType(type[m]);
		vector<boost::shared_ptr<BaseMeasure<T> > > thetaM;
		if(typeIter==NIW_SAMPLED) {
			uint32_t Diter = dim[m];
			T nu, kappa;
			Matrix<T,Dynamic,Dynamic> scatter(Diter, Diter), sigma(Diter,Diter);
			Matrix<T,Dynamic,1> theta(Diter), mu(Diter);
			for(uint32_t k=0; k<K_; ++k) {
				//get nu and kappa
					in>> nu;
					in>> kappa;
				//get theta
					for(uint32_t n=0; n<Diter; ++n)
						in >> theta(n);
					theta = theta.transpose();
				//get scatter
					for(uint32_t n=0; n<Diter*Diter; ++n)
						in >> scatter((n-(n%Diter))/Diter, n%Diter);
				//get mean
					for(uint32_t n=0; n<Diter; ++n)
						in >> mu(n);
					mu = mu.transpose();
				//get sigma
					for(uint32_t n=0; n<Diter*Diter; ++n)
						in >> sigma((n-(n%Diter))/Diter, n%Diter);

				//build theta[m][k]
				NIW<T> niw(scatter,theta,nu,kappa,rng);
				Normal<T> normal(mu,sigma,rng);
				boost::shared_ptr<NiwSampled<T> > baseIter( new NiwSampled<T>(niw, normal));

				//set
				thetaM.push_back(boost::shared_ptr<BaseMeasure<T> >(baseIter));
			}
		} else if(typeIter==NIW_SPHERE) {
			uint32_t Diter = dim[m]-1;
			T nu;
			T counts;
			Matrix<T,Dynamic,Dynamic> scatter(Diter, Diter), sigma(Diter,Diter),delta(Diter,Diter);
			Matrix<T,Dynamic,1> mu(Diter+1), mu_prior(Diter), north(Diter+1);
			for(uint32_t k=0; k<K_; ++k) {
				//get nu
					in>> nu;
					in>> counts;
				//get prior mean
					for(uint32_t n=0; n<Diter; ++n)
						in>> mu_prior(n);
				//get scatter
					for(uint32_t n=0; n<Diter*Diter; ++n)
						in >> scatter((n-(n%Diter))/Diter, n%Diter);
				//get delta
					for(uint32_t n=0; n<Diter*Diter; ++n)
						in >> delta((n-(n%Diter))/Diter, n%Diter);
				//get mean
					for(uint32_t n=0; n<(Diter+1); ++n)
						in >> mu(n);
					mu = mu.transpose();
				//get sigma
					for(uint32_t n=0; n<Diter*Diter; ++n)
						in >> sigma((n-(n%Diter))/Diter, n%Diter);
				//get north
					for(uint32_t n=0; n<(Diter+1); ++n)
						in >> north(n);
					north = north.transpose();

				//build theta[m][k]
				IW<T> iw(delta,nu, scatter, mu_prior, counts, rng);
				//IW<T> iw(delta,nu, rng);
				NiwSphere<T> niwSp(iw,rng);
				niwSp.S_ = Sphere<T>(north);
				niwSp.normalS_ = NormalSphere<T>(mu,sigma,rng);
				

				//set
				thetaM.push_back(boost::shared_ptr<BaseMeasure<T> >(niwSp.copy()));
				//boost::shared_ptr<NiwSphere<T> > baseIter( new NiwSphere<T>(iw,rng));
				//thetaM.push_back(boost::shared_ptr<BaseMeasure<T> >(baseIter));
				
			}

		} else if(typeIter==DIR_SAMPLED) {
			uint32_t Diter = dim[m];
			T localCount;
			Matrix<T,Dynamic,1> post_alpha(Diter), counts(Diter), pdf(Diter);

			for(uint32_t k=0; k<K_; ++k) {
				//get dir alpha
				for(uint32_t n=0; n<Diter; ++n)
					in >> post_alpha(n); 		

				//get dir counts
				for(uint32_t n=0; n<Diter; ++n)
					in >> counts(n); 		

				//get disc pdf
				for(uint32_t n=0; n<Diter; ++n)
					in >> pdf(n); 		

				in >> localCount;

				//build theta[m][k]
				Dir<Cat<T>,T> dirBase(post_alpha,counts, rng);
				DirSampled<Cat<T>,T> dirSamp(dirBase);
				Cat<T> disc(pdf,rng);
				dirSamp.disc_ = disc;
				dirSamp.count_	= localCount;

				//set
				thetaM.push_back(boost::shared_ptr<BaseMeasure<T> >(dirSamp.copy()));
			}

		} else if(typeIter==MF_T) {
//			uint32_t Diter = dim[m];
//			T localCount;
			Matrix<T,Dynamic,1> post_alpha(6), counts(6), pi_pdf(6);

      Matrix<T,3,3>  R;
      Dir<Cat<T>, T> alpha(post_alpha,rng);
      Cat<T> pi(pi_pdf,rng);
      std::vector<shared_ptr<IwTangent<T> > > iwTs;
      std::vector<NormalSphere<T> > TGs;
      uint32_t nIter;

			for(uint32_t k=0; k<K_; ++k)
      {
        in >> nIter;
        //get dir alpha
        for(uint32_t n=0; n<6; ++n) in >> post_alpha(n); 		
        //get dir counts
        for(uint32_t n=0; n<6; ++n) in >> counts(n); 		
        //get pi pdf
        for(uint32_t n=0; n<6; ++n) in >> pi_pdf(n); 		
        alpha.alpha_ = post_alpha;
        alpha.setCounts(counts);
        pi.pdf(pi_pdf);
        // get rotation
        for(uint32_t n=0; n<9; ++n) in >> R(n/3,n%3); 		
        for(uint32_t j=0; j<6; ++j)
        {
          // load IW in tangent space
          Matrix<T,Dynamic,Dynamic> Delta(2,2);
          Matrix<T,Dynamic,Dynamic> Scatter(2,2);
          Matrix<T,Dynamic,1> mean(2);
          T nu,count;
          in >> nu;
          for(uint32_t n=0; n<4; ++n) in >> Delta(n/2,n%2); 		
          for(uint32_t n=0; n<4; ++n) in >> Scatter(n/2,n%2); 		
          for(uint32_t n=0; n<2; ++n) in >> mean(n); 		
          in >> count;
          IW<T> iw(Delta,nu,Scatter,mean,count,rng);
          iwTs.push_back(shared_ptr<IwTangent<T> >(
                new IwTangent<T>(iw,rng)));
          // load tangent space gaussians
          Matrix<T,Dynamic,Dynamic> Sigma(2,2);
          Matrix<T,Dynamic,1> mu(3);
          for(uint32_t n=0; n<4; ++n) in >> Sigma(n/2,n%2); 		
          for(uint32_t n=0; n<3; ++n) in >> mu(n); 		
          TGs.push_back(NormalSphere<T>(mu,Sigma,rng));
          iwTs[j]->normalS_ = TGs[j];
        };

        MfPrior<T> mfPrior(alpha, iwTs , nIter);
        MF<T> mf(R,pi,TGs);
				//set
				thetaM.push_back(boost::shared_ptr<BaseMeasure<T> >(
              new MfBase<T>(mfPrior, mf)));
			}

		} else {
				std::cerr << "[DirMultiNaiveBayes::dump_clean] error saving...returning" << endl;
				return;
		}
		thetas_.push_back(thetaM);
	}

	dir_ =  Dir<Cat<T>, T>(alpha,rng);
	pi_ = Cat<T>(pi,rng);


	this->initialize_sampler();

	//cout << "M:" << M_ << " K: " << K_ << " nd: " << Nd_ << endl;
	//for(uint32_t m = 0; m<M_; ++m){
	//	cout << m << ": d="<< dim[m] << ", type=" << type[m] << endl;
	//}
	//cout << "z_ " << z_.transpose() << endl;
	//cout << "alpha " << alpha.transpose() << endl;
}

template<typename T>
DirMultiNaiveBayes<T>::DirMultiNaiveBayes(const Dir<Cat<T>,T>& alpha,
    const vector<boost::shared_ptr<BaseMeasure<T> > >& thetas) :
  sampler_(NULL), K_(alpha.K_), dir_(alpha), pi_(dir_.sample()), M_(uint32_t(thetas.size()))
{
	for (uint32_t m=0; m<M_; ++m)
	{	
		vector<boost::shared_ptr<BaseMeasure<T> > > temp;
		for (uint32_t k=0; k<K_; ++k)
		{   	
      		temp.push_back(boost::shared_ptr<BaseMeasure<T> >(thetas[m]->copy()));
      	}
      thetas_.push_back(temp);
    }

#ifndef NDEBUG
	for(uint32_t m=0; m<int(M_); ++m) {
		for(int k=0; k<int(K_); ++k) {
				thetas_[m][k]->print();
		}
	}
#endif


};

template<typename T>
DirMultiNaiveBayes<T>::DirMultiNaiveBayes(const Dir<Cat<T>,T>& alpha,
    const vector< vector<boost::shared_ptr<BaseMeasure<T> > > >& theta) :
 K_(alpha.K_), M_(uint32_t(theta.size())),dir_(alpha),
  pi_(dir_.sample()), sampler_(NULL),thetas_(theta)
{ };

template<typename T>
DirMultiNaiveBayes<T>::~DirMultiNaiveBayes()
{
  if (sampler_ != NULL) {
	delete sampler_;
	sampler_ = NULL;
  }
};

template <typename T>
Matrix<T,Dynamic,1> DirMultiNaiveBayes<T>::getCounts()
{
  return counts<T,uint32_t>(z_,K_);
};

template<typename T>
void DirMultiNaiveBayes<T>::loadData(const vector<vector<Matrix<T,Dynamic,Dynamic> > > &x){
  x_ = x;
  this->helper_setDims();
}

template<typename T>
void DirMultiNaiveBayes<T>::initialize(const vector< vector< Matrix<T,Dynamic,Dynamic> > > &x)
{
  uint32_t Nd= uint32_t(x.front().size());

  // randomly init labels from prior
  VectorXu z;
  z.setZero(Nd);
  Cat<T> pi = dir_.sample();
  pi.sample(z);

  //delegate the initialization to the main intitialization function
  this->initialize(x,z);
};

template<typename T>
void DirMultiNaiveBayes<T>::initialize(const vector< vector< Matrix<T,Dynamic,Dynamic> > > &x, VectorXu &z)
{
  Nd_= uint32_t(x.front().size());

  //init data and labels from given
  x_ = x;
  z_ = z;

  pi_ = dir_.sample();

  pdfs_.setZero(Nd_,K_);

  this->initialize_sampler();
  this->helper_setDims();
  this->sampleParameters();
};



template<typename T>
void DirMultiNaiveBayes<T>::helper_setDims()
{
	dataDim.clear();
	dataDim.reserve(M_);
	for(uint32_t m=0; m<M_; ++m) {
		VectorXu temp(x_[m].size());
		for(uint32_t t=0; t<x_[m].size(); ++t)
			temp(t) = uint32_t(x_[m][t].cols());
		dataDim.push_back(temp);
	}
}

template<typename T>
void DirMultiNaiveBayes<T>::sampleLabels()
{
	// obtain posterior categorical under labels
	pi_ = dir_.posterior(z_).sample();
	//  cout<<pi_.pdf().transpose()<<endl;
	this->updatePDF();
	// sample z_i
	sampler_->sampleDiscPdf(pdfs_,z_);
};


template<typename T>
void DirMultiNaiveBayes<T>::updatePDF() {

// compute categorical distribution over label z_i
// no need to re-compute the array and log every iteration)
VectorXd logPdf_z_value = pi_.pdf().array().log();

#pragma omp parallel for
  for(int32_t d=0; d<int32_t(Nd_); ++d)
  {
	VectorXd logPdf_z = logPdf_z_value;
	for(uint32_t m=0; m<uint32_t(M_); ++m)
	{
		for(uint32_t k=0; k<K_; ++k)
		{
			//updated to SS
			logPdf_z[k] += thetas_[m][k]->logLikelihoodFromSS(x_[m][d]);
		}
	}
//    cout<<endl;
    // make pdf sum to 1. and exponentiate
    pdfs_.row(d) = (logPdf_z.array()-logSumExp(logPdf_z)).exp().matrix().transpose();
//    cout<<pi_.pdf().transpose()<<endl;
//    cout<<pdf.transpose()<<" |.|="<<pdf.sum();
//    cout<<" z_i="<<z_[d]<<endl;
  }

}


template<typename T>
void DirMultiNaiveBayes<T>::MAPLabel()
{
	/* it chooses the MAP label rather than sampling */
	this->updatePDF();

	#pragma omp parallel for
	 for(int32_t d=0; d<int32_t(Nd_); ++d) {
		 int r,c;
		 pdfs_.row(d).maxCoeff(&r, &c);
		 z_(d) = c;
	 }

};



template<typename T>
void DirMultiNaiveBayes<T>::sampleParameters()
{
//unpacks the contains here vector<vector<Matrix>> into what the posterior expects Matrix
	MatrixXu dim(M_,K_);
	for(uint32_t m=0; m<M_; ++m) {
		#pragma omp parallel for
		for(int32_t k=0; k<int32_t(K_); ++k) {
			VectorXu temp = (z_.array()==k).select(dataDim[m],0);
			dim(m,k) = temp.sum();	
		}
	}

	for(int32_t m=0; m<M_; ++m) {
	#ifdef _WINDOWS	
		//#pragma omp parallel for
		for(int32_t k=0; k<int32_t(K_); ++k) {
	#else
		#pragma omp parallel for
		for(int32_t k=0; k<int32_t(K_); ++k) {
	#endif

			if(dim(m,k)!=0) {

				//Matrix<T,Dynamic,1> ssIn = Matrix<T,Dynamic,1>::Zero(x_[m].front().rows());
				vector< Matrix<T,Dynamic,1> >dataIn;
				dataIn.reserve(dim(m,k));
				
				uint32_t count=0;

				for(int32_t d=0; d<Nd_; ++d) {
					if(z_[d]==k) {
						int add_size =int(x_[m][d].cols());
						//ssIn += x_[m][d]; //update iteration SS
						dataIn.push_back(x_[m][d]);
						count+=add_size;

						if(count==dim(m,k))
							break; //early out if you found them all
					}
				}
				//update values
				//thetas_[m][k]->posteriorFromSS(ssIn);

				//always sends zeros and look for zeros
				thetas_[m][k]->posteriorFromSS(dataIn,VectorXu::Zero(dim(m,k)),0);
				
			} else {
				//Matrix<T,Dynamic,1> ssIn = Matrix<T,Dynamic,1>::Zero(x_[m].front().rows());
				//ssIn[0]=1; //set counts to 1 to avoid inf
				//the posterior needs to reset	
				//thetas_[m][k]->posteriorFromSS(ssIn);
				//passing in one data point (all zeros, with 1 index value=0 and looking for 1)

				vector<Matrix<T,Dynamic,1> >dataIn;
				dataIn.push_back(Matrix<T,Dynamic,1>::Zero(x_[m].front().rows(),1));
				//the posterior needs to reset
        VectorXu zz = VectorXu::Zero(1);
				thetas_[m][k]->posteriorFromSS(dataIn,zz,1);
			}
		}
	}
};


template<typename T>
T DirMultiNaiveBayes<T>::logJoint(bool verbose)
{
  T logJoint = dir_.logPdf(pi_);
  if(verbose)
  	cout<<"\tlog p(pi)="<< logJoint << endl;


  for(int32_t m=0; m<int32_t(M_); ++m) {
	  T logPriorM = 0;
	  #pragma omp parallel for reduction(+:logPriorM)
	  for (int32_t k=0; k<int32_t(K_); ++k) {
		logPriorM = logPriorM + thetas_[m][k]->logPdfUnderPrior();
	  }
	  logJoint+=logPriorM;
	  if(verbose)
	  	cout<<"\tlog p(theta_" << m << ")="<< logPriorM << endl;

  }

	for (int32_t m=0; m<int32_t(M_); ++m) {
		T logThetaM=0;
		#pragma omp parallel for reduction(+:logThetaM)
		for (int32_t d=0; d<int32_t(Nd_); ++d) {
			logThetaM = logThetaM + thetas_[m][z_[d]]->logLikelihoodFromSS(x_[m][d]);
		}
		logJoint += logThetaM;
		if(verbose)
			cout<<"\tlog p(x|z,theta_" << m << ")=" << logThetaM << endl;
		
	}
    if(verbose)
		cout<<"log p(pi)*p(theta)*p(x|z,theta)=" << logJoint << endl;

  return logJoint;
};



template<typename T>
void DirMultiNaiveBayes<T>::inferAll(uint32_t nIter, bool verbose)
{
  if(verbose){
  	cout<<"[DirMultiNaiveBayes::inferALL] ------ inferingALL (nIter=" << nIter << ") ------"<<endl;
  	if(Nd_<=100) {
		cout <<"initial labels:"<< endl;
  		cout<<this->labels().transpose()<<endl;
	}
  }


  logJointIterEval.clear(); logJointHist.clear();
  if(verbose) {
	//all iterations stored
	logJointIterEval.reserve(nIter); logJointHist.reserve(nIter);
  } else {
	  //only mod 100 stored
	  logJointIterEval.reserve(int((nIter/100) + 1)); logJointHist.reserve(int((nIter/100) + 1));
  }

  for(uint32_t t=0; t<nIter; ++t)
  {
    this->sampleLabels();
    this->sampleParameters();
    if(verbose)
    {
      for(int m=0; m<int(M_); ++m) {
        for(int k=0; k<int(K_); ++k) {
          thetas_[m][k]->print();
        }
      }
    }
    if(verbose || t%100==0)
    {
      T iterLogJoint = this->logJoint(true) ;
      //log iterJoint Prob
      logJointIterEval.push_back(t);
      logJointHist.push_back(iterLogJoint);

      if(Nd_<=10) {
        cout << "[" << std::setw(3)<< std::setfill('0')
          << t <<"] label: "
          << this->labels().transpose()
          << " [joint= " << std::setw(6) << iterLogJoint << "]"<< endl;
      } else {
        cout << "[" << std::setw(3)<< std::setfill('0')
          << t <<"] joint= "
          << std::setw(6) << iterLogJoint << endl;
      }
    }
    if(verbose)
    {
      VectorXu Ns = counts<uint32_t,uint32_t>(this->labels(),K_).transpose();
      uint32_t K = K_;
      for(uint32_t k = 0; k<K_; ++k)
        if (Ns(k) == 0) --K;
      cout<<"@i "<<t<<": # "<<K<<" "<<std::setw(1) <<Ns.transpose() <<endl;
    }
  }
  //keeps the MAP label in memory
  this->MAPLabel();
}


template <typename T>
void DirMultiNaiveBayes<T>::dump(std::ofstream& fOutMeans, std::ofstream& fOutCovs)
{
	cout << "dumping MultiObs naiveBayes" << endl;
	cout << "doc index: " << endl;
	cout << this->labels().transpose() << endl;
	
	cout << "printing num components: " << endl;
	cout << M_ << endl;

	cout << "printing cluster params: " << endl;
	cout << K_ << endl;

	for(uint32_t m=0; m<M_; ++m) {
		cout << "component: " << m  << endl;
		for(uint32_t k=0; k<K_; ++k) {
			cout << "theta: " << k  << endl;
			thetas_[m][k]->print();
		}
	}

	cout << "printing mixture params: " << endl;
	pi_.print();
}


template <typename T>
void DirMultiNaiveBayes<T>::dump_clean(std::ofstream &out){
	//clean dump, only data with specific format
//FORMAT:
	//M 1x1
	//K	1x1
	//Nd 1x1
	//D[m] 1xM
	//Type[m] 1xM
	//labels 1xNd
	//Dir alpha 1xK
	//pi pdf 1xK
	//pdf KxNd
	// mixture parameters
	//params Loop over M then K each contains data type for specific type
	//for type 1 (NIWSampled)
		//---prior--- (NIW)
			//nu 1x1
			//kappa 1x1
			//theta 1xD
			//scatter DxD
		//---estimate (normal)
			//mu 1xD
			//Sigma DxD
	//for type 2 (NIWSphereFull)
		//----prior--- (IW)
			//nu 1x1
			//count 1x1
			//mean 1x(D-1)
			//scatter (D-1)x(D-1)
			//Delta	 (D-1)x(D-1)
		//--posterior (NormalSphere)
			//mean 1x(D-1)
			//Sigma (D-1)x(D-1)
		//--sphere--- (Sphere)
			//north 1x(D-1)
	//for type 3 (DirSampled)
		//--posterior (Dir)
			//alpha 1xK
			//counts 1xK
		//--distribution (Cat)
			//pdf 1xK
		//--counts (scalar)
			//counts 1x1
	//logJoint history
		// Niter 1x1
		// iterValue 1xNiter (iteration corresponding to the logValue)
		// logJoint	 1xNiter (logJoint)

	//this fixes issues with eigen matrices printing (eg, 00-0.7 )
	int curPres = int(out.precision());
	out.precision(10);
	IOFormat fullPresPrint(FullPrecision,DontAlignCols);

	//prints headers
	out << M_ << endl
		 << K_ << endl
		 << Nd_ << endl;

	//print dim
	for(uint32_t m=0; m<M_; ++m) {
		vector<boost::shared_ptr<BaseMeasure<T> > >  theta_base = this->getThetas(m);
		uint32_t temp = theta_base.front()->getDim();
		out << temp << " ";
		//out << x_[m].front().rows() << " ";
	}
	out << endl;
	//print type
	for(uint32_t m=0; m<M_; ++m) {
		out << thetas_[m].front()->getBaseMeasureType() << " ";
	}
	out << endl;

	//print labels
	out << this->labels().transpose() << endl;

	//print mixture parameters
	out << this->dir_.alpha_.transpose() << endl;
	out << this->pi_.pdf_.transpose().format(fullPresPrint) << endl;
	//out << this->pdfs_.transpose().format(fullPresPrint) << endl;

	//print parameters
	for(uint32_t m=0; m<M_; ++m) {
		vector<boost::shared_ptr<BaseMeasure<T> > >  theta_base = this->getThetas(m);
		for(uint32_t k=0; k<K_; ++k) {
			baseMeasureType type = theta_base[k]->getBaseMeasureType();
			if(type==NIW_SAMPLED) {
				boost::shared_ptr<NiwSampled<T> >  *theta_iter =
						reinterpret_cast<boost::shared_ptr<NiwSampled<T> >* >( &theta_base[k]);
					//printing prior
					NIW<T> prior = theta_iter->get()->niw0_;
					out << prior.nu_				 << endl <<
						    prior.kappa_			 << endl <<
						    prior.theta_.transpose() << endl <<
						    prior.Delta_.format(fullPresPrint)	<<	endl;

					//printing posterior
					Normal<T> norm = theta_iter->get()->normal_;
					out << norm.mu_.transpose() << endl;
					out << norm.Sigma().format(fullPresPrint) << endl;
			} else if(type==NIW_SPHERE) {
        boost::shared_ptr<NiwSphere<T> >  *theta_iter =
          reinterpret_cast<boost::shared_ptr<NiwSphere<T> >* >(
              &theta_base[k]);
				//prior
				IW<T> prior = theta_iter->get()->iw0_;
				out << prior.nu_ 				 << endl
					 << prior.count()			 << endl
				 	 << prior.mean().transpose() << endl
					 << prior.scatter().format(fullPresPrint)			 << endl
					 << prior.Delta_.format(fullPresPrint)			 << endl;

				//posterior 	
				NormalSphere<T> norm = theta_iter->get()->normalS_;
				out << norm.getMean().transpose().format(fullPresPrint) << endl;
				out << norm.Sigma().format(fullPresPrint) << endl;
				//sphere 	
				Sphere<T> sp = theta_iter->get()->S_;
				out << sp.north().transpose() << endl;
			} else if(type==DIR_SAMPLED) {
				boost::shared_ptr<DirSampled<Cat<T>,T> >  *theta_iter =
						reinterpret_cast<boost::shared_ptr<DirSampled<Cat<T>,T> >* >( &theta_base[k]);
				//posterior
				Dir<Catd,T>  post = theta_iter->get()->dir0_;
				out <<	post.alpha_.transpose()		<< endl;
				out <<	post.counts().transpose()	<< endl;

				//distribution
				Catd dist = theta_iter->get()->disc_;
				out <<	dist.pdf_.transpose() << endl;
				
				//counts
				T counts = theta_iter->get()->count_;
				out << counts << endl;

			} else if(type==MF_T) {
        boost::shared_ptr<MfBase<T> > *theta_iter =
          reinterpret_cast<boost::shared_ptr<MfBase<T> >* >(
              &theta_base[k]);

        out<< theta_iter->get()->mf0_.T_;

				//posterior
				Dir<Cat<T>, T>  post = theta_iter->get()->mf0_.dirMM().Alpha();
				out <<	post.alpha_.transpose()		<< endl;
				out <<	post.counts().transpose()	<< endl;
				Cat<T> pi = theta_iter->get()->mf0_.dirMM().Pi();
				out <<	pi.pdf().transpose() << endl;

        out << theta_iter->get()->mf_.R().format(fullPresPrint)<<endl;
        for(uint32_t j=0; j<6; ++j)
        {
          out<< theta_iter->get()->mf0_.theta(j)->iw0_.nu_;
          out<< theta_iter->get()->mf0_.theta(j)->iw0_.Delta_;
          out<< theta_iter->get()->mf0_.theta(j)->iw0_.scatter();
          out<< theta_iter->get()->mf0_.theta(j)->iw0_.mean();
          out<< theta_iter->get()->mf0_.theta(j)->iw0_.count();
          out<< theta_iter->get()->mf0_.theta(j)->normalS_.Sigma();
          out<< theta_iter->get()->mf0_.theta(j)->normalS_.getMean();
        }
				
			} else {
					std::cerr << "[DirMultiNaiveBayes::dump_clean] error saving...returning" << endl;
					return;
			}
		}
	}


	//print logHistory
	out << int(logJointHist.size()) << endl;
	for(int i=0; i<logJointIterEval.size(); ++i)
		out << logJointIterEval[i] << " ";
	out << endl;

	for(int i=0; i<logJointHist.size(); ++i)
		out << logJointHist[i] << " ";
	out << endl;

	out.precision(curPres);

}


//template <typename T>
//void DirMultiNaiveBayes<T>::dump_clean(std::ofstream &out){
//	streambuf *coutbuf = std::cout.rdbuf(); //save old cout buffer
//	cout.rdbuf(out.rdbuf()); //redirect std::cout to fout1 buffer
//	this->dump_clean(); //write using cout to the specified buffer
//	std::cout.rdbuf(coutbuf); //reset to standard output again
//}


template <typename T>
T DirMultiNaiveBayes<T>::evalLogLik(const vector<Matrix<T,Dynamic,1> > xnew,
							  const uint32_t clusterInd, const vector<uint32_t> comp2eval)
{
	//T logJoint = pi_.pdf_(clusterInd);
	T logJoint  = 0;
	for (int32_t m=0; m<int32_t(comp2eval.size()); ++m)
	{
		logJoint += thetas_[comp2eval[m]][clusterInd]->logLikelihoodFromSS(xnew[m]);
	}

  return logJoint;
}


template <typename T>
vector<T> DirMultiNaiveBayes<T>::evalLogLik(const vector<Matrix<T,Dynamic,1> > xnew,
											const vector<uint32_t> clusterInd,
											const vector<uint32_t> comp2eval) {
	
	vector<uint32_t> comp2evalLocal = comp2eval;
	if(comp2evalLocal.empty()) {
		for(uint32_t m=0; m<M_; ++m)
			comp2evalLocal.push_back(m);
	}

	vector<T> out;
	for(uint32_t k=0; k<uint32_t(clusterInd.size()); ++k) {
		out.push_back(this->evalLogLik(xnew,clusterInd[k],comp2evalLocal));
	}
	return(out);
}


template <typename T>
uint32_t DirMultiNaiveBayes<T>::labels_sample_max(const vector<Matrix<T,Dynamic,1> > xnew,
												  const vector<uint32_t> comp2eval, const bool return_MAP_labels)
{
	/* xnew in the form x[docs][m][SS] */

VectorXd logPdf_z = pi_.pdf().array().log();

for(int32_t m=0; m<comp2eval.size(); ++m)
{
	for(int32_t k=0; k<int32_t(K_); ++k)
	{
		logPdf_z[k] += thetas_[comp2eval[m]][k]->logLikelihoodFromSS(xnew[m]);
	}
}

// make pdf sum to 1. and exponentiate
Matrix<T,Dynamic,Dynamic> pdfLocal =  Matrix<T,Dynamic,Dynamic>(1,K_);

pdfLocal = (logPdf_z.array()-logSumExp(logPdf_z)).exp().matrix().transpose();

VectorXu zout = VectorXu(1);

if(return_MAP_labels) {
	// return MAP label
	int r,c;
	pdfLocal.maxCoeff(&r, &c);
	zout(0) = c;
} else {
	// sample z_i
	sampler_->sampleDiscPdf(pdfLocal,zout);
}

  return(zout(0));
};




template <typename T>
uint32_t DirMultiNaiveBayes<T>::sampleLabels(const vector<Matrix<T,Dynamic,1> > xnew,
											 const vector<uint32_t> comp2eval) {
	return(this->labels_sample_max(xnew, comp2eval, false));
}

template <typename T>
vector<uint32_t> DirMultiNaiveBayes<T>::sampleLabels(const vector<vector<Matrix<T,Dynamic,1> > > xnew,
													 const vector<uint32_t> comp2eval)
{
	/* xnew in the form x[doc][m][SS] */
	vector<uint32_t> out;
	for(uint32_t d=0; d<xnew.size(); ++d) {
		out.push_back(this->sampleLabels(xnew[d],comp2eval));
	}
	return(out);
}


template <typename T>
uint32_t DirMultiNaiveBayes<T>::MAPLabels(const vector<Matrix<T,Dynamic,1> > xnew,
										  const vector<uint32_t> comp2eval) {
	return(this->labels_sample_max(xnew, comp2eval, true));
}


template <typename T>
vector<uint32_t> DirMultiNaiveBayes<T>::MAPLabels(const vector<vector<Matrix<T,Dynamic,1> > > xnew,
												  const vector<uint32_t> comp2eval) {
	/* xnew in the form x[docs][m][SS] */
	vector<uint32_t> out;
	for(uint32_t d=0; d<xnew.size(); ++d) {
		out.push_back(this->MAPLabels(xnew[d],comp2eval));
	}
	return(out);
}

template<typename T>
MatrixXu DirMultiNaiveBayes<T>::mostLikelyInds(uint32_t n, Matrix<T,Dynamic,Dynamic>& logLikes)
{
  MatrixXu inds = MatrixXu::Zero(n,K_);
  logLikes = Matrix<T,Dynamic,Dynamic>::Ones(n,K_);

#pragma omp parallel for
  for (int32_t k=0; k<K_; ++k)
  {
    for (uint32_t i=0; i<z_.size(); ++i)
      if(z_(i) == k)
      {
        T logLike = 0.;
        // iterate over datasources and sum up their logLikes
        for(uint32_t m=0; m<uint32_t(M_); ++m)
        {
          logLike += thetas_[m][z_[i]]->logLikelihoodFromSS(x_[m][i]);
        }
        // keep only the top n and sorted
        for (uint32_t j=0; j<n; ++j)
          if(logLikes(j,k) < logLike)
          {
            for(uint32_t l=n-1; l>j; --l)
            {
              logLikes(l,k) = logLikes(l-1,k);
              inds(l,k) = inds(l-1,k);
            }
            logLikes(j,k) = logLike;
            inds(j,k) = i;
//            cout<<"after update "<<logLike<<endl;
//            Matrix<T,Dynamic,Dynamic> out(n,K_*2);
//            out<<logLikes.cast<T>(),inds.cast<T>();
//            cout<<out<<endl;
            break;
          }
      }
  }
  cout<<"::mostLikelyInds: logLikes"<<endl;
  cout<<logLikes<<endl;
  cout<<"::mostLikelyInds: inds"<<endl;
  cout<<inds<<endl;
  return inds;
};

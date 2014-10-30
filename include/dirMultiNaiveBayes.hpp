#pragma once
#include <iostream>
#include <stdint.h>
#include <vector>
#include <Eigen/Dense>

#include <boost/shared_ptr.hpp>

#include "dpMM.hpp"
#include "cat.hpp"
#include "dir.hpp"
#include "niw.hpp"
#include "sampler.hpp"
#include "basemeasure.hpp"
#include "niwBaseMeasure.hpp"

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
  virtual void initialize(const boost::shared_ptr<ClData<T> >&cld)
    {cout<<"not supported"<<endl; assert(false);};

  virtual void sampleLabels();
  virtual void sampleParameters();

  virtual T logJoint(bool verbose=false);
  virtual const VectorXu& labels(){return z_;};
  virtual const VectorXu& getLabels(){return z_;};
  virtual uint32_t getK() const { return K_;};

//  virtual MatrixXu mostLikelyInds(uint32_t n);
  //virtual MatrixXu mostLikelyInds(uint32_t n, Matrix<T,Dynamic,Dynamic>& logLikes);

  Matrix<T,Dynamic,1> getCounts();

  virtual void inferAll(uint32_t nIter, bool verbose=false);
  
  virtual void dump(std::ofstream& fOutMeans, std::ofstream& fOutCovs); 
  virtual void dump_clean();
  virtual void dump_clean(std::ofstream &out);
  
  virtual vector<boost::shared_ptr<BaseMeasure<T> > > getThetas(uint32_t m) {
	  return(thetas_[m]);
  };
  virtual boost::shared_ptr<BaseMeasure<T> > getThetas(uint32_t m, uint32_t k) {
	  return(thetas_[m][k]);
  };

protected: 
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

  vector<vector<Matrix<T,Dynamic,Dynamic> > > x_; //x_[M][doc](:,word)
  VectorXu z_;
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
sampler_(NULL), dir_(Matrix<T,2,1>::Ones(),rng), pi_(dir_.sample()){
	//initialize the class from the file pointer given
	in >> M_; 
	in >> K_;
	in >> Nd_;

	vector<uint> dim;
	vector<baseMeasureType> type;
	Matrix<T,Dynamic,1> alpha(K_), pi(K_); 

	for(uint m = 0; m<M_; ++m){
		uint temp; 
		in >> temp; 
		dim.push_back(temp);
	}

	for(uint m = 0; m<M_; ++m){
		uint temp;
		in >> temp;
		type.push_back(baseMeasureType(temp));
	}

	z_ = VectorXu(Nd_);
	for(uint n=0; n<Nd_; ++n) 
		in >> z_(n);	

	for(uint k=0; k<K_; ++k)
		in >> alpha(k);

	for(uint k=0; k<K_; ++k)
		in >> pi(k); 

	pdfs_ = Matrix<T,Dynamic,Dynamic>(Nd_,K_);
	for(uint n=0; n<Nd_*K_; ++n)
		in >> pdfs_(n%Nd_, (n-(n%Nd_))/Nd_); 
		//in >> pdfs_((n-(n%Nd_))/Nd_, n%Nd_); 
	//pdfs_ = pdfs_.transpose();

	//get parameters
	for(uint32_t m=0; m<M_; ++m) {
		baseMeasureType typeIter = baseMeasureType(type[m]);
		vector<boost::shared_ptr<BaseMeasure<T> > > thetaM; 
		if(typeIter==NIW_SAMPLED) {
			uint Diter = dim[m];
			T nu, kappa; 
			Matrix<T,Dynamic,Dynamic> scatter(Diter, Diter), sigma(Diter,Diter); 
			Matrix<T,Dynamic,1> theta(Diter), mu(Diter); 
			for(uint32_t k=0; k<K_; ++k) {
				//get nu and kappa
					in>> nu;
					in>> kappa;
				//get theta
					for(uint n=0; n<Diter; ++n)
						in >> theta(n); 
					theta = theta.transpose(); 
				//get scatter
					for(uint n=0; n<Diter*Diter; ++n)
						in >> scatter((n-(n%Diter))/Diter, n%Diter); 
				//get mean
					for(uint n=0; n<Diter; ++n)
						in >> mu(n); 
					mu = mu.transpose();
				//get sigma
					for(uint n=0; n<Diter*Diter; ++n)
						in >> sigma((n-(n%Diter))/Diter, n%Diter); 

				//build theta[m][k]
				NIW<T> niw(scatter,theta,nu,kappa,rng);
				Normal<T> normal(mu,sigma,rng);
				boost::shared_ptr<NiwSampled<T> > baseIter( new NiwSampled<T>(niw, normal));

				//set 
				thetaM.push_back(boost::shared_ptr<BaseMeasure<T> >(baseIter));

			}
		} else if(typeIter==NIW_SPHERE_FULL) {
			assert(false);
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
	//for(uint m = 0; m<M_; ++m){
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
};

template<typename T>
DirMultiNaiveBayes<T>::DirMultiNaiveBayes(const Dir<Cat<T>,T>& alpha, 
    const vector< vector<boost::shared_ptr<BaseMeasure<T> > > >& theta) :
 sampler_(NULL), K_(alpha.K_), dir_(alpha), pi_(dir_.sample()), M_(uint32_t(theta.size()), thetas_(theta) ) 
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
}

template<typename T>
void DirMultiNaiveBayes<T>::initialize(const vector< vector< Matrix<T,Dynamic,Dynamic> > > &x)
{
  uint Nd= uint32_t(x.front().size());

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

#pragma omp parallel for
for(int32_t m=0; m<int32_t(M_); ++m)
  for(uint32_t k=0; k<K_; ++k)
		thetas_[m][k]->posterior(x_[m],z_,k);
};



template<typename T>
void DirMultiNaiveBayes<T>::sampleLabels()
{
// obtain posterior categorical under labels
pi_ = dir_.posterior(z_).sample();
//  cout<<pi_.pdf().transpose()<<endl;
  
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
	//      cout<<thetas_[k].logLikelihood(x_.col(d))<<" ";
			for(uint32_t w=0; w<x_[m][d].cols(); ++w)
			{
				logPdf_z[k] += thetas_[m][k]->logLikelihood(x_[m][d],w);
			}
		}
	}
//    cout<<endl;
    // make pdf sum to 1. and exponentiate
    pdfs_.row(d) = (logPdf_z.array()-logSumExp(logPdf_z)).exp().matrix().transpose();
//    cout<<pi_.pdf().transpose()<<endl;
//    cout<<pdf.transpose()<<" |.|="<<pdf.sum();
//    cout<<" z_i="<<z_[d]<<endl;
  }


  // sample z_i
  sampler_->sampleDiscPdf(pdfs_,z_);
};


template<typename T>
void DirMultiNaiveBayes<T>::sampleParameters()
{
#pragma omp parallel for 
	for(int32_t m=0; m<int32_t(M_); ++m) 
	{
	  for(uint32_t k=0; k<K_; ++k)
	  {
		thetas_[m][k]->posterior(x_[m],z_,k);
	//    cout<<"k:"<<k<<endl;
	//    thetas_[k]->print();
	  }
	}
};


template<typename T>
T DirMultiNaiveBayes<T>::logJoint(bool verbose)
{
  T logJoint = dir_.logPdf(pi_);
  if(verbose)
  	cout<<"log p(pi)="<<logJoint<<" -> ";

  for (int32_t m=0; m<int32_t(M_); ++m)
  {
	  #pragma omp parallel for reduction(+:logJoint)  
	  for (int32_t k=0; k<int32_t(K_); ++k)
	  {
		logJoint = logJoint + thetas_[m][k]->logPdfUnderPrior();
	  }
  }

  if(verbose)
	cout<<"log p(pi)*p(theta)="<<logJoint<<" -> ";

	for (int32_t m=0; m<int32_t(M_); ++m) 
	{
		for (int32_t d=0; d<int32_t(Nd_); ++d)
		{
			#pragma omp parallel for reduction(+:logJoint)  
			for(int32_t w=0; w<x_[m][d].cols(); ++w)
			{
				logJoint = logJoint + thetas_[m][z_[d]]->logLikelihood(x_[m][d],w);
			}
		}
	}
  
  if(verbose)
  	cout<<"log p(phi)*p(theta)*p(x|z,theta)="<<logJoint<<"]"<<endl;
  
  return logJoint;
};


//template<typename T>
//MatrixXu DirMultiNaiveBayes<T>::mostLikelyInds(uint32_t n, Matrix<T,Dynamic,Dynamic>& logLikes)
//{
//  MatrixXu inds = MatrixXu::Zero(n,K_);
//  logLikes = Matrix<T,Dynamic,Dynamic>::Ones(n,K_);
//  logLikes *= -99999.0;
//  
//#pragma omp parallel for 
//  for (int32_t k=0; k<int32_t(K_); ++k)
//  {
//    for (uint32_t i=0; i<z_.size(); ++i)
//      if(z_(i) == k)
//      {
//        T logLike = thetas_[z_[i]]->logLikelihood(x_[i]);
//        for (uint32_t j=0; j<n; ++j)
//          if(logLikes(j,k) < logLike)
//          {
//            for(uint32_t l=n-1; l>j; --l)
//            {
//              logLikes(l,k) = logLikes(l-1,k);
//              inds(l,k) = inds(l-1,k);
//            }
//            logLikes(j,k) = logLike;
//            inds(j,k) = i;
////            cout<<"after update "<<logLike<<endl;
////            Matrix<T,Dynamic,Dynamic> out(n,K_*2);
////            out<<logLikes.cast<T>(),inds.cast<T>();
////            cout<<out<<endl;
//            break;
//          }
//      }
//  } 
//  cout<<"::mostLikelyInds: logLikes"<<endl;
//  cout<<logLikes<<endl;
//  cout<<"::mostLikelyInds: inds"<<endl;
//  cout<<inds<<endl;
//  return inds;
//};


template<typename T>
void DirMultiNaiveBayes<T>::inferAll(uint32_t nIter, bool verbose)
{ 
  if(verbose){
  	cout<<"[DirMultiNaiveBayes::inferALL] ------ inferingALL (nIter=" << nIter << ") ------"<<endl;
  	cout <<"initial labels:"<< endl;
  	cout<<this->labels().transpose()<<endl;
  }
  for(uint32_t t=0; t<nIter; ++t)
  {
    this->sampleLabels();
    this->sampleParameters();
    if(verbose){
		cout << "[" << std::setw(3)<< std::setfill('0')  << t <<"] label: " 
    	<< this->labels().transpose()
      	<< " [joint= " << std::setw(6) << this->logJoint(false) << "]"<< endl;
    }
  }

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
void DirMultiNaiveBayes<T>::dump_clean() {
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
	//for type XX (NIWSphereFull)
	//?
	
	//prints headers
	cout << M_ << endl 
		 << K_ << endl
		 << Nd_ << endl;

	//print dim
	for(uint m=0; m<M_; ++m) {
		cout << x_[m].front().rows() << " "; 
	}
	cout << endl;
	//print type
	for(uint m=0; m<M_; ++m) {
		cout << thetas_[m].front()->getBaseMeasureType() << " "; 
	}
	cout << endl;

	//print labels
	cout << this->labels().transpose() << endl; 

	//print mixture parameters
	cout << this->dir_.alpha_.transpose() << endl;
	cout << this->pi_.pdf_.transpose() << endl;
	cout << this->pdfs_.transpose() << endl; 

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
					cout << prior.nu_				 << endl << 
						    prior.kappa_			 << endl <<
						    prior.theta_.transpose() << endl;
					cout << prior.Delta_<<endl;

					//printing posterior
					Normal<T> norm = theta_iter->get()->normal_; 
					cout << norm.mu_.transpose() << endl;
					cout << norm.Sigma() << endl;
			} else if(type==NIW_SPHERE_FULL) {
					assert(false);
			} else {
					std::cerr << "[DirMultiNaiveBayes::dump_clean] error saving...returning" << endl;
					return;
			}
		}
	}

}

template <typename T>
void DirMultiNaiveBayes<T>::dump_clean(std::ofstream &out){
	streambuf *coutbuf = std::cout.rdbuf(); //save old cout buffer
	cout.rdbuf(out.rdbuf()); //redirect std::cout to fout1 buffer
	this->dump_clean(); //write using cout to the specified buffer
	std::cout.rdbuf(coutbuf); //reset to standard output again
}

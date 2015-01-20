/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>, Randi Cabezas <rcabezas@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.                
 */

#include "niw.hpp"

// ---------------------------------------------------------------------------

template<typename T>
NIW<T>::NIW(const Matrix<T,Dynamic,Dynamic>& Delta, 
  const Matrix<T,Dynamic,Dynamic>& theta, T nu,  T kappa, boost::mt19937 *pRndGen)
: Distribution<T>(pRndGen), Delta_(Delta), theta_(theta), nu_(nu), kappa_(kappa), 
  D_(theta.size()), 
  scatter_(Matrix<T,Dynamic,Dynamic>::Zero(Delta_.rows(),Delta_.cols())),
  mean_(Matrix<T,Dynamic,1>::Zero(Delta_.rows())), count_(0)
{
  assert(Delta_.rows()==theta_.size()); 
  assert(Delta_.cols()==theta_.size());
};

template<typename T>
NIW<T>::NIW(const NIW& niw)
: Distribution<T>(niw.pRndGen_), Delta_(niw.Delta_), theta_(niw.theta_), 
  nu_(niw.nu_), kappa_(niw.kappa_), D_(niw.D_), scatter_(niw.scatter()),
  mean_(niw.mean()), count_(niw.count())
{
  assert(Delta_.rows()==theta_.size()); 
  assert(Delta_.cols()==theta_.size());
};

template<typename T>
NIW<T>::~NIW()
{};

template<typename T>
T NIW<T>::logPdfUnderPriorMarginalizedMerged(const NIW<T>& other) const
{
  Matrix<T,Dynamic,Dynamic> scatterM;
  Matrix<T,Dynamic,1> muM;
  T countM = 0;
  computeMergedSS( *this, other, scatterM, muM, countM);
  return logLikelihoodMarginalized(scatterM,muM,countM);
}

template<typename T>
NIW<T> NIW<T>::posterior(const Matrix<T,Dynamic,Dynamic>& x, const VectorXu& z, 
    uint32_t k)
{
  getSufficientStatistics(x,z,k);
  return posterior();
};

template<typename T>
NIW<T> NIW<T>::posterior(const vector<Matrix<T,Dynamic,Dynamic> > &x, const VectorXu& z, 
    uint32_t k) 
{
  getSufficientStatistics(x,z,k);
  return posterior();
}; 

template<typename T>
NIW<T> NIW<T>::posterior() const 
{
//  cout<<Delta_<<endl<<" + "<<endl<<scatter_<<endl<<" + "<<endl
//    <<((kappa_*count_)/(kappa_+count_))
//      *(mean_-theta_)*(mean_-theta_).transpose()<<endl;
//  cout<<count_<<endl;
//  cout<<(kappa_*theta_+ count_*mean_)/(kappa_+count_)<<endl;
  return NIW<T>(
    Delta_+scatter_ + ((kappa_*count_)/(kappa_+count_))
      *(mean_-theta_)*(mean_-theta_).transpose(), 
    (kappa_*theta_+ count_*mean_)/(kappa_+count_),
    nu_+count_,
    kappa_+count_,
    this->pRndGen_);
};

template<typename T>
void NIW<T>::resetSufficientStatistics()
{
  scatter_.setZero(D_,D_);
  mean_.setZero(D_);
  count_ = 0.;
};

template<typename T>
void NIW<T>::getSufficientStatistics(const Matrix<T,Dynamic,Dynamic>& x, 
  const VectorXu& z, uint32_t k)
{
  this->resetSufficientStatistics();
  // TODO: be carefull here when parallelizing since all are writing to the same 
  // location in memory
#pragma omp parallel for
  for (int32_t i=0; i<z.size(); ++i)
  {
    if(z(i) == k)
    {      
      Matrix<T,Dynamic,Dynamic> outer = x.col(i) * x.col(i).transpose();
#pragma omp critical
      {
        mean_ += x.col(i);
        scatter_ += outer;
        count_++;
      }
    }
  }
  if (count_ > 0.)
  {
    mean_ /= count_;
    scatter_ -= (mean_*mean_.transpose())*count_;
  }
#ifndef NDEBUG
  cout<<" -- updating ss "<<count_<<endl;
  cout<<"mean="<<mean_.transpose()<<endl;
  cout<<"scatter="<<endl<<scatter_<<endl;
  posterior().print();
#endif
};

template<typename T>
void NIW<T>::getSufficientStatistics(const vector<Matrix<T,Dynamic,Dynamic> >&x, 
    const VectorXu& z, uint32_t k) 
{
	scatter_.setZero(D_,D_);
	mean_.setZero(D_);
	count_ = 0.;
	// TODO: be carefull here when parallelizing since all are writing to the same 
	// location in memory
	for (uint32_t i=0; i<z.size(); ++i) //loop over docs
	{
		if(z(i) == k)
		{      
			#pragma omp parallel for
			for (int32_t j=0; j<x[i].cols(); ++j) //loop over words
			{
				Matrix<T,Dynamic,Dynamic> outer = x[i].col(j) * x[i].col(j).transpose();
				#pragma omp critical
				{
					mean_ += x[i].col(j);
					scatter_ += outer;
					count_++;
				}
			}
		}
	}
	if (count_ > 0.)
	{
		mean_ /= count_;
		scatter_ -= (mean_*mean_.transpose())*count_;
	}
	#ifndef NDEBUG
		cout<<" -- updating ss "<<count_<<endl;
		cout<<"mean="<<mean_.transpose()<<endl;
		cout<<"scatter="<<endl<<scatter_<<endl;
		posterior().print();
	#endif
};

template<typename T>
T NIW<T>::logProb(const Matrix<T,Dynamic,Dynamic>& x_i) const
{

//  cout<<"    Delta_post=\n"<<Delta_<<endl;
//  cout<<"    x_i="<<x_i.transpose()<<endl;
  // using multivariate student-t distribution
  Matrix<T,Dynamic,Dynamic> scaledDelta = Delta_*(kappa_+1.)/kappa_;                       

//  cout<<"x="<<x_i.transpose()<<" theta="<<theta_.transpose()<<" -> ";
//  cout<<((x_i-theta_).transpose()*scaledDelta.lu().solve(x_i-theta_)).sum();
//  cout<<" "<<((x_i-theta_).transpose()*scaledDelta.lu().solve(x_i-theta_));

  T logProb = boost::math::lgamma(0.5*(nu_+1));
  logProb -= (0.5*(nu_+1.))
    *log(1.+((x_i-theta_).transpose()*scaledDelta.lu().solve(x_i-theta_)).sum());
  logProb -= 0.5*D_*log(PI);
  logProb -= boost::math::lgamma(0.5*(nu_-D_+1.));
  //logProb -= D_*(log(kappa_+1.)-log(kappa_));
  logProb -= 0.5*((scaledDelta.eigenvalues()).array().log().sum()).real();
  // DONE - seems to be equivalent
  //cout<<"det="<<log(scaledDelta.determinant())<<" "<<((scaledDelta.eigenvalues()).array().log().sum()).real()<<endl;
  //cout<<logProb<<endl;

//  vec eigval = eig_sym(Sig);                                                    
//  T logDetSig = sum(log(eigval)); // this computaiton of the determinant works for very small matrices too

//  cout<<" => logPdf="<<logProb;

  //Normal momentMatched(theta_,((kappa_+1.)/(kappa_*(nu_-D_-1.)))* Delta_,this->pRndGen_);
//  cout<< " momentMatched="<<momentMatched.logPdf(x_i)<<endl;

  return logProb;
};

template<typename T>
T NIW<T>::logPosteriorProb(const Matrix<T,Dynamic,Dynamic>& x, VectorXu& z, uint32_t k, uint32_t i)
{
  uint32_t z_i = z[i];
  z[i] = k+1; // so that we definitely not use x_i in posterior computation 
  // (since the posterior is only computed form x_{z==k})
  NIW posterior = this->posterior(x,z,k);
  z[i] = z_i; // reset to old value
  return posterior.logProb(x.col(i));
};

template<typename T>
NIW<T>* NIW<T>::copy()
{
  NIW<T>* newNiw =  new NIW<T>(Delta_,theta_,nu_,kappa_,this->pRndGen_);
  newNiw->scatter() = scatter_;
  newNiw->mean() = mean_;
  newNiw->count() = count_;
  return newNiw;
}

template<typename T>
Normal<T> NIW<T>::sample()
{
  // do the cholesky decomposition
  //Eigen::Map<Eigen::Matrix<T,Dynamic,Dynamic>> sigma(Delta, D_, D_);
// TODO: do I have to divide by nu or not?
  LLT<Matrix<T,Dynamic,Dynamic> > myLlt(Delta_/nu_);
  Matrix<T,Dynamic,Dynamic> chol = myLlt.matrixL();

  Matrix<T,Dynamic,Dynamic> Sigma(D_,D_);

  for (uint32_t d=0; d<D_; d++)
  {
    // TODO: could move this into constructor
    boost::random::chi_squared_distribution<> chiSq_(nu_-d);
    Sigma(d,d) = sqrt(chiSq_(*this->pRndGen_)); //sqrt(gsl_ran_chisq(r, nu-d));
    for (uint32_t d2=d+1; d2<D_; d2++)
    {
      Sigma.data()[d2+d*D_] = 0;
      Sigma.data()[d+d2*D_] = gauss_(*this->pRndGen_); //gsl_ran_gaussian(r, 1);
    }
  }
//  cout<<nu_<<endl;
//  cout<<Sigma<<endl;
  // diag * cov^-1
  Sigma = sqrt(nu_)*chol*Sigma.inverse();
  Sigma = Sigma*Sigma.transpose();

  // temp now contains the covariance
  myLlt.compute(Sigma);
  chol = myLlt.matrixL();

  // populate the mean
  Matrix<T,Dynamic,1> mean(D_);
  for (uint32_t d=0; d<D_; d++)
    mean[d] = gauss_(*this->pRndGen_); //gsl_ran_gaussian(r,1);
  //Eigen::Map<Eigen::Matrix<T,Dynamic,1>> emean(_param.mean, D_);
  mean = chol*mean / sqrt(kappa_) + theta_;

  //_param.logDetCov = log(det(_param.cov,D_));
  return Normal<T>(mean,Sigma,this->pRndGen_);
};


template<typename T>
T NIW<T>::logPdf(const Normal<T>& normal) const
{
  T logPdf = 0.5*nu_*((Delta_.eigenvalues()).array().log().sum()).real();
  logPdf += 0.5*D_*log(kappa_);
  logPdf -= 0.5*D_*log(2.0*PI);
  logPdf -= 0.5*D_*nu_*log(2.0);
  logPdf -= lgamma_mult(nu_*0.5,D_);
  logPdf -= (0.5*(nu_+D_)+1.)*normal.logDetSigma();
  logPdf -= 0.5*(Delta_*normal.Sigma().inverse()).trace();
  T temp = (normal.mu_ - theta_).transpose()*
	normal.SigmaLDLT().solve(normal.mu_ - theta_);
  logPdf -= 0.5*kappa_*temp;
  //logPdf -= 0.5*kappa_*(normal.mu_ - theta_).transpose()*
  // normal.SigmaLDLT().solve(normal.mu_ - theta_);
  return logPdf;
};

template<typename T>
T NIW<T>::logLikelihoodMarginalized(const Matrix<T,Dynamic,Dynamic>& Scatter, 
      const Matrix<T,Dynamic,1>& mean, T count) const
{
  Matrix<T,Dynamic,Dynamic> DeltaPost = Delta_+Scatter + 
    ((kappa_*count)/(kappa_+count)) *(mean-theta_)*(mean-theta_).transpose();
  
//  cout<<endl;
//  cout<<"mean="<<mean_.transpose()<<" count="<<count_<<endl;
//  cout<<"scatter="<<endl<<scatter_<<endl;
//  cout<<"deltaPosterior="<<endl<<DeltaPost<<endl;
//  cout<<"delta="<<endl<<Delta_<<endl;
//  cout<<lgamma_mult((nu_+count_)*0.5,D_) - lgamma_mult(nu_*0.5,D_)<<endl;

  T logPdf = -0.5*count*D_*LOG_PI;
  logPdf += lgamma_mult((nu_+count)*0.5,D_); 
  logPdf -= lgamma_mult(nu_*0.5,D_); 
  logPdf += 0.5*nu_*((Delta_.eigenvalues()).array().log().sum()).real();
  logPdf -= 0.5*(nu_+count)
            *(((DeltaPost).eigenvalues()).array().log().sum()).real();
  logPdf += 0.5*D_*(log(kappa_) -log(kappa_+count));
//  cout<<"::logPdfMarginalized = "<<logPdf<<endl;
  return logPdf;
}

template<typename T>
T NIW<T>::logPdfMarginalized() const
{
    return logLikelihoodMarginalized(scatter_,mean_,count_);
};

template<typename T>
NIW<T>* NIW<T>::merge(const NIW<T>& other)
{
  NIW<T>* merged = this->copy();
  merged->fromMerge(*this,other);

//  merged->mean() = (merged->count()*merged->mean() + other.count()*other.mean())
//    /(merged->count() + other.count()); 
////  cout<<merged->mean().transpose()<<endl;
//  merged->count() += other.count(); 
//
//  merged->scatter() =  
//   (merged->scatter() + merged->count()*merged->mean()*merged->mean().transpose()) 
//   +(other.scatter() + other.count()*other.mean()*other.mean().transpose()) 
//   - merged->count()*merged->mean()*merged->mean().transpose();
////  cout<<merged->mean().transpose()<<endl
////    << other.mean().transpose()<<endl;
////  cout<<merged->count()<<" "<<other.count()<<endl;
  return merged;
};


template<typename T>
void NIW<T>::computeMergedSS( const NIW<T>& niwA, 
    const NIW<T>& niwB, Matrix<T,Dynamic,Dynamic>& scatterM, 
    Matrix<T,Dynamic,1>& muM, T& countM) const
{
  countM = niwA.count() + niwB.count(); 
  muM = (niwA.count()*niwA.mean() + niwB.count()*niwB.mean())/countM; 
  scatterM = 
     (niwA.scatter() + niwA.count()*niwA.mean()*niwA.mean().transpose()) 
    +(niwB.scatter() + niwB.count()*niwB.mean()*niwB.mean().transpose()) 
    - countM * muM*muM.transpose();
};

template<typename T>
void NIW<T>::fromMerge(const NIW<T>& niwA, const NIW<T>& niwB)
{
//  this->mean() = (niwA.count()*niwA.mean() + niwB.count()*niwB.mean())
//    /(niwA.count() + niwB.count()); 
//  this->count() = niwA.count() + niwB.count(); 
//  this->scatter() = 
//     (niwA.scatter() + niwA.count()*niwA.mean()*niwA.mean().transpose()) 
//    +(niwB.scatter() + niwB.count()*niwB.mean()*niwB.mean().transpose()) 
//    - this->count()*this->mean()*this->mean().transpose();

    computeMergedSS(niwA, niwB, scatter_, mean_, count_);
//  cout<<" -- from merge "<<count_<<endl;
//  cout<<"mean="<<mean_.transpose()<<endl;
//  cout<<"scatter="<<endl<<scatter_<<endl;
//  print();
};

template<typename T>
void NIW<T>::print() const
{
  cout<<"nu="<<nu_<<" kappa="<<kappa_<<"\t theta="<<theta_.transpose()<<endl;
  cout<<Delta_<<endl;
};

template class NIW<double>;
template class NIW<float>;

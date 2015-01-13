/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */
#include "iw.hpp"


template<typename T>
: Distribution<T>(pRndGen), Delta_(Delta), nu_(nu), D_(Delta.cols()),
  scatter_(Matrix<T,Dynamic,Dynamic>::Zero(Delta_.rows(),Delta_.cols())),
  mean_(Matrix<T,Dynamic,1>::Zero(Delta_.rows())), count_(0)
{
  assert(Delta_.cols() == Delta_.rows());
};


template<typename T>
IW<T>::IW(const Matrix<T,Dynamic,Dynamic>& Delta, T nu, const Matrix<T,Dynamic,Dynamic>& scatter, 
	 const Matrix<T,Dynamic,1>& mean, T counts, boost::mt19937 *pRndGen) 
: Distribution<T>(pRndGen), Delta_(Delta), nu_(nu), D_(Delta.cols()),
  scatter_(scatter), mean_(mean), count_(counts)
{
  assert(Delta_.cols() == Delta_.rows());
};

template<typename T>
IW<T>::IW(const IW& iw)
: Distribution<T>(iw.pRndGen_), Delta_(iw.Delta_), nu_(iw.nu_), D_(iw.D_),
  scatter_(iw.scatter()), mean_(iw.mean()), count_(iw.count())
{};

template<typename T>
IW<T>::~IW()
{};

template<typename T>
void IW<T>::resetSufficientStatistics()
{
  scatter_.setZero(D_,D_);
  mean_.setZero(D_);
  count_ = 0.;
};

template<typename T>
IW<T> IW<T>::posterior(const Matrix<T,Dynamic,Dynamic>& x, const VectorXu& z,
    uint32_t k, uint32_t zDivider)
{
  resetSufficientStatistics();
//  cout<<"posterior for indicators "<<k<<endl;
  // TODO: be carefull here when parallelizing since all are writing to the same 
  // location in memory
#pragma omp parallel for
  for (int32_t i=0; i<z.size(); ++i)
  {
    if(z(i)/zDivider == k)
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
    cout<<"    scatter=\n"<<scatter_<<endl;
    cout<<"    mean =  "<<mean_.transpose()<<endl;
    cout<<"    count = "<<count_<<endl;
  #endif
  return posterior();
};

template<typename T>
IW<T> IW<T>::posterior() const
{
  assert(scatter_.cols() == Delta_.cols());
  assert(scatter_.rows() == Delta_.rows());
//  #ifndef NDEBUG
//    cout<<"    scatter=\n"<<scatter_<<endl;
//    cout<<"    Delta=\n"<<Delta_<<endl;
//    cout<<"    mean =  "<<mean_.transpose()<<endl;
//    cout<<"    count = "<<count_<<endl;
//    cout<<"    mean*mean.T = \n"<<(mean_*mean_.transpose())<<endl;
//  #endif
  //TODO this one ignores the sample mean
//  return IW<T>(Delta_+scatter_, nu_+count_, this->pRndGen_);
  //TODO this one assumes zero mean Gaussian
  return IW<T>(Delta_+scatter_+(mean_*mean_.transpose())*count_, 
      nu_+count_, this->pRndGen_);
};

template<typename T>
Matrix<T,Dynamic,Dynamic> IW<T>::mode() const
{
  return Delta_/(nu_+D_+1);
};

template<typename T>
IW<T>* IW<T>::copy()
{
  return new IW<T>(Delta_,nu_,this->pRndGen_);
}

template<typename T>
Matrix<T,Dynamic,Dynamic> IW<T>::sample()
{
  // do the cholesky decomposition
// TODO: do I have to divide by nu or not?
  LLT<Matrix<T,Dynamic,Dynamic> > myLlt(Delta_/nu_);
  Matrix<T,Dynamic,Dynamic> chol = myLlt.matrixL();

  Matrix<T,Dynamic,Dynamic> Sigma(D_,D_);

  for (uint32_t d=0; d<D_; d++)
  {
    // TODO: could move this into constructor
    boost::random::chi_squared_distribution<> chiSq_(nu_-d);
    Sigma.data()[d+d*D_] = sqrt(chiSq_(*this->pRndGen_)); //sqrt(gsl_ran_chisq(r, nu-d));
    for (uint32_t d2=d+1; d2<D_; d2++)
    {
      Sigma.data()[d2+d*D_] = 0;
      Sigma.data()[d+d2*D_] = gauss_(*this->pRndGen_); //gsl_ran_gaussian(r, 1);
    }
  }
//  cout<<Delta_<<endl;
//  cout<<Sigma<<endl;
  // diag * cov^-1
  Sigma = sqrt(nu_)*chol*Sigma.inverse();
  Sigma = Sigma*Sigma.transpose();
  return Sigma;
};

//template<typename T>
//Normal<T> IW<T>::sample()
//{
//  return Normal<T>(this->sample(),this->pRndGen_);
//};

template<typename T>
T IW<T>::logPdf(const Normal<T>& normal) const
{
  return logPdf(normal.Sigma());
};

template<typename T>
T IW<T>::logPdf(const Matrix<T,Dynamic,Dynamic>& Sigma) const
{
  assert(nu_ + 1. -D_ > 0);

  T logPdf = 0.5*nu_*((Delta_.eigenvalues()).array().log().sum()).real();
  logPdf -= 0.5*(nu_+D_+1.)*((Sigma.eigenvalues()).array().log().sum()).real();
  logPdf -= 0.5*D_*nu_*LOG_2;
  logPdf -= lgamma_mult(nu_*0.5,D_);
  logPdf -= 0.5*(Delta_*Sigma.inverse()).trace();
  return logPdf;
};

template<typename T>
T IW<T>::logLikelihoodMarginalized() const // log pdf of SS under NIW prior
{
    return logLikelihoodMarginalized(scatter_,count_);
}

template<typename T>
T IW<T>::logLikelihoodMarginalized(const Matrix<T,Dynamic,Dynamic>& scatter, 
  T count) const
{
  assert(nu_ + 1. - D_ > 0);
  assert(count + nu_ + 1. -D_ > 0);

  T logPdf = 0.5*nu_*((Delta_.eigenvalues()).array().log().sum()).real();
  logPdf -= 0.5*D_*count*LOG_PI;
  logPdf -= 0.5*(nu_+count)
    *(((Delta_+scatter).eigenvalues()).array().log().sum()).real();
  logPdf += lgamma_mult((count+nu_)*0.5,D_);
  logPdf -= lgamma_mult(nu_*0.5,D_);
  return logPdf;
};

// ---------------------------------------------------------------------------
template<typename T>
IW_spherical<T>::IW_spherical( T delta, T nu, uint32_t D,
    boost::mt19937 *pRndGen)
  : Distribution<T>(pRndGen), delta_(delta), nu_(nu), D_(D)
{};

template<typename T>
IW_spherical<T>::IW_spherical(const IW_spherical& iw)
: Distribution<T>(iw.pRndGen_), delta_(iw.delta_), nu_(iw.nu_), D_(iw.D_)
{};

template<typename T>
IW_spherical<T>::~IW_spherical()
{};

template<typename T>
IW_spherical<T> IW_spherical<T>::posterior(const Matrix<T,Dynamic,Dynamic>& x, 
    const VectorXu& z, uint32_t k, uint32_t zDivider)
{
  T s =0.0;
  Matrix<T,Dynamic,1> mean(D_); 
  mean.setZero(D_);
  T N=0.;
  for (uint32_t i=0; i<z.size(); ++i)
    if(z[i]/zDivider == int32_t(k))
    {      
      mean += x.col(i);
      N++;
    }
  if (N > 0.)
  {
    mean /= N;
    for (uint32_t i=0; i<z.size(); ++i)
      if(z[i] == int32_t(k))
      {   
        // TODO: make sure this is not just a hack
        T s_avg =0.0;
        for(uint32_t d=0; d<D_; ++d)
          s_avg += (x(d,i)-mean(d))*(x(d,i)-mean(d));
        s += s_avg/static_cast<T>(D_);
      }
  }
//  cout<<"    S=\n"<<S<<endl;
//  cout<<"    mean="<<mean.transpose()<<endl;
  return IW_spherical<T>(delta_+s, nu_+N, D_, this->pRndGen_);
};


template<typename T>
IW_spherical<T>* IW_spherical<T>::copy()
{
  return new IW_spherical<T>(delta_,nu_,D_,this->pRndGen_);
}

//Matrix<T,Dynamic,Dynamic> IW_spherical<T>::sample()
//{
//  // do the cholesky decomposition
//// TODO: do I have to divide by nu or not?
//  LLT<Matrix<T,Dynamic,Dynamic>> myLlt(Delta_/nu_);
//  Matrix<T,Dynamic,Dynamic> chol = myLlt.matrixL();
//
//  Matrix<T,Dynamic,Dynamic> Sigma(D_,D_);
//
//  for (uint32_t d=0; d<D_; d++)
//  {
//    // TODO: could move this into constructor
//    boost::random::chi_squared_distribution<> chiSq_(nu_-d);
//    Sigma.data()[d+d*D_] = sqrt(chiSq_(*this->pRndGen_)); //sqrt(gsl_ran_chisq(r, nu-d));
//    for (uint32_t d2=d+1; d2<D_; d2++)
//    {
//      Sigma.data()[d2+d*D_] = 0;
//      Sigma.data()[d+d2*D_] = gauss_(*this->pRndGen_); //gsl_ran_gaussian(r, 1);
//    }
//  }
////  cout<<Delta_<<endl;
////  cout<<Sigma<<endl;
//  // diag * cov^-1
//  Sigma = sqrt(nu_)*chol*Sigma.inverse();
//  Sigma = Sigma*Sigma.transpose();
//  return Sigma;
//};
//
//
//T IW_spherical<T>::logPdf(const Normal& normal)
//{
//  return logPdf(normal.Sigma_);
//};
//
//T IW_spherical<T>::logPdf(const Matrix<T,Dynamic,Dynamic>& Sigma)
//{
//  T logPdf = 0.5*nu_*((Delta_.eigenvalues()).array().log().sum()).real();
//  logPdf -= 0.5*(nu_+D_+1.)
//    *((Sigma.eigenvalues()).array().log().sum()).real();
//  logPdf -= 0.5*D_*nu_*LOG_2;
//  logPdf -= lgamma_mult(nu_*0.5,D_);
//  logPdf -= 0.5*(Delta_*Sigma.inverse()).trace();
//  return logPdf;
//};



///* prior for diagonal covariances (i.e. \Sigma = diag(\sigmas)) */
//class IW_diag : public Distribution
//{
//public:
//  Matrix<T,Dynamic,1> deltas_;
//  T nu_;
//  uint32_t D_;
//
//  IW_diag(const Matrix<T,Dynamic,1>& deltas, T nu, boost::mt19937 *this->pRndGen);
//  IW_diag(const IW& iw);
//  ~IW_diag();
//
//  IW_diag* copy();
//
//  IW_diag posterior(const Matrix<T,Dynamic,Dynamic>& x, const VectorXu& z, uint32_t k);
//
//  T sample();
//
//  T logPdf(const Matrix<T,Dynamic,1>& sigmas);
////  T logPdf(const Normal& normal);
//
//private:
//
//  boost::random::normal_distribution<> gauss_;
//
//};

template class IW<float>;
template class IW<double>;

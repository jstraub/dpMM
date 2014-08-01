#include "dir.hpp"

// ----------------------------------------------------------------------------
template<typename T>
Dir<T>::Dir(const Matrix<T,Dynamic,1>& alpha, boost::mt19937* pRndGen) 
  : Distribution<T>(pRndGen), K_(alpha.size()), alpha_(alpha)
{
  for(uint32_t k=0; k<K_; ++k)
    gammas_.push_back(boost::random::gamma_distribution<>(alpha_(k)));
};

template<typename T>
Dir<T>::Dir(const Dir& other)
  : Distribution<T>(other.pRndGen_), K_(other.K_), alpha_(other.alpha_)
{
  for(uint32_t k=0; k<K_; ++k)
    gammas_.push_back(boost::random::gamma_distribution<>(alpha_(k)));
};

template<typename T>
Dir<T>::~Dir()
{};

template<typename T>
Matrix<T,Dynamic,1> Dir<T>::samplePdf()
{
  // sampling from Dir via gamma distribution 
  // http://en.wikipedia.org/wiki/Dirichlet_distribution
  Matrix<T,Dynamic,1> pi(K_);
  for (uint32_t k=0; k<K_; ++k)
    pi(k) = gammas_[k](*this->pRndGen_);
  return pi/pi.sum();
};

template<typename T>
Cat<T> Dir<T>::sample()
{
  return Cat<T>(this->samplePdf(),this->pRndGen_);
};

template<typename T>
Dir<T> Dir<T>::posterior(const VectorXu& z)
{
//  cout << "posterior alpha: "<<(alpha_+counts(z,K_)).transpose()<<endl;
  return Dir<T>(alpha_+counts(z,K_).cast<T>(),this->pRndGen_);   
};

template<typename T>
Dir<T> Dir<T>::posteriorFromCounts(const Matrix<T,Dynamic,1>& counts)
{
  return Dir<T>(alpha_+counts,this->pRndGen_);
};

template<typename T>
Dir<T> Dir<T>::posteriorFromCounts(const VectorXu& counts)
{
  return Dir<T>(alpha_+counts.cast<T>(),this->pRndGen_);
};

template<typename T>
T Dir<T>::logPdf(const Cat<T>& cat)
{
  //gammaln(np.sum(s.alpha)) - np.sum(gammaln(s.alpha))
  //+ np.sum((s.alpha-1)*np.log(pi)) 
  T logPdf = boost::math::lgamma(alpha_.sum());
  for(uint32_t k=0; k<K_; ++k)
    logPdf += -boost::math::lgamma(alpha_[k]) + (alpha_[k]-1.)*log(cat.pdf()[k]);
  return logPdf;
};

template class Dir<double>;
template class Dir<float>;

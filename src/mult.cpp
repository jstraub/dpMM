
#include "mult.hpp"

// ---------------------------------------------------------------------------

template<typename T>
Mult<T>::Mult(const Matrix<T,Dynamic,1>& pdf, boost::mt19937 *pRndGen)
: Distribution<T>(pRndGen), K_(pdf.size()), pdf_(pdf)
{
};

template<typename T>
Mult<T>::Mult(const VectorXu& z, boost::mt19937 *pRndGen)
: Distribution<T>(pRndGen), K_(z.maxCoeff()+1)
{
  pdf_ = counts<T,uint32_t>(z,K_);
  pdf_ /= pdf_.sum();
};

template<typename T>
Mult<T>::Mult(const Mult<T>& other)
  : Distribution<T>(other.pRndGen_), K_(other.K_), pdf_(other.pdf_)
{};

template<typename T>
Mult<T>::~Mult()
{};

template<typename T>
T Mult<T>::logPdf(const Matrix<T,Dynamic,1>& x) const 
{
  // log(Gamma(n+1)) = n!
  T logPdf = lgamma(x.sum()+1);
  for(uint32_t k=0; k<K_; ++k)
    logPdf += - lgamma(x(k)+1) + x(k)*log(pdf_(k));
  // TODO maybe buffer logPdf_
  return logPdf;
};


template<typename T>
uint32_t Mult<T>::sample()
{
  assert(false);// TODO
//  T r=unif_(*this->pRndGen_);
  //cout<<cdf_.transpose()<<" -> "<<r<<endl;
//  for (uint32_t k=0; k<K_; ++k)
//    if (r<cdf_(k)){return k-1;}
  return K_-1;
};

template<typename T>
void Mult<T>::sample(VectorXu& z)
{
  assert(false);// TODO
  for(uint32_t i=0; i<z.size(); ++i)
    z(i) = this->sample();
};

template<typename T>
void Mult<T>::print() const
{
  cout<<"pi="<<pdf_.transpose()<<endl;
}

template class Mult<double>;
template class Mult<float>;

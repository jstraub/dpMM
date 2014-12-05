#pragma once

#include <Eigen/Dense>

#include "basemeasure.hpp"
#include "niw.hpp"

/*
 * NIW base measure; integrates over Normal distribution parameters
 */
template<typename T>
class NiwMarginalized : public BaseMeasure<T>
{
public:
  NIW<T> niw0_; 
  NIW<T> niw_; 

  NiwMarginalized(const NIW<T>& niw);
  ~NiwMarginalized();

  virtual baseMeasureType getBaseMeasureType() const {return(NIW_MARGINALIZED); }

  virtual BaseMeasure<T>* copy();

  T logLikelihood(const Matrix<T,Dynamic,1>& x) const;

  void posterior(const Matrix<T,Dynamic,Dynamic>& x, const VectorXu& z, 
      uint32_t k);

  T logPdfUnderPrior() const;

  void print() const ;
  virtual uint32_t getDim() const {return(uint32_t(niw_.D_));}; 
};

/*
 * NIW base measure; samples Normal distribution parameters
 */
template<typename T>
class NiwSampled : public BaseMeasure<T>
{
public:
  NIW<T> niw0_; 
  Normal<T> normal_;

  NiwSampled(const NIW<T>& niw);
  NiwSampled(const NIW<T>& niw, const Normal<T> &normal);
  ~NiwSampled();

  virtual baseMeasureType getBaseMeasureType() const {return(NIW_SAMPLED); }
  
  virtual BaseMeasure<T>* copy();
  virtual NiwSampled<T>* copyNative();

  T logLikelihood(const Matrix<T,Dynamic,1>& x) const;
  T logLikelihood(const Matrix<T,Dynamic,Dynamic>& x, uint32_t i) const 
    {return logLikelihood(x.col(i));};
  void posterior(const Matrix<T,Dynamic,Dynamic>& x, const VectorXu& z, 
    uint32_t k);
  void posterior(const vector<Matrix<T,Dynamic,Dynamic> >&x, const VectorXu& z, 
    uint32_t k);
  void sample();

  T logPdfUnderPrior() const;
  T logPdfUnderPriorMarginalized() const;
  T logPdfUnderPriorMarginalizedMerged(const shared_ptr<NiwSampled<T> >& other) const;

  virtual NiwSampled<T>* merge(const NiwSampled<T>& other);
  void fromMerge(const NiwSampled<T>& niwA, const NiwSampled<T>& niwB);

  void print() const;
  virtual uint32_t getDim() const {return(uint32_t(normal_.D_));}; 

  const Matrix<T,Dynamic,Dynamic>& scatter() const {return niw0_.scatter();};
  const Matrix<T,Dynamic,1>& mean() const {return niw0_.mean();};
  T count() const {return niw0_.count();};
//  T& count() {return niw0_.count_;};
  const Matrix<T,Dynamic,1>& getMean() const {return normal_.mu_;};

  const Matrix<T,Dynamic,Dynamic>& Sigma() const {return normal_.Sigma();};
private:

};

typedef NiwSampled<double> NiwSampledd;
typedef NiwSampled<float> NiwSampledf;

// ---------------------------------------------------------------------------


template<typename T>
NiwSampled<T>::NiwSampled(const NIW<T>& niw)
  : niw0_(niw), normal_(niw0_.sample())
{};

template<typename T>
NiwSampled<T>::NiwSampled(const NIW<T>& niw, const Normal<T> &normal)
 : niw0_(niw), normal_(normal)
{}; 

template<typename T>
NiwSampled<T>::~NiwSampled()
{};

template<typename T>
BaseMeasure<T>* NiwSampled<T>::copy()
{
  NiwSampled<T>* niwSampled = new NiwSampled<T>(niw0_);
  niwSampled->normal_ = normal_;
  return niwSampled;
};

template<typename T>
NiwSampled<T>* NiwSampled<T>::copyNative()
{
  NiwSampled<T>* niwSampled = new NiwSampled<T>(niw0_);
  niwSampled->normal_ = normal_;
  return niwSampled;
};

template<typename T>
T NiwSampled<T>::logLikelihood(const Matrix<T,Dynamic,1>& x) const
{
//  normal_.print();
  T logLike = normal_.logPdf(x);
//  cout<<x.transpose()<<" -> " <<logLike<<endl;
//  cout<<x.transpose()<<" -> " <<normal_.logPdfSlower(x)<<endl;
  return logLike;
};


template<typename T>
void NiwSampled<T>::posterior(const Matrix<T,Dynamic,Dynamic>& x, 
    const VectorXu& z, uint32_t k)
{
  normal_ = niw0_.posterior(x,z,k).sample();
};

template<typename T>
void NiwSampled<T>::posterior(const vector<Matrix<T,Dynamic,Dynamic> >&x, 
	const VectorXu& z, uint32_t k) 
{
	normal_ = niw0_.posterior(x,z,k).sample();
}

template<typename T>
T NiwSampled<T>::logPdfUnderPrior() const
{
  return niw0_.logPdf(normal_);
};

template<typename T>
T NiwSampled<T>::logPdfUnderPriorMarginalized() const
{
  // evaluates log pdf of sufficient statistics stored within niw0_
//  niw0_.print();
  return niw0_.logPdfMarginalized();
};

template<typename T>
T NiwSampled<T>::logPdfUnderPriorMarginalizedMerged(
    const shared_ptr<NiwSampled<T> >& other) const
{
  return niw0_.logPdfUnderPriorMarginalizedMerged(other->niw0_);
};

template<typename T>
void NiwSampled<T>::fromMerge(const NiwSampled<T>& niwA, 
  const NiwSampled<T>& niwB) 
{
  niw0_.fromMerge(niwA.niw0_,niwB.niw0_);
  normal_ = niw0_.posterior().sample();
};

template<typename T>
NiwSampled<T>* NiwSampled<T>::merge(const NiwSampled<T>& other)
{
  NiwSampled<T>* newNiw = this->copyNative();
  newNiw->niw0_.fromMerge(niw0_,other.niw0_);
  newNiw->sample();
  return newNiw;
};


template<typename T>
void NiwSampled<T>::sample() 
{
  normal_ = niw0_.posterior().sample();
};

template<typename T>
void NiwSampled<T>::print() const
{
  niw0_.posterior().print();
  normal_.print();
};

// ----------------------------------------------------------------------------

template<typename T>
NiwMarginalized<T>::NiwMarginalized(const NIW<T>& niw)
  : niw0_(niw),niw_(niw)
{};

template<typename T>
NiwMarginalized<T>::~NiwMarginalized()
{};

template<typename T>
BaseMeasure<T>* NiwMarginalized<T>::copy()
{
  return new NiwMarginalized<T>(niw0_);
};

template<typename T>
T NiwMarginalized<T>::logLikelihood(const Matrix<T,Dynamic,1>& x) const
{
  return niw_.logProb(x);
};

template<typename T>
void NiwMarginalized<T>::posterior(const Matrix<T,Dynamic,Dynamic>& x,
    const VectorXu& z, uint32_t k)
{
  niw_ = niw0_.posterior(x,z,k); 
};

template<typename T>
T NiwMarginalized<T>::logPdfUnderPrior() const
{
  // 0 since we integrate over parameters -> there is no parameters for which
  // we would want to eval a pdf
  return 0.0; 
};

template<typename T>
void NiwMarginalized<T>::print() const
{
};

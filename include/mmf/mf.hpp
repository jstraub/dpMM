/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license.  See the license file LICENSE.
 */

#pragma once

#include <Eigen/Dense>

#include <dpMM/distribution.hpp>
#include <dpMM/normalSphere.hpp>

template<typename T>
class MF : public Distribution<T>
{
public:

  MF(const Matrix<T,3,3>& R, const NormalSphere<T>& TGs,
      boost::mt19937 *pRndGen);
  MF(const Matrix<T,3,3>& R, const std::vector<NormalSphere<T> >& 
    TGs, boost::mt19937 *pRndGen);
  MF(const MF<T>& mf);
  ~MF();

  MF<T>* copy();

  MF<T> posterior(const Matrix<T,Dynamic,Dynamic>& x, const VectorXu& z,
    uint32_t k);
  MF<T> posterior(const vector<Matrix<T,Dynamic,Dynamic> >&x, const
      VectorXu& z, uint32_t k);
  // assumes vector [N, sum(x), flatten(sum(outer(x,x)))]
  MF<T> posteriorFromSS(const Matrix<T,Dynamic,1>& x);
  MF<T> posteriorFromSS(const vector<Matrix<T,Dynamic,1> >&x, const
      VectorXu& z, uint32_t k);

  MF<T> posterior() const;

  T logPdf(const Matrix<T,Dynamic,1>& x) const;

  const Matrix<T,3,3>& R() const {return R_;};
  const Matrix<T,Dynamic,Dynamic>& Sigma(uint32_t j) const
  {return TGs_[j].Sigma();};
  const std::vector<NormalSphere<T> >& TGs() const
  {return TGs_;};

  boost::mt19937 *pRndGen_;
protected:
  Matrix<T,3,3> R_;
  std::vector<NormalSphere<T> > TGs_;
  VectorXu z_; // assignment to MF axes

};


// ------------------------------------------------------

template<typename T>
MF<T>::MF(const Matrix<T,3,3>& R, const NormalSphere<T>& TG,
      boost::mt19937 *pRndGen)
  : R_(R), TGs_(6,TG), pRndGen_(pRndGen)
{};
template<typename T>
MF<T>::MF(const Matrix<T,3,3>& R, const std::vector<NormalSphere<T> >& 
    TGs, boost::mt19937 *pRndGen)
  : R_(R), TGs_(TGs), pRndGen_(pRndGen)
{};
template<typename T>
MF(const MF<T>& mf)
  : R_(mf.R()), TGs_(mf.TGs()), pRndGen_(mf.pRndGen_)
{};

template<typename T>
~MF()
{};

template<typename T>
MF<T>::logPdf(const Matrix<T,Dynamic,1>& x)
{};

template<typename T>
MF<T>::()
{};
template<typename T>
MF<T>::()
{};
template<typename T>
MF<T>::()
{};
template<typename T>
MF<T>::()
{};
template<typename T>
MF<T>::()
{};
template<typename T>
MF<T>::()
{};

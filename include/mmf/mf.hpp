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

//  MF(const Matrix<T,3,3>& R, const NormalSphere<T>& TGs,
//      boost::mt19937 *pRndGen);
//  MF();
  MF(const Matrix<T,3,3>& R, 
      const Cat<T>& pi,
      const std::vector<NormalSphere<T> >& TGs);
//      boost::mt19937 *pRndGen);
  MF(const MF<T>& mf);
  ~MF();

  T logPdf(const Matrix<T,Dynamic,1>& x) const;

  const Matrix<T,3,3>& R() const {return R_;};
  const Cat<T>& Pi() const {return pi_;};
  const Matrix<T,Dynamic,Dynamic>& Sigma(uint32_t j) const
  {return TGs_[j].Sigma();};
  const std::vector<NormalSphere<T> >& TGs() const
  {return TGs_;};

  void print() const;

//  boost::mt19937 *pRndGen_;
protected:
  Matrix<T,3,3> R_;
  std::vector<NormalSphere<T> > TGs_;
  Cat<T> pi_;
};


// ------------------------------------------------------

//template<typename T>
//MF<T>::MF(const Matrix<T,3,3>& R, const NormalSphere<T>& TG,
//      boost::mt19937 *pRndGen)
//  : R_(R), TGs_(6,TG), pRndGen_(pRndGen)
//{};
template<typename T>
MF<T>::MF(const Matrix<T,3,3>& R, 
    const Cat<T>& pi,
    const std::vector<NormalSphere<T> >& TGs)
//    boost::mt19937 *pRndGen)
  : Distribution<T>(NULL),
    R_(R), TGs_(TGs), pi_(pi)
//  pRndGen_(pRndGen)
{};

//template<typename T>
//MF<T>::MF()
//  : Distribution<T>(NULL),
//    TGs_(TGs), pi_(pi)
//{
//  R_ = Matrix<T,3,3>::Identity();
//  for(uint32_t k=0; k<6; ++k)
//    TGs_.push_back(NormalSphere<T>());
//
//};

template<typename T>
MF<T>::MF(const MF<T>& mf)
  : Distribution<T>(NULL),
    R_(mf.R()), TGs_(mf.TGs()), pi_(mf.Pi()) //, pRndGen_(mf.pRndGen_)
{};

template<typename T>
MF<T>::~MF()
{};

template<typename T>
T MF<T>::logPdf(const Matrix<T,Dynamic,1>& x) const
{
  Matrix<T,Dynamic,1> logPdf(6);
  logPdf.fill(0.);
  for(uint32_t k=0; k<6; ++k)
  {
    logPdf(k)= pi_.logPdf(k) + TGs_[k].logPdf(x);
//    cout<< (pi_.logPdf(k) + TGs_[k].logPdf(x)) << "\t";
  }
//  cout<<" -> "<<logPdf<<endl;
  return logSumExp<T>(logPdf);
};

template<typename T>
void MF<T>::print() const
{
  cout<<" -- MF: R, pi"<<endl
    << R_<<endl
    << pi_.pdf().transpose()<<endl;
  for(uint32_t k=0; k<6; ++k)
  {
    cout<<"TG "<<k<<": "<< TGs_[k].getMean().transpose()<<endl
      << TGs_[k].Sigma()<<endl;
  }
};
//template<typename T>
//MF<T>::()
//{};
//template<typename T>
//MF<T>::()
//{};
//template<typename T>
//MF<T>::()
//{};
//template<typename T>
//MF<T>::()
//{};
//template<typename T>
//MF<T>::()
//{};

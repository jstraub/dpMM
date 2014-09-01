#include <iostream>
#include <stdint.h>
#include <vector>
#include <Eigen/Dense>

#include "dpMM.hpp"
#include "cat.hpp"
#include "niw.hpp"

using namespace Eigen;
using namespace std;

/*
 * DP mixture model
 * following Neal [[http://www.stat.purdue.edu/~rdutta/24.PDF]]
 * Algo 3
 */
template <class Dist>
class DpStickMM : public DpMM<double>
{
public:
  DpStickMM(double alpha, const Dist& H)
    : alpha_(alpha), H_(H)
   {};
  ~DpStickMM()
  {};

  virtual void initialize(const MatrixXd& x);
  virtual void initialize(const boost::shared_ptr<ClData<double> >& cld)
    {cout<<"not supported"<<endl; assert(false);};
  virtual void sampleLabels();
  virtual void sampleParameters(){;}; 
  virtual const VectorXu & getLabels() {return z_;};
  virtual uint32_t getK() const { return z_.maxCoeff()+1;};

  VectorXd getCounts();

private:
  void removeEmptyClusters();

  double alpha_; // 
  vector<Dist> models_;

  MatrixXd x_;
  VectorXu z_; // indicators
  Dist H_; // 
};

// ---------------- impl -----------------------------------------------------
template <class Dist> 
void DpStickMM<Dist>::initialize(const MatrixXd& x)
{
  uint32_t K0=1;
  x_ = x;
  z_.setZero(x.cols(),1);
  for (uint32_t k=0; k<K0; ++k)
    models_.push_back(Dist(H_));
  cout<<models_.size()<<endl;
  //cout<<x_<<endl;
  cout<<alpha_<<endl;
  if (K0>1)
  {
    VectorXd pi(K0);
    pi.setOnes();
    pi /= static_cast<double>(K0);
    Catd cat(pi,H_.pRndGen_);
    for (uint32_t i=0; i<z_.size(); ++i)
      z_(i) = cat.sample();
  }
};

template <class Dist>
VectorXd DpStickMM<Dist>::getCounts()
{
  VectorXd counts(getK());
  counts.setZero();
//TODO
//  for(uint32_t k=0; k<getK(); ++k)
//    counts(k) = models_[k]->count();
  return counts;
};

template <class Dist> 
void DpStickMM<Dist>::sampleLabels()
{
  double N = z_.size();
  for(uint32_t i=0; i<z_.size(); ++i)
  {
    // compute clustercounts 
    VectorXd Nk(models_.size()); Nk.setZero(models_.size());
    for(uint32_t ii=0; ii<z_.size(); ++ii)
      Nk(z_(ii))++;
    // compute distribution pi over indicators
    VectorXd pi(models_.size()+1);
    for (uint32_t k=0; k<models_.size(); ++k)
      pi(k) = log(Nk(k))-log(N+alpha_) +models_[k].logPosteriorProb(x_,z_,k,i);
    pi(models_.size()) = log(alpha_)-log(N+alpha_)+H_.logProb(x_.col(i));
    // normalize pi and exponentiate it
    double pi_max = pi.maxCoeff();
    pi = (pi.array()-(pi_max + log((pi.array() - pi_max).exp().sum()))).exp().matrix();
    // sample new indicator
    z_(i) = Catd(pi,H_.pRndGen_).sample();
    // if z_i was a new cluster
    if(z_(i)==models_.size())
      models_.push_back(H_);

    // -------- outputs --------------
    if(i%100 ==0)
    {
      cout<<" @i="<<i
        <<"\tcounts="<<Nk.transpose()<<endl;
    }
  }
  this->removeEmptyClusters();
};

template <class Dist>
void DpStickMM<Dist>::removeEmptyClusters()
{
  for(uint32_t k=models_.size()-1; k>=0; --k)
  {
    bool haveCluster_k = false;
    for(uint32_t i=0; i<z_.size(); ++i)
      if(z_(i)==k)
      {
        haveCluster_k = true;
        break;
      }
    if (!haveCluster_k)
    {
      for (uint32_t i=0; i<z_.size(); ++i)
        if(z_(i) >= k) z_(i) --;
      models_.erase(models_.begin()+k);
    }
  }
}


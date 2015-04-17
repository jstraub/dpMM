/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license.  See the license file LICENSE.
 */
 
#pragma once

#include <Eigen/Dense>

#include <dpMM/normalSphere.hpp>

template<typename T>
class MfPrior
{
public:
  // prior on the TGMs 
  NiwSphereFull<T> niwS0_;
  // hold the SS for the 6 differrent tangent spaces
  std::vector<NiwSphereFull<T> > niwSs_; 

  MfPrior(const MfPrior<T>& mf);
  ~MfPrior();

  MfPrior<T>* copy();

  MfPrior<T> posterior(const Matrix<T,Dynamic,Dynamic>& x, const VectorXu& z, 
    uint32_t k);
  MfPrior<T> posterior(const vector<Matrix<T,Dynamic,Dynamic> >&x, const
      VectorXu& z, uint32_t k);
  // assumes vector [N, sum(x), flatten(sum(outer(x,x)))]
  MfPrior<T> posteriorFromSS(const Matrix<T,Dynamic,1>& x);
  MfPrior<T> posteriorFromSS(const vector<Matrix<T,Dynamic,1> >&x, const
      VectorXu& z, uint32_t k);

  MfPrior<T> posterior() const;
};

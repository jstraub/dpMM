/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>                    
 * Licensed under the MIT license. See the license file LICENSE.                
 */
#pragma once
#include <iostream>
#include <stdint.h>
#include <vector>
#include <Eigen/Dense>

#include <boost/shared_ptr.hpp>

#include "dirMM.hpp"


using namespace Eigen;
using std::endl, std::cout, std::vector;
using boost::shared_ptr;

template <typename T>
class DirMMsplitMerge : public DirMM<T>
{
public:
  DirMMsplitMerge(const Dir<T>& alpha, const shared_ptr<BaseMeasure<T> >& theta);
  virtual ~DirMMsplitMerge();

  virtual void sampleParameters();

protected:
  proposeSplit();
  BaseMeasure<T>* proposeMerge();
  merge();
  split();

  mhRatioMerge();
  mhRatioSplit();

};

// -------------------------------------- impl --------------------------------
template <typename T>
DirMMsplitMerge<T>::DirMMsplitMerge()
{};

template <typename T>
DirMMsplitMerge<T>::~DirMMsplitMerge()
{};


template <typename T>
DirMMsplitMerge<T>::sampleParameters()
{
  DirMM<T>::sampleParameters();
  queue<uint32_t> emptyClusters;
  for(uint32_t k=0; k<this->K_; ++k)
    if(this->thetas_[k]->count() == 0)
    {
      emptyClusters.push(k);
    }
  for(uint32_t k=0; k<this->K_; ++k)
    if(!emptyClusters.empty())
    {
      std::pair< BaseMeasure<T>*, BaseMeasure<T>* > split = proposeSplit(k);
      if(split.first && split.second)
      {
        this->thetas_[k].reset(split.first);
        this->thetas_[emptyClusters.front()].reset(split.second);
        emptyClusters.pop();
      }
    }
  for(uint32_t k=0; k<this->K_; ++k)
    for(uint32_t j=0; j<this->K_; ++j)
    {
      BaseMeasure<T>* merged = proposeMerge(k,j);
      if(merged) 
      {
        this->thetas_[k].reset(merged);
        this->thetas_[j].reset(new theta0_->copy());
      }
    }
};


template <typename T>
std::pair< BaseMeasure<T>*, BaseMeasure<T>* > DirMMsplitMerge<T>::proposeSplit(
    uint32_t k)
{

  return std::pair< BaseMeasure<T>*, BaseMeasure<T>* > (NULL,NULL);
};

template <typename T>
BaseMeasure<T>* DirMMsplitMerge<T>::proposeMerge(uint32_t k, uint32_t j)
{
  BaseMeasure<T>* merged(this->thetas_[k]->merge(*this->thetas_[j]));


  return NULL;
};



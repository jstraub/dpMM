/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */
 

#pragma once

#include <Eigen/Dense>

template<typename T>
class SufficientStatistics
{
  
};

/* sufficient statistics for gaussian distribution
 * has outer product, sum over data and count
 */
template<typename T>
class ssGauss : public SufficientStatistics<T>
{

};

/* sufficient statistics for categorical distribution 
 * mainly just counts
 */
template<typename T>
class ssCat : public SufficientStatistics<T>
{

};

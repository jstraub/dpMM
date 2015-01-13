/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */
// =============================================================================
// == clusters.cpp
// == --------------------------------------------------------------------------
// == A class for all Gaussian clusters with sub-clusters
// == --------------------------------------------------------------------------
// == Copyright 2013. MIT. All Rights Reserved.
// == Written by Jason Chang 11-03-2013
// == --------------------------------------------------------------------------
// == If this code is used, the following should be cited:
// == 
// == [1] J. Chang and J. W. Fisher II, "Parallel Sampling of DP Mixtures
// ==     Models using Sub-Cluster Splits". Neural Information Processing
// ==     Systems (NIPS 2013), Lake Tahoe, NV, USA, Dec 2013.
// =============================================================================

#ifndef _CLUSTERS_H_INCLUDED_
#define _CLUSTERS_H_INCLUDED_

#include <math.h>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>

#include <omp.h>

//#include "matrix.h"
//#include "mex.h"
//#include "helperMEX.h"
//#include "debugMEX.h"

#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"
#include "gsl/gsl_permutation.h"
#include "gsl/gsl_cdf.h"

#include "dpmmSubclusters/common.h"

#include "dpmmSubclusters/niw_sampled.h"
#include "dpmmSubclusters/cluster_sampledT.cpp"
#include "dpmmSubclusters/linkedList.cpp"

#include "dpmmSubclusters/reduction_array.h"
#include "dpmmSubclusters/reduction_array2.h"
#include "dpmmSubclusters/linear_algebra.h"
#include "dpmmSubclusters/sample_categorical.h"


using std::vector;
using std::cout;
using std::endl;

class clusters
{
public:
   int N; // number of data points
   int D; // dimensionality of data
   int D2; // D*D
   int K; // number of non-empty clusters
   int Nthreads; // number of threads to use in parallel

   double* data; // not dynamically allocated, just a pointer to the data
   double* phi;  // not dynamically allocated, just a pointer to the heights
   vector<bool> randomSplitIndex; // dynamically allocated

   // cluster parameters
   niw_sampled hyper;
   vector< cluster_sampledT<niw_sampled>* > params;
   vector<double> likelihoodOld;
   vector<double> likelihoodDelta;
   vector<bool> splittable;

   // DP stick-breaking stuff
   double alpha;
   double logalpha;
   vector<double> sticks;
   bool always_splittable;

   // misc
   vector<int> k2z;
   vector<int> z2k;
   linkedList<int> alive;
   vector< linkedListNode<int>* > alive_ptrs;

   // random number generation

   // temporary space for parallel processing
   vector< vector<double> > probsArr;
   vector<gsl_rng*> rArr;

   // supercluster stuff
   bool useSuperclusters;
   vector<int> superclusters;
   vector<int> supercluster_labels;
   vector<int> supercluster_labels_count;


public:
   // --------------------------------------------------------------------------
   // -- clusters
   // --   constructor; initializes to empty
   // --------------------------------------------------------------------------
   clusters();
   // --------------------------------------------------------------------------
   // -- clusters
   // --   copy constructor;
   // --------------------------------------------------------------------------
   clusters(const clusters& that);
   // --------------------------------------------------------------------------
   // -- operator=
   // --   assignment operator
   // --------------------------------------------------------------------------
   clusters& operator=(const clusters& that);
   // --------------------------------------------------------------------------
   // -- copy
   // --   returns a copy of this
   // --------------------------------------------------------------------------
   void copy(const clusters& that);
   // --------------------------------------------------------------------------
   // -- clusters
   // --   constructor; intializes to all the values given
   // --------------------------------------------------------------------------
   clusters(int _N, int _D, double* _data, double* _phi,
            double _alpha, niw_sampled &_hyper, int _Nthreads,
            bool _useSuperclusters, bool _always_splittable);

   // --------------------------------------------------------------------------
   // -- ~clusters
   // --   destructor
   // --------------------------------------------------------------------------
   virtual ~clusters();

public:
   // --------------------------------------------------------------------------
   // -- initialize
   // --   populates the initial statistics
   // --------------------------------------------------------------------------
//   void initialize(const mxArray* cluster_params);
   void initialize();

   // --------------------------------------------------------------------------
   // -- write_output
   // --   creates and writes the output cluster structure
   // --------------------------------------------------------------------------
//   void write_output(mxArray* &plhs);

   // --------------------------------------------------------------------------
   // -- populate_k2z_z2k
   // --   Populates the k and z mappings
   // --------------------------------------------------------------------------
   void populate_k2z_z2k();

   // --------------------------------------------------------------------------
   // -- sample_params
   // --   Samples the parameters for each cluster and the mixture weights
   // --------------------------------------------------------------------------
   void sample_params();

   // --------------------------------------------------------------------------
   // -- sample_superclusters
   // --   Samples the supercluster assignments
   // --------------------------------------------------------------------------
   void sample_superclusters();

   // --------------------------------------------------------------------------
   // -- sample_labels
   // --   Samples the label assignments for each data point
   // --------------------------------------------------------------------------
   void sample_labels();

   // --------------------------------------------------------------------------
   // -- propose_merges
   // --   Samples the label assignments for each data point
   // --------------------------------------------------------------------------
   void propose_merges();

   // --------------------------------------------------------------------------
   // -- propose_splits
   // --   Samples the label assignments for each data point
   // --------------------------------------------------------------------------
   void propose_splits();

   void propose_random_split_assignments();
   void propose_random_splits();
   void propose_random_merges();
   
   double joint_loglikelihood();
   
   int getK() const;
   int getNK() const;
};

inline int clusters::getK() const
{
   return alive.getLength();
}
inline int clusters::getNK() const
{
   int maxNK = 0;
   linkedListNode<int>* node = alive.getFirst();
   while (node!=NULL)
   {
      int m = node->getData();
      maxNK = std::max(maxNK, params[m]->getN());
      node = node->getNext();
   }
   return maxNK;
}


#endif

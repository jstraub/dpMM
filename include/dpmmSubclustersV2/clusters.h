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

#include "matrix.h"
#include "mex.h"
#include <math.h>

#include "helperMEX.h"
#include "debugMEX.h"

#include "normalmean_sampled.h"
#include "cluster_sampledT.cpp"
#include "linkedList.cpp"

#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"
#include "gsl/gsl_permutation.h"
#include "gsl/gsl_cdf.h"

#include "filters.h"

#include <vector>
using std::vector;


class clusters
{
public:
   int X, Y; //dimension of image
   int N; // number of data points
   int D; // dimensionality of data
   int D2; // D*D
   int K; // number of non-empty clusters
   int maxK; // max number of non-empty clusters
   int Nthreads; // number of threads to use in parallel

   double* data; // not dynamically allocated, just a pointer to the data
   double* phi;  // not dynamically allocated, just a pointer to the heights
   bool* omask;
   unsigned long* closestIndices;
   vector<bool> randomSplitIndex; // dynamically allocated

   int numSigma;

   // cluster parameters
   normalmean_sampled hyper;
   vector< cluster_sampledT<normalmean_sampled>* > params;
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
   gsl_rng *rand_gen;

   // temporary space for parallel processing
   vector< vector<double> > probsArr;
   vector<gsl_rng*> rArr;

   // supercluster stuff
   bool useSuperclusters;
   vector<int> superclusters;
   vector<int> supercluster_labels;
   vector<int> supercluster_labels_count;

   // new split/merge stuff
   vector<double> logpzMtx;

   // blurred stuff
   vector< vector<double> > blurKernels;
   vector< vector<double> > blurKernelOthers;
   int blurKernelNum;
   vector<double> meanImage;
   vector<double> meanSubImage;

   double MRFw;


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
   clusters(int _N, int _D, int _X, int _Y, double* _data, double* _phi, normalmean_sampled &_hyper, const mxArray* inparams);

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
   void initialize(const mxArray* cluster_params);

   // --------------------------------------------------------------------------
   // -- write_output
   // --   creates and writes the output cluster structure
   // --------------------------------------------------------------------------
   void write_output(mxArray* &plhs, mxArray* &plhs2, mxArray* &plhs3);

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
   void sample_mu();
   void sample_sigma();
   void sample_sigma_split(int splitk);

   void max_params();
   void max_sigma();
   void max_sigma_split(int splitk);

   void setSigmas(int numSigma);

   // --------------------------------------------------------------------------
   // -- sample_superclusters
   // --   Samples the supercluster assignments
   // --------------------------------------------------------------------------
   void sample_superclusters();


   void sample_blurkernel();
   void max_blurkernel();

   void populate_meanDifference();
   void populate_meanImage();
   void populate_meanSubImage();
   void copy_meanImage();

   // --------------------------------------------------------------------------
   // -- sample_labels
   // --   Samples the label assignments for each data point
   // --------------------------------------------------------------------------
   void sample_labels();
   void sample_labels2();
   void sample_labels3();

   bool isBlurring();
   void sample_labels3_noblur(double weight);
   void max_labels2();
   void max_labels3();
   void max_labels3_noblur(double weight);

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

   void propose_merges2();
   void propose_splits2();

   double calc_sigma_merge(int k1, int k2);
   void propose_merges3();
   double calc_sigma_split(int splitk);
   void propose_splits3();

   void propose_random_split_assignments();
   void propose_random_splits();
   void propose_random_merges();
   
   double joint_loglikelihood();
   
   int getK() const;
   int getNK() const;
   void checkData();
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
      maxNK = max(maxNK, params[m]->getN());
      node = node->getNext();
   }
   return maxNK;
}


#endif

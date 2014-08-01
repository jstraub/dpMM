// =============================================================================
// == cluster_sampledT.h
// == --------------------------------------------------------------------------
// == A template class for the sub-clusters.  You will typically want to write
// == your own prior class.
// == --------------------------------------------------------------------------
// == Copyright 2013. MIT. All Rights Reserved.
// == Written by Jason Chang 11-06-2013
// =============================================================================


//#include "matrix.h"
//#include "mex.h"
#include <math.h>
//#include "array.h"

#include "gsl/gsl_rng.h"

#include "dpmmSubclusters/common.h"
#include "dpmmSubclusters/myfuncs.h"

#define mexPrintf printf
#define mexErrMsgTxt printf

//#include "helperMEX.h"
//#include "debugMEX.h"

template <typename prior>
class cluster_sampledT
{
private:
   // prior hyperparameters
   double alphah; // concentration parameter

   prior params; // regular-cluster parameter
   prior paramsl; // left sub-cluster parameter
   prior paramsr; // right sub-cluster parameter
   double stickl; // left sub-cluster weight
   double stickr; // right sub-cluster weight

   prior randoml; // left sub-cluster used for random split/merges
   prior randomr; // right sub-cluster used for random split/merges
   double randomstickl; // left sub-cluster used for random split/merges
   double randomstickr; // right sub-cluster used for random split/merges

public:
   // --------------------------------------------------------------------------
   // -- cluster_sampledT
   // --   initializes with the specified parameters
   // --------------------------------------------------------------------------
   cluster_sampledT(const prior& that, double _alphah);

   // --------------------------------------------------------------------------
   // -- empty
   // --   Empties out the statistics of the regular- and sub-clusters
   // -- Typically used after parameters have been sampled.
   // --------------------------------------------------------------------------
   void empty();
   void empty_subclusters();
   // --------------------------------------------------------------------------
   // -- clear
   // --   Empties out the statistics of the regular- and sub-clusters and 
   // -- resamples parameters for the empty cluster.
   // --------------------------------------------------------------------------
   void clear();
   void clear_subclusters();

   // --------------------------------------------------------------------------
   // -- isempty
   // --   Returns true iff the cluster contains no data in it
   // --------------------------------------------------------------------------
   bool isempty() const;

   // --------------------------------------------------------------------------
   // -- get
   // --   gets the particular parameter for a regular- or sub-cluster
   // --------------------------------------------------------------------------
   int getN() const;
   int getlN() const;
   int getrN() const;
   int getrandomlN() const;
   int getrandomrN() const;
   double getstickl() const;
   double getstickr() const;
   double getrandomstickl() const;
   double getrandomstickr() const;

   // --------------------------------------------------------------------------
   // -- set
   // --   Sets the weight for the particular sub-cluster
   // --------------------------------------------------------------------------
   void setstickl(double _stickl);
   void setstickr(double _stickr);
   void setrandomstickl(double _randomstickl);
   void setrandomstickr(double _randomstickr);

   // --------------------------------------------------------------------------
   // -- get/set regular- and sub-cluster parameters
   // --   Gets or sets the the parameters for a regular- or sub- cluster
   // --------------------------------------------------------------------------
   prior* get_params();
   prior* get_paramsl();
   prior* get_paramsr();
   prior* get_randoml();
   prior* get_randomr();
   void set_params(prior* _params);
   void set_paramsl(prior* _params);
   void set_paramsr(prior* _params);
   void set_randoml(prior* _params);
   void set_randomr(prior* _params);

   // --------------------------------------------------------------------------
   // -- update_posteriors
   // --   Updates the posterior hyperparameters based on the data
   // --------------------------------------------------------------------------
   void update_posteriors();

   // --------------------------------------------------------------------------
   // -- update_posteriors_sampled
   // --   Updates the posterior hyperparameters based on the data and samples
   // -- new parameters
   // --------------------------------------------------------------------------
   void update_posteriors_sample();

   // --------------------------------------------------------------------------
   // -- sample
   // --   Samples regular- and sub-cluster parameters based on the current
   // -- posterior hyperparameters. Also samples sub-cluster weights
   // --------------------------------------------------------------------------
   void sample_param();
   // --------------------------------------------------------------------------
   // -- sample_xxx
   // --   Samples regular- or sub-cluster parameters based on the current
   // -- posterior hyperparameters.
   // --------------------------------------------------------------------------
   void sample_param_regular();
   void sample_param_subclusters();

   // --------------------------------------------------------------------------
   // -- sample_subcluster_label
   // --   Samples a new subcluster label. If it should belong to the left sub-
   // -- cluster, will return 0.25. If it should belong to the right sub-cluster
   // -- will return 0.75. Also returns the loglikelihood of the data point
   // -- conditioned on the subcluster label and parameter.
   // --
   // --   parameters:
   // --     - logpl - the data log likelihood of belonging to sub-cluster l
   // --     - logpr - the data log likelihood of belonging to sub-cluster r
   // --     - rand_num - a GSL random number generator
   // --   return parameters:
   // --     - loglikelihood - the log likelihood of the data conditioned on the
   // --       chosen cluster
   // --   return value:
   // --     - 0.25 or 0.75 depending on sampled l or r
   // --------------------------------------------------------------------------
   double sample_subcluster_label(double logpl, double logpr, gsl_rng *rand_num, double &loglikelihood) const;

   // --------------------------------------------------------------------------
   // -- update_upwards_posterior
   // --   Updates regular- and sub-cluster posterior hyperparameters based on 
   // -- the summary statistics in the left and right sub-clusters
   // --------------------------------------------------------------------------
   void update_upwards_posterior();

   // --------------------------------------------------------------------------
   // -- update_subclusters_sampled
   // --   Updates sub-cluster posterior hyperparameters based on the summary
   // -- statistics and samples the sub-cluster parameters
   // --------------------------------------------------------------------------
   void update_subclusters_sample();

   // --------------------------------------------------------------------------
   // -- update_subclusters_posteriors
   // --   Updates sub-cluster posterior hyperparameters based on the summary
   // -- statistics
   // --------------------------------------------------------------------------
   void update_subclusters_posteriors();

   // --------------------------------------------------------------------------
   // -- update_random_posteriors
   // --   Updates random sub-cluster posterior hyperparameters based on the
   // -- summary statistics
   // --------------------------------------------------------------------------
   void update_random_posteriors();


   // --------------------------------------------------------------------------
   // -- data_loglikelihood_marginalized
   // --   Finds the log likelihood of the data marginalizing out the parameters
   // -- assuming conjugate priors. This calculates p(x|lambda)
   // --------------------------------------------------------------------------
   double data_loglikelihood_marginalized();
   double data_lloglikelihood_marginalized();
   double data_rloglikelihood_marginalized();
   double data_randomlloglikelihood_marginalized();
   double data_randomrloglikelihood_marginalized();

   // --------------------------------------------------------------------------
   // -- data_loglikelihood_marginalized_testmerge
   // --   Finds the log likelihood of the data marginalizing out the parameters
   // -- assuming conjugate priors if this cluster was merged with other
   // --
   // --   parameters:
   // --     - other : a poitner to another cluster_sampledT
   // --------------------------------------------------------------------------
   double data_loglikelihood_marginalized_testmerge(cluster_sampledT* &other) const;

   // --------------------------------------------------------------------------
   // -- data_loglikelihood
   // --   Finds the log likelihood of the data conditioned on the current
   // -- parameters. This calculates p(x|theta)
   // --------------------------------------------------------------------------
   double data_loglikelihood() const;
   double data_lloglikelihood() const;
   double data_rloglikelihood() const;
   double data_randomlloglikelihood() const;
   double data_randomrloglikelihood() const;

   // --------------------------------------------------------------------------
   // -- merge_with
   // --   Merges this cluster with other, sets left sub-cluster to be the old
   // -- regular cluster, and right-subcluster to be the old other regular
   // -- cluster. dosample specifies if the parameters should be resampled.
   // --------------------------------------------------------------------------
   void merge_with(cluster_sampledT* other, bool dosample = true);
   
   // --------------------------------------------------------------------------
   // -- split_from
   // --   Modifies the current cluster as if it was just split off from other.
   // -- Copies the right sub-cluster over.
   // --------------------------------------------------------------------------
   void split_from(cluster_sampledT* other);
   // --------------------------------------------------------------------------
   // -- split_fix
   // --   Modifies the current cluster as if it was just split off from the
   // -- regular cluster. Copies the left sub-cluster over.
   // --------------------------------------------------------------------------
   void split_fix();
   // --------------------------------------------------------------------------
   // -- split_from_random
   // --   Modifies the current cluster as if it was just split off from other.
   // -- Copies the right random over.
   // --------------------------------------------------------------------------
   void split_from_random(cluster_sampledT* other);
   // --------------------------------------------------------------------------
   // -- split_fix_random
   // --   Modifies the current cluster as if it was just split off from the
   // -- regular cluster. Copies the left random cluster over.
   // --------------------------------------------------------------------------
   void split_fix_random();

   // --------------------------------------------------------------------------
   // -- Jdivergence
   // --   Calculates the J-Divergence between this regular-cluster and other's
   // -- regular cluster. If J-Divergence is difficult to calculate, any 
   // -- distance measure can be used.
   // --------------------------------------------------------------------------
   double Jdivergence(cluster_sampledT* &other);
};


//#endif

// =============================================================================
// == cluster_sampledT.cpp
// == --------------------------------------------------------------------------
// == A template class for the sub-clusters.  You will typically want to write
// == your own prior class.
// == --------------------------------------------------------------------------
// == Copyright 2013. MIT. All Rights Reserved.
// == Written by Jason Chang 11-06-2013
// =============================================================================

#include "cluster_sampledT.h"


// --------------------------------------------------------------------------
// -- cluster_sampledT
// --   initializes with the specified parameters
// --------------------------------------------------------------------------
template <typename prior>
cluster_sampledT<prior>::cluster_sampledT(const prior& that, double _alphah) :
   params(that), paramsl(that), paramsr(that), alphah(_alphah),
   randoml(that), randomr(that)
{
   stickl = log(0.5);
   stickr = log(0.5);
}




// --------------------------------------------------------------------------
// -- empty
// --   Empties out the statistics of the regular- and sub-clusters
// -- Typically used after parameters have been sampled.
// --------------------------------------------------------------------------
template <typename prior>
void cluster_sampledT<prior>::empty()
{
   params.empty();
   paramsl.empty();
   paramsr.empty();
}
template <typename prior>
void cluster_sampledT<prior>::empty_subclusters()
{
   paramsl.empty();
   paramsr.empty();
}

// --------------------------------------------------------------------------
// -- clear
// --   Empties out the statistics of the regular- and sub-clusters and 
// -- resamples parameters for the empty cluster.
// --------------------------------------------------------------------------
template <typename prior>
void cluster_sampledT<prior>::clear()
{
   params.clear();
   paramsl.clear();
   paramsr.clear();
}
template <typename prior>
void cluster_sampledT<prior>::clear_subclusters()
{
   paramsl.clear();
   paramsr.clear();
}

// --------------------------------------------------------------------------
// -- isempty
// --   Returns true iff the cluster contains no data in it
// --------------------------------------------------------------------------
template <typename prior>
bool cluster_sampledT<prior>::isempty() const        { return params.isempty();}

// --------------------------------------------------------------------------
// -- get
// --   gets the particular parameter for a regular- or sub-cluster
// --------------------------------------------------------------------------
template <typename prior>
int cluster_sampledT<prior>::getN() const                   { return params.getN();}
template <typename prior>
int cluster_sampledT<prior>::getlN() const                  { return paramsl.getN();}
template <typename prior>
int cluster_sampledT<prior>::getrN() const                  { return paramsr.getN();}
template <typename prior>
int cluster_sampledT<prior>::getrandomlN() const            { return randoml.getN();}
template <typename prior>
int cluster_sampledT<prior>::getrandomrN() const            { return randomr.getN();}
template <typename prior>
double cluster_sampledT<prior>::getstickl() const           { return stickl;}
template <typename prior>
double cluster_sampledT<prior>::getstickr() const           { return stickr;}
template <typename prior>
double cluster_sampledT<prior>::getrandomstickl() const     { return randomstickl;}
template <typename prior>
double cluster_sampledT<prior>::getrandomstickr() const     { return randomstickr;}

// --------------------------------------------------------------------------
// -- set
// --   Sets the weight for the particular sub-cluster
// --------------------------------------------------------------------------
template <typename prior>
void cluster_sampledT<prior>::setstickl(double _stickl)              { stickl = _stickl;}
template <typename prior>
void cluster_sampledT<prior>::setstickr(double _stickr)              { stickr = _stickr;}
template <typename prior>
void cluster_sampledT<prior>::setrandomstickl(double _randomstickl)  { randomstickl = _randomstickl;}
template <typename prior>
void cluster_sampledT<prior>::setrandomstickr(double _randomstickr)  { randomstickr = _randomstickr;}

// --------------------------------------------------------------------------
// -- get/set regular- and sub-cluster parameters
// --   Gets or sets the the parameters for a regular- or sub- cluster
// --------------------------------------------------------------------------
template <typename prior>
prior* cluster_sampledT<prior>::get_params()                { return &params;}
template <typename prior>
prior* cluster_sampledT<prior>::get_paramsl()               { return &paramsl;}
template <typename prior>
prior* cluster_sampledT<prior>::get_paramsr()               { return &paramsr;}
template <typename prior>
prior* cluster_sampledT<prior>::get_randoml()               { return &randoml;}
template <typename prior>
prior* cluster_sampledT<prior>::get_randomr()               { return &randomr;}
template <typename prior>
void cluster_sampledT<prior>::set_params(prior* _params)    { params = *_params;}
template <typename prior>
void cluster_sampledT<prior>::set_paramsl(prior* _params)   { paramsl = *_params;}
template <typename prior>
void cluster_sampledT<prior>::set_paramsr(prior* _params)   { paramsr = *_params;}
template <typename prior>
void cluster_sampledT<prior>::set_randoml(prior* _params)   { randoml = *_params;}
template <typename prior>
void cluster_sampledT<prior>::set_randomr(prior* _params)   { randomr = *_params;}


// --------------------------------------------------------------------------
// -- update_posteriors
// --   Updates the posterior hyperparameters based on the data
// --------------------------------------------------------------------------
template <typename prior>
void cluster_sampledT<prior>::update_posteriors()
{
   params.update_posteriors();
   paramsl.update_posteriors();
   paramsr.update_posteriors();
}

// --------------------------------------------------------------------------
// -- update_posteriors_sampled
// --   Updates the posterior hyperparameters based on the data and samples
// -- new parameters
// --------------------------------------------------------------------------
template <typename prior>
void cluster_sampledT<prior>::update_posteriors_sample()
{
   params.update_posteriors_sample();
   paramsl.update_posteriors_sample();
   paramsr.update_posteriors_sample();
}

// --------------------------------------------------------------------------
// -- sample_param
// --   Samples regular- and sub-cluster parameters based on the current
// -- posterior hyperparameters. Also samples sub-cluster weights
// --------------------------------------------------------------------------
template <typename prior>
void cluster_sampledT<prior>::sample_param()
{
   params.sample();
   paramsl.sample();
   paramsr.sample();

   double alphal = paramsl.getN() + alphah;
   double alphar = paramsr.getN() + alphah;
   stickl = gsl_ran_gamma(params.get_r(),0.5*alphal,1);
   stickr = gsl_ran_gamma(params.get_r(),0.5*alphar,1);
   double total = stickl+stickr;
   stickl = log(stickl) - log(total);
   stickr = log(stickr) - log(total);
}
// --------------------------------------------------------------------------
// -- sample_param_xxx
// --   Samples regular- or sub-cluster parameters based on the current
// -- posterior hyperparameters.
// --------------------------------------------------------------------------
template <typename prior>
void cluster_sampledT<prior>::sample_param_regular()
{
   params.sample();
}
template <typename prior>
void cluster_sampledT<prior>::sample_param_subclusters()
{
   paramsl.sample();
   paramsr.sample();
}





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
template <typename prior>
double cluster_sampledT<prior>::sample_subcluster_label(double logpl, double logpr, gsl_rng *rand_num, double &loglikelihood) const
{
   double l = logpl + stickl;
   double r = logpr + stickr;
   double norm = logsumexp(l,r);

   if (my_rand(rand_num) < exp(l-norm))
   {
      loglikelihood = l;
      return 0.25;
   }
   else
   {
      loglikelihood = r;
      return 0.75;
   }
}



// --------------------------------------------------------------------------
// -- update_upwards_posterior
// --   Updates regular- and sub-cluster posterior hyperparameters based on 
// -- the summary statistics in the left and right sub-clusters
// --------------------------------------------------------------------------
template <typename prior>
void cluster_sampledT<prior>::update_upwards_posterior()
{
   params.empty();
   paramsl.update_posteriors();
   paramsr.update_posteriors();
   params.merge_with(paramsl, paramsr, false);
}
// --------------------------------------------------------------------------
// -- update_subclusters_sampled
// --   Updates sub-cluster posterior hyperparameters based on the summary
// -- statistics and samples the sub-cluster parameters
// --------------------------------------------------------------------------
template <typename prior>
void cluster_sampledT<prior>::update_subclusters_sample()
{
   paramsl.update_posteriors_sample();
   paramsr.update_posteriors_sample();
}
// --------------------------------------------------------------------------
// -- update_subclusters_posteriors
// --   Updates sub-cluster posterior hyperparameters based on the summary
// -- statistics
// --------------------------------------------------------------------------
template <typename prior>
void cluster_sampledT<prior>::update_subclusters_posteriors()
{
   paramsl.update_posteriors();
   paramsr.update_posteriors();
}
// --------------------------------------------------------------------------
// -- update_random_posteriors
// --   Updates random sub-cluster posterior hyperparameters based on the
// -- summary statistics
// --------------------------------------------------------------------------
template <typename prior>
void cluster_sampledT<prior>::update_random_posteriors()
{
   randoml.update_posteriors();
   randomr.update_posteriors();
}




// --------------------------------------------------------------------------
// -- data_loglikelihood_marginalized
// --   Finds the log likelihood of the data marginalizing out the parameters
// -- assuming conjugate priors. This calculates p(x|lambda)
// --------------------------------------------------------------------------
template <typename prior>
double cluster_sampledT<prior>::data_loglikelihood_marginalized()          { return params.data_loglikelihood_marginalized();}
template <typename prior>
double cluster_sampledT<prior>::data_lloglikelihood_marginalized()         { return paramsl.data_loglikelihood_marginalized(); }
template <typename prior>
double cluster_sampledT<prior>::data_rloglikelihood_marginalized()         { return paramsr.data_loglikelihood_marginalized(); }
template <typename prior>
double cluster_sampledT<prior>::data_randomlloglikelihood_marginalized()   { return randoml.data_loglikelihood_marginalized(); }
template <typename prior>
double cluster_sampledT<prior>::data_randomrloglikelihood_marginalized()   { return randomr.data_loglikelihood_marginalized(); }

// --------------------------------------------------------------------------
// -- data_loglikelihood_marginalized_testmerge
// --   Finds the log likelihood of the data marginalizing out the parameters
// -- assuming conjugate priors if this cluster was merged with other
// --
// --   parameters:
// --     - other : a poitner to another cluster_sampledT
// --------------------------------------------------------------------------
template <typename prior>
double cluster_sampledT<prior>::data_loglikelihood_marginalized_testmerge(cluster_sampledT* &other) const
{
   return params.data_loglikelihood_marginalized_testmerge(&(other->params));
   //prior temp = params;
   //temp.merge_with(other->params, false);
   //return temp.data_loglikelihood_marginalized();
}

// --------------------------------------------------------------------------
// -- data_loglikelihood
// --   Finds the log likelihood of the data conditioned on the current
// -- parameters. This calculates p(x|theta)
// --------------------------------------------------------------------------
template <typename prior>
double cluster_sampledT<prior>::data_loglikelihood() const       { return params.data_loglikelihood();}
template <typename prior>
double cluster_sampledT<prior>::data_lloglikelihood() const      { return paramsl.data_loglikelihood();}
template <typename prior>
double cluster_sampledT<prior>::data_rloglikelihood() const      { return paramsr.data_loglikelihood();}
template <typename prior>
double cluster_sampledT<prior>::data_randomlloglikelihood() const      { return randoml.data_loglikelihood();}
template <typename prior>
double cluster_sampledT<prior>::data_randomrloglikelihood() const      { return randomr.data_loglikelihood();}




// --------------------------------------------------------------------------
// -- merge_with
// --   Merges this cluster with other, sets left sub-cluster to be the old
// -- regular cluster, and right-subcluster to be the old other regular
// -- cluster. dosample specifies if the parameters should be resampled.
// --------------------------------------------------------------------------
template <typename prior>
void cluster_sampledT<prior>::merge_with(cluster_sampledT* other, bool dosample)
{
   paramsl = params;
   paramsr = other->params;
   params.clear();
   params.merge_with(paramsl, paramsr, dosample);
}

// --------------------------------------------------------------------------
// -- split_from
// --   Modifies the current cluster as if it was just split off from other.
// -- Copies the right sub-cluster over.
// --------------------------------------------------------------------------
template <typename prior>
void cluster_sampledT<prior>::split_from(cluster_sampledT* other)
{
   params = other->paramsr;
   stickl = log(0.5);
   stickr = log(0.5);
}
// --------------------------------------------------------------------------
// -- split_fix
// --   Modifies the current cluster as if it was just split off from the
// -- regular cluster. Copies the left sub-cluster over.
// --------------------------------------------------------------------------
template <typename prior>
void cluster_sampledT<prior>::split_fix()
{
   params = paramsl;
   stickl = log(0.5);
   stickr = log(0.5);
}

// --------------------------------------------------------------------------
// -- split_from_random
// --   Modifies the current cluster as if it was just split off from other.
// -- Copies the right random over.
// --------------------------------------------------------------------------
template <typename prior>
void cluster_sampledT<prior>::split_from_random(cluster_sampledT* other)
{
   params = other->randomr;
   stickl = log(0.5);
   stickr = log(0.5);
   params.sample();
}
// --------------------------------------------------------------------------
// -- split_fix_random
// --   Modifies the current cluster as if it was just split off from the
// -- regular cluster. Copies the left random cluster over.
// --------------------------------------------------------------------------
template <typename prior>
void cluster_sampledT<prior>::split_fix_random()
{
   params = randoml;
   stickl = log(0.5);
   stickr = log(0.5);
   params.sample();
}

// --------------------------------------------------------------------------
// -- Jdivergence
// --   Calculates the J-Divergence between this regular-cluster and other's
// -- regular cluster. If J-Divergence is difficult to calculate, any 
// -- distance measure can be used.
// --------------------------------------------------------------------------
template <typename prior>
double cluster_sampledT<prior>::Jdivergence(cluster_sampledT* &other)
{
   return params.Jdivergence(other->params);
}

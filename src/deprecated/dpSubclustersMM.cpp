// =============================================================================
// == clusters.h
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

#include "dpSubclustersMM.hpp"

// --------------------------------------------------------------------------
// -- clusters
// --   constructor; initializes to empty
// --------------------------------------------------------------------------
clusters::clusters() :
   N(0), Nthreads(0),
   randomSplitIndex(0),
   params(0), 
   likelihoodOld(0), likelihoodDelta(0), splittable(0), 
   sticks(0), k2z(0), z2k(0), probsArr(0),
   rArr(0), superclusters(0), supercluster_labels(0), supercluster_labels_count(0)
{
   D2 = D*D;
   //TODO
   rand_gen = initialize_gsl_rand(mx_rand());
   useSuperclusters = false;
   always_splittable = false;
}

// --------------------------------------------------------------------------
// -- clusters
// --   copy constructor;
// --------------------------------------------------------------------------
clusters::clusters(const clusters& that)
{
   copy(that);
}
// --------------------------------------------------------------------------
// -- operator=
// --   assignment operator
// --------------------------------------------------------------------------
clusters& clusters::operator=(const clusters& that)
{
   if (this != &that)
   {
      if (N>0)
      {
         for (int i=0; i<N; i++)
            if (params[i]!=NULL)
               delete params[i];
      }
      if (Nthreads>0)
      {
         for (int t=0; t<Nthreads; t++)
            if (rArr[t]!=NULL) gsl_rng_free(rArr[t]);
      }

      gsl_rng_free(rand_gen);
      copy(that);
   }
   return *this;
}
// --------------------------------------------------------------------------
// -- copy
// --   returns a copy of this
// --------------------------------------------------------------------------
void clusters::copy(const clusters& that)
{
   N = that.N;
   D = that.D;
   D2 = D*D;
   Nthreads = that.Nthreads;

   data = that.data;
   phi = that.phi;

   alpha = that.alpha;
   logalpha = log(alpha);
   params.assign(N,NULL);
   sticks.resize(N);
   k2z.resize(N);
   z2k.resize(N);
   alive = that.alive;
   alive_ptrs.assign(N,NULL);
   likelihoodOld.resize(N);
   likelihoodDelta.resize(N);
   splittable.resize(N);

   randomSplitIndex.resize(N);

   for (int i=0; i<N; i++)
      if (that.params[i]!=NULL)
         params[i] = new cluster_sampledT<niw_sampled>(*(that.params[i]));

   memcpy(sticks.data(), that.sticks.data(), sizeof(double)*N);
   memcpy(k2z.data(), that.k2z.data(), sizeof(int)*N);
   memcpy(z2k.data(), that.z2k.data(), sizeof(int)*N);
   memcpy(likelihoodOld.data(), that.likelihoodOld.data(), sizeof(double)*N);
   memcpy(likelihoodDelta.data(), that.likelihoodDelta.data(), sizeof(double)*N);
   splittable = that.splittable;

   linkedListNode<int>* node = alive.getFirst();
   while (node!=NULL)
   {
      alive_ptrs[node->getData()] = node;
      node = node->getNext();
   }

   rand_gen = initialize_gsl_rand(mx_rand());

   probsArr.resize(Nthreads);
   rArr.resize(Nthreads);
   for (int t=0; t<Nthreads; t++)
   {
      probsArr[t].resize(N);
      rArr[t] = initialize_gsl_rand(mx_rand());
   }

   useSuperclusters = that.useSuperclusters;
   always_splittable = that.always_splittable;
}

// --------------------------------------------------------------------------
// -- clusters
// --   constructor; intializes to all the values given
// --------------------------------------------------------------------------
clusters::clusters(int _N, int _D, double* _data, double* _phi,
   double _alpha, niw_sampled &_hyper, int _Nthreads, bool _useSuperclusters,
   bool _always_splittable) :
   N(_N), D(_D), Nthreads(_Nthreads),
   data(_data), phi(_phi), hyper(_hyper), alpha(_alpha), 
   always_splittable(_always_splittable),
   useSuperclusters(_useSuperclusters),
   superclusters(0), supercluster_labels(0), supercluster_labels_count(0)
{
   D2 = D*D;
   params.assign(N,NULL);
   sticks.resize(N);
   k2z.assign(N,-1);
   z2k.assign(N,-1);
   alive_ptrs.assign(N,NULL);
   likelihoodOld.resize(N);
   likelihoodDelta.resize(N);
   splittable.resize(N);

   randomSplitIndex.resize(N);

   logalpha = log(alpha);

   rand_gen = initialize_gsl_rand(mx_rand());

   probsArr.resize(Nthreads);
   rArr.resize(Nthreads);
   for (int t=0; t<Nthreads; t++)
   {
      probsArr[t].resize(N);
      rArr[t] = initialize_gsl_rand(mx_rand());
   }
}


// --------------------------------------------------------------------------
// -- ~clusters
// --   destructor
// --------------------------------------------------------------------------
clusters::~clusters()
{
   if (N>0)
   {
      for (int i=0; i<N; i++)
         if (params[i]!=NULL)
            delete params[i];
   }
   if (Nthreads>0)
   {
      for (int t=0; t<Nthreads; t++)
         if (rArr[t]!=NULL) gsl_rng_free(rArr[t]);
   }

   gsl_rng_free(rand_gen);
}


// --------------------------------------------------------------------------
// -- initialize
// --   populates the initial statistics
// --------------------------------------------------------------------------
void clusters::initialize()
{
   // assume that the phi's are compressed between [0,K-1)
   K = 1;
   for (int k=0; k<K; k++)
   {
     // indices of the instantiated clusters
      int m = k;
      // unique ids of alive clusters
      alive_ptrs[m] = alive.addNodeEnd(m);
      params[m] = new cluster_sampledT<niw_sampled>(hyper,alpha);

      sticks[m] = log(0.9) ; // anything just allocate space
      double* mu =  new double[D];
      double* mu_l = new double[D];
      double* mu_r = new double[D];
      double* Sigma = new double[D2];
      double* Sigma_l = new double[D2]; 
      double* Sigma_r = new double[D2]; 
      likelihoodOld[m] = -1000.0;
      likelihoodDelta[m] = -1000.0;
      splittable[m] = true;

      params[m]->get_params()->set_normal(mu, Sigma);
      params[m]->get_paramsl()->set_normal(mu_l, Sigma_l);
      params[m]->get_paramsr()->set_normal(mu_r, Sigma_r);
   }
   populate_k2z_z2k();

   // calculate the initial statistics since they aren't explicitly stored
   cout<<"   calculate statistics over N="<<N<<" D="<<D<<" K="<<K<<endl;
   reduction_array<int> NArr(Nthreads, K*2, 0);
   reduction_array2<double> tArr(Nthreads, K*2, D, 0);
   reduction_array2<double> TArr(Nthreads, K*2, D2, 0);
   #pragma omp parallel for
   for (int i=0; i<N; i++)
   {
      int m = phi[i];
      int k = z2k[m];
      //double tlogp_lr, tlogpl_lr, tlogpr_lr;
      int proc = omp_get_thread_num();

//      cout<<i<<" "<<endl;
      int bin = k*2 + (int)(2*(phi[i] - m));
      // accumulate the stats
      NArr.reduce_inc(proc, bin);
      tArr.reduce_add(proc, bin, data+(i*D));
      TArr.reduce_add_outerprod(proc, bin, data+i*D);
//      cout<<"i="<<i<<" proc="<<proc<<endl;
   }
   int* fNArr = NArr.final_reduce_add();
   double* ftArr = tArr.final_reduce_add();
   double* fTArr = TArr.final_reduce_add();
   for (int k=0; k<K; k++)
   {
      int m = k2z[k];
      params[m]->get_paramsl()->set_stats(fNArr[k*2],   ftArr+(k*D*2),   fTArr+(k*D2*2));
      params[m]->get_paramsr()->set_stats(fNArr[k*2+1], ftArr+(k*D*2)+D, fTArr+(k*D2*2)+D2);
      params[m]->update_upwards_posterior();
   }
}

//void clusters::initialize(const mxArray* cluster_params)
//{
//   // assume that the phi's are compressed between [0,K-1)
//   K = mxGetNumberOfElements(cluster_params);
//   for (int k=0; k<K; k++)
//   {
//     // indices of the instantiated clusters
//      int m = getInput<double>(mxGetField(cluster_params,k,"z"));
//      // unique ids of alive clusters
//      alive_ptrs[m] = alive.addNodeEnd(m);
//      params[m] = new cluster_sampledT<niw_sampled>(hyper,alpha);
//
//      sticks[m] = getInput<double>(mxGetField(cluster_params,k,"logpi"));
//      double* mu = getArrayInput<double>(mxGetField(cluster_params,k,"mu"));
//      double* mu_l = getArrayInput<double>(mxGetField(cluster_params,k,"mu_l"));
//      double* mu_r = getArrayInput<double>(mxGetField(cluster_params,k,"mu_r"));
//      double* Sigma = getArrayInput<double>(mxGetField(cluster_params,k,"Sigma"));
//      double* Sigma_l = getArrayInput<double>(mxGetField(cluster_params,k,"Sigma_l"));
//      double* Sigma_r = getArrayInput<double>(mxGetField(cluster_params,k,"Sigma_r"));
//      likelihoodOld[m] = getInput<double>(mxGetField(cluster_params,k,"logsublikelihood"));
//      likelihoodDelta[m] = getInput<double>(mxGetField(cluster_params,k,"logsublikelihoodDelta"));
//      splittable[m] = getInput<bool>(mxGetField(cluster_params,k,"splittable"));
//
//      params[m]->get_params()->set_normal(mu, Sigma);
//      params[m]->get_paramsl()->set_normal(mu_l, Sigma_l);
//      params[m]->get_paramsr()->set_normal(mu_r, Sigma_r);
//   }
//   populate_k2z_z2k();
//
//   // calculate the initial statistics since they aren't explicitly stored
//   reduction_array<int> NArr(Nthreads, K*2, 0);
//   reduction_array2<double> tArr(Nthreads, K*2, D, 0);
//   reduction_array2<double> TArr(Nthreads, K*2, D2, 0);
//   #pragma omp parallel for
//   for (int i=0; i<N; i++)
//   {
//      int m = phi[i];
//      int k = z2k[m];
//      double tlogp_lr, tlogpl_lr, tlogpr_lr;
//      int proc = omp_get_thread_num();
//
//      int bin = k*2 + (int)(2*(phi[i] - m));
//      // accumulate the stats
//      NArr.reduce_inc(proc, bin);
//      tArr.reduce_add(proc, bin, data+(i*D));
//      TArr.reduce_add_outerprod(proc, bin, data+i*D);
//   }
//   int* fNArr = NArr.final_reduce_add();
//   double* ftArr = tArr.final_reduce_add();
//   double* fTArr = TArr.final_reduce_add();
//   for (int k=0; k<K; k++)
//   {
//      int m = k2z[k];
//      params[m]->get_paramsl()->set_stats(fNArr[k*2],   ftArr+(k*D*2),   fTArr+(k*D2*2));
//      params[m]->get_paramsr()->set_stats(fNArr[k*2+1], ftArr+(k*D*2)+D, fTArr+(k*D2*2)+D2);
//      params[m]->update_upwards_posterior();
//   }
//}

// --------------------------------------------------------------------------
// -- write_output
// --   creates and writes the output cluster structure
// --------------------------------------------------------------------------
//void clusters::write_output(mxArray* &plhs)
//{
//   linkedListNode<int>* node = alive.getFirst();
//   populate_k2z_z2k();
//
//   /*#pragma omp parallel for
//   for (int i=0; i<N; i++)
//   {
//      int m = phi[i];
//      int k = z2k[m];
//      phi[i] += k-m;
//   }*/
//
//   const char* names[11] = {"z","logpi","mu","mu_l","mu_r","Sigma","Sigma_l","Sigma_r","logsublikelihood","logsublikelihoodDelta","splittable"};
//   plhs = mxCreateStructMatrix(K,1,11,names);
//   mxArray* cluster_params = plhs;
//   for (int k=0; k<K; k++)
//   {
//      int m = k2z[k];
//      if (params[m]->getN()==0)
//         mexErrMsgTxt("empty cluster\n");
//      mxWriteField(cluster_params, k, "z",     mxCreateScalar((double)m));
//      mxWriteField(cluster_params, k, "logpi", mxCreateScalar(sticks[m]));
//
//      mxWriteField(cluster_params, k, "mu"   , mxCreateArray(D,1,params[m]->get_params()->get_mean()));
//      mxWriteField(cluster_params, k, "mu_l" , mxCreateArray(D,1,params[m]->get_paramsl()->get_mean()));
//      mxWriteField(cluster_params, k, "mu_r" , mxCreateArray(D,1,params[m]->get_paramsr()->get_mean()));
//
//      mxWriteField(cluster_params, k, "Sigma"   , mxCreateArray(D,D,params[m]->get_params()->get_cov()));
//      mxWriteField(cluster_params, k, "Sigma_l" , mxCreateArray(D,D,params[m]->get_paramsl()->get_cov()));
//      mxWriteField(cluster_params, k, "Sigma_r" , mxCreateArray(D,D,params[m]->get_paramsr()->get_cov()));
//
//      mxWriteField(cluster_params, k, "logsublikelihood",      mxCreateScalar(likelihoodOld[m]));
//      mxWriteField(cluster_params, k, "logsublikelihoodDelta", mxCreateScalar(likelihoodDelta[m]));
//      mxWriteField(cluster_params, k, "splittable",            mxCreateScalar(splittable[m]));
//   }
//}

// --------------------------------------------------------------------------
// -- populate_k2z_z2k
// --   Populates the k and z mappings
// --------------------------------------------------------------------------
void clusters::populate_k2z_z2k()
{
   K = alive.getLength();
   linkedListNode<int>* node = alive.getFirst();
   for (int k=0; k<K; k++)
   {
      int m = node->getData();
      k2z[k] = m;
      z2k[m] = k;
      node = node->getNext();
   }
}

// --------------------------------------------------------------------------
// -- sample_params
// --   Samples the parameters for each cluster and the mixture weights
// --------------------------------------------------------------------------
void clusters::sample_params()
{
   // populate the mapping and the stats for the dirichlet
   //linkedListNode<int>* node = alive.getFirst();
   populate_k2z_z2k();
   for (int k=0; k<K; k++)
   {
      int m = k2z[k];
      sticks[m] = params[m]->getN();
   }

//   cout<<"omp_get_thread_num()="<<omp_get_thread_num()<<endl;
//   cout<<"#threads"<<Nthreads<<endl;
   // posterior
   // sample the cluster parameters and the gamma distributions
   double total = 0;
   #pragma omp parallel for reduction(+:total)
   for (int k=0; k<K; k++)
   {
      int m = k2z[k];
      params[m]->sample_param();
      sticks[m] = gsl_ran_gamma(rArr[omp_get_thread_num()], sticks[m], 1);
      total += sticks[m];
   }

   // store the log of the stick lenghts
   for (int k=0; k<K; k++)
   {
      int m = k2z[k];
      sticks[m] = log(sticks[m]) - log(total);
   }
}


// --------------------------------------------------------------------------
// -- sample_superclusters
// --   Samples the supercluster assignments
// --------------------------------------------------------------------------
void clusters::sample_superclusters()
{
   if (!useSuperclusters)
      return;

   superclusters.assign(K,-1);
   supercluster_labels.assign(K*K,-1);
   supercluster_labels_count.assign(K,0);
   
   if (K<=3)
   {
      for (int k=0; k<K; k++)
      {
         superclusters[k] = 0;
         supercluster_labels[k] = k;
      }
      supercluster_labels_count[0] = K;
      return;
   }

   vector<double> adjMtx(K*K);
   #pragma omp parallel for
   for (int k=0; k<K*K; k++)
   {
      if (k%K!=k/K)
      {
         int k1 = k%K;
         int k2 = k/K;
         if (k1>k2)
         {
            int m1 = k2z[k1];
            int m2 = k2z[k2];
            adjMtx[k2*K+k1] = 1.0/exp(4*params[m1]->Jdivergence(params[m2]));
            //adjMtx[k2*K+k1] = 1.0/(params[m1]->KLdivergence(params[m2]) + params[m2]->KLdivergence(params[m1]));
            adjMtx[k1*K+k2] = adjMtx[k2*K+k1];
         }
      }
      else
         adjMtx[k] = 0;
   }

   vector<linkedList<int> > neighbors(K);
   double* tprobabilities = probsArr[0].data();
   for (int k=0; k<K; k++)
   {
      double totalProb = 0;
      for (int k2=0; k2<K; k2++)
      {
         double temp = adjMtx[k+k2*K] + adjMtx[k2+k*K];
         tprobabilities[k2] = temp;
         totalProb += temp;
      }
      int k2 = sample_categorical(tprobabilities, K, totalProb, rArr[0]);
      neighbors[k].addNodeEnd(k2);
      neighbors[k2].addNodeEnd(k);
   }

   vector<bool> done(K,false);
   int count = 0;
   for (int k=0; k<K; k++) if (!done[k])
   {
      int label_count = 0;
      done[k] = true;
      superclusters[k] = count;
      int* supercluster_labels_k = supercluster_labels.data()+count*K;
      supercluster_labels_k[label_count++] = (k);
      while (!neighbors[k].isempty())
      {
         int k1 = neighbors[k].popFirst();
         if (!done[k1])
         {
            done[k1] = true;
            superclusters[k1] = count;
            supercluster_labels_k[label_count++] = (k1);
            neighbors[k].merge_with(neighbors[k1]);
         }
      }
      supercluster_labels_count[count] = label_count;
      count++;
   }
   //mexPrintf("%d/%d\n", count,K);
}

// --------------------------------------------------------------------------
// -- sample_labels
// --   Samples the label assignments for each data point
// --------------------------------------------------------------------------
void clusters::sample_labels()
{
   linkedListNode<int>* node;

   // =========================================================================
   // propose local label changes within superclusters
   // =========================================================================

   // temporary reduction array variables
   reduction_array<int> NArr(Nthreads, K*2, 0);
   reduction_array2<double> tArr(Nthreads, K*2, D, 0);
   reduction_array2<double> TArr(Nthreads, K*2, D2, 0);
   reduction_array<double> likelihoodArr(Nthreads, K, 0);

//   cout<<"N="<<N<<" D="<<D<<" K="<<K<<endl;
   // loop through points and sample labels
   int k0 = z2k[(int)phi[0]];
   #pragma omp parallel for
   for (int i=0; i<N; i++)
   {
      int proc = omp_get_thread_num();
      double* tprobabilities = probsArr[proc].data();
      double tphi;
      double phii = phi[i];
      int k;
      // find the distribution over possible ones
      if (K==1)
      {
         k = k0;
//         cout<<"k="<<k<<endl;
      }
      else if (useSuperclusters)
      {
         double maxProb = -mxGetInf();
         int ki = z2k[(int)phii];
         int sci = superclusters[ki];
         for (int k2i=0; k2i<supercluster_labels_count[sci]; k2i++)
         {
            int k2 = supercluster_labels[sci*K + k2i];
            int m = k2z[k2];

            // find the probability of the data belonging to this component
            double prob = params[m]->get_params()->predictive_loglikelihood(data+i*D) + sticks[m];
            maxProb = std::max(maxProb, prob);
            tprobabilities[k2i] = prob;
         }

         // sample a new cluster label
         double totalProb = total_logcategorical(tprobabilities, supercluster_labels_count[sci], maxProb);
         int k2i = sample_logcategorical(tprobabilities, supercluster_labels_count[sci], maxProb, totalProb, rArr[proc]);
         k = supercluster_labels[sci*K+k2i];
      }
      else
      {
         double maxProb = -mxGetInf();
//         cout<<maxProb<<endl;
         for (int k2=0; k2<K; k2++)
         {
            int m = k2z[k2];

            // find the probability of the data belonging to this component
            double prob = params[m]->get_params()->predictive_loglikelihood(data+i*D) + sticks[m];
            maxProb = std::max(maxProb, prob);
            tprobabilities[k2] = prob;
         }

         // sample a new cluster label
         double totalProb = total_logcategorical(tprobabilities, K, maxProb);
         k = sample_logcategorical(tprobabilities, K, maxProb, totalProb, rArr[proc]);
      }
      int m = k2z[k];

      double loglikelihood;
      double logpl = params[m]->get_paramsl()->predictive_loglikelihood(data+i*D);
      double logpr = params[m]->get_paramsr()->predictive_loglikelihood(data+i*D);
      tphi = params[m]->sample_subcluster_label(logpl, logpr, rArr[proc], loglikelihood);

      // update changes
      likelihoodArr.reduce_add(proc, k, loglikelihood);

      // update stats
      phi[i] = m + tphi;

      // accumulate
      int bin = k*2 + (int)(tphi*2);
      NArr.reduce_inc(proc, bin);
      tArr.reduce_add(proc, bin, data+(i*D));
      TArr.reduce_add_outerprod(proc, bin, data+i*D);
   }

   // accumulate cluster statistics
//   cout<<" accumulate cluster statistics"<<endl;
   arr(int) fNArr = NArr.final_reduce_add();
   arr(double) ftArr = tArr.final_reduce_add();
   arr(double) fTArr = TArr.final_reduce_add();
   arr(double) flikelihood = likelihoodArr.final_reduce_add();
   for (int k=0; k<K; k++)
   {
      int m = k2z[k];
      params[m]->empty();
      params[m]->get_paramsl()->set_stats(fNArr[k*2],   ftArr+(k*D*2),   fTArr+(k*D2*2));
      params[m]->get_paramsr()->set_stats(fNArr[k*2+1], ftArr+(k*D*2)+D, fTArr+(k*D2*2)+D2);
      params[m]->update_upwards_posterior();

      double newlikelihoodDelta = (flikelihood[k]/params[m]->getN()) - likelihoodOld[m];
      if (newlikelihoodDelta<0)// && likelihoodDelta[m]<0)
         splittable[m] = true;
      likelihoodDelta[m] = newlikelihoodDelta;
      likelihoodOld[m] = (flikelihood[k]/params[m]->getN());
   }

   // get rid of dead nodes
   bool deleted = false;
   node = alive.getFirst();

   while (node!=NULL)
   {
      int m = node->getData();
      node = node->getNext();
      if (params[m]->isempty())
      {
         alive.deleteNode(alive_ptrs[m]);
         alive_ptrs[m] = NULL;
         delete params[m];
         params[m] = NULL;
         deleted = true;
      }
   }
   if (deleted)
   {
      populate_k2z_z2k();
      // resample superclusters because the mapping is all broken
      sample_superclusters();
   }
}

// --------------------------------------------------------------------------
// -- propose_merges
// --   Samples the label assignments for each data point
// --------------------------------------------------------------------------
void clusters::propose_merges()
{
   vector<int> merge_with(K, -1);
   int numMerges = 0;
   for (int km=0; km<K; km++) if (splittable[k2z[km]])
   {
      int zm = k2z[km];
      if (merge_with[km]<0)
         for (int kn=km+1; kn<K; kn++) if (splittable[k2z[kn]])
         {
            int zn = k2z[kn];
            if ((!useSuperclusters || superclusters[km]==superclusters[kn]) && merge_with[km]<0 && merge_with[kn]<0)
            {
               int Nm = params[zm]->getN();
               int Nn = params[zn]->getN();
               int Nkh = Nm+Nn;

               cluster_sampledT<niw_sampled>* paramsm = params[zm];
               cluster_sampledT<niw_sampled>* paramsn = params[zn];

               double HR = -logalpha - myloggamma(Nm) - myloggamma(Nn) + myloggamma(Nkh);
               HR += paramsm->data_loglikelihood_marginalized_testmerge(paramsn);
               HR += -paramsm->data_loglikelihood_marginalized() - paramsn->data_loglikelihood_marginalized();
               //HR += -paramsm->logmu_prior() - paramsn->logmu_prior() - paramsm->data_loglikelihood() - paramsn->data_loglikelihood();

               if ((HR>0 || my_rand(rand_gen) < exp(HR)))
               {
                  merge_with[km] = km;
                  merge_with[kn] = km;
                  numMerges++;

                  // move on to the next one
                  break;
               }
            }
         }
   }
   if (numMerges>0)
   {
//      mexPrintf("NumMerges=%d\n",numMerges);
      // fix the phi's
      #pragma omp parallel for
      for (int i=0; i<N; i++)
      {
         int zi = phi[i];
         int ki = z2k[zi];
         if (merge_with[ki]>=0)
            phi[i] = k2z[merge_with[ki]] + ((merge_with[ki]==ki) ? 0.25 : 0.75);
      }
      for (int kn=0; kn<K; kn++) if (merge_with[kn]>=0 && merge_with[kn]!=kn)
      {
         int zn = k2z[kn];
         int km = merge_with[kn];
         int zm = k2z[km];

         // mark them as done
         merge_with[km] = -1;
         merge_with[kn] = -1;

         splittable[zm] = false;
         likelihoodOld[zm] = -mxGetInf();
         likelihoodDelta[zm] = mxGetInf();

         params[zm]->merge_with(params[zn]);
         params[zm]->setstickl(sticks[zm]);
         params[zm]->setstickr(sticks[zn]);

         sticks[zm] = logsumexp(sticks[zm], sticks[zn]);
         sticks[zn] = -mxGetInf();

         // sample a new set of parameters for the highest level
         params[zm]->sample_param_regular();

         alive.deleteNode(alive_ptrs[zn]);
         alive_ptrs[zn] = NULL;
         delete params[zn];
         params[zn] = NULL;
         numMerges--;
         if (numMerges==0)
            break;
      }
      populate_k2z_z2k();
   }
}



// --------------------------------------------------------------------------
// -- propose_splits
// --   Samples the label assignments for each data point
// --------------------------------------------------------------------------
void clusters::propose_splits()
{
   vector<bool> do_split(K, false);
   vector<bool> do_reset(K, false);
   int num_splits = 0;
   int num_resets = 0;
   //#pragma omp parallel for reduction(+:num_splits)
   //mexPrintf("====================\n");
   for (int k=0; k<K; k++)
   {
      // check to see if the subclusters have converged
      if (always_splittable || splittable[k2z[k]])//paramsChanges[k2z[k]]/((double)params[k2z[k]]->getN())<=0.1)
      {
         int proc = omp_get_thread_num();
         int kz = k2z[k];
         cluster_sampledT<niw_sampled>* paramsk = params[kz];
         int Nk = paramsk->getN();
         int Nmh = paramsk->getlN();
         int Nnh = paramsk->getrN();

         if (Nmh==0 || Nnh==0)
         {
            do_reset[k] = true;
            num_resets++;
            continue;
         }

         // full ratios
         double HR = logalpha + myloggamma(Nmh) + myloggamma(Nnh) - myloggamma(Nk);
         HR += paramsk->data_lloglikelihood_marginalized() + paramsk->data_rloglikelihood_marginalized();
         HR += -paramsk->data_loglikelihood_marginalized();
         //HR += paramsk->logmul_prior() + paramsk->logmur_prior() - paramsk->logmu_prior();
         //HR += paramsk->data_lloglikelihood() + paramsk->data_rloglikelihood() - paramsk->data_loglikelihood();
         //mexPrintf("HR=%e\n", HR);

         if ((HR>0 || my_rand(rArr[proc]) < exp(HR)))
         {
            do_split[k] = true;
            num_splits++;
         }
      }
   }


   if (num_resets>0)
   {
      // correct the labels
      reduction_array<int> NArr(Nthreads, K*2, 0);
      reduction_array2<double> tArr(Nthreads, K*2, D, 0);
      reduction_array2<double> TArr(Nthreads, K*2, D2, 0);

      #pragma omp parallel for
      for (int i=0; i<N; i++) if (do_reset[z2k[(int)(phi[i])]])
      {
         int proc = omp_get_thread_num();
         int kz = phi[i];
         int newk = z2k[kz];
         double tphi = (my_rand(rArr[proc])<0.5) ? 0.25 : 0.75;
         phi[i] = kz + tphi;

         // updates stats
         int bin = newk*2 + (int)(tphi*2);
         NArr.reduce_inc(proc, bin);
         tArr.reduce_add(proc, bin, data+(i*D));
         TArr.reduce_add_outerprod(proc, bin, data+i*D);
      }

      // accumulate cluster statistics
      int* fNArr = NArr.final_reduce_add();
      double* ftArr = tArr.final_reduce_add();
      double* fTArr = TArr.final_reduce_add();
      for (int k=0; k<K; k++) if (do_reset[k])
      {
         int m = k2z[k];
         params[m]->empty();
         params[m]->get_paramsl()->set_stats(fNArr[k*2],   ftArr+(k*D*2),   fTArr+(k*D2*2));
         params[m]->get_paramsr()->set_stats(fNArr[k*2+1], ftArr+(k*D*2)+D, fTArr+(k*D2*2)+D2);
         params[m]->update_upwards_posterior();
         params[m]->sample_param_subclusters();
      }
   }


   if (num_splits>0)
   {
//      mexPrintf("NumSplits=%d\n", num_splits);
      int temp_num_splits = num_splits;
      // figure out which labels to split into and prepare simple split stuff
      vector<int> split_into(K);
      vector<int> split_into_k(K);
      int k = 0;
      for (k=0; k<K; k++)
         if (do_split[k])
            break;
      for (int nzh=0; nzh<N; nzh++)
      {
         if (params[nzh]==NULL)
         {
            // found an empty one
            int kz = k2z[k];
            split_into[k] = nzh;
            split_into_k[k] = K+(num_splits-temp_num_splits);

            // mark it as unsplittable for now
            splittable[kz] = false;
            likelihoodOld[kz] = -mxGetInf();
            likelihoodDelta[kz] = mxGetInf();
            splittable[nzh] = false;
            likelihoodOld[nzh] = -mxGetInf();
            likelihoodDelta[nzh] = mxGetInf();

            alive_ptrs[nzh] = alive.addNodeEnd(nzh);
            params[nzh] = new cluster_sampledT<niw_sampled>(hyper,alpha);
            temp_num_splits--;

            // sample a new set of pi's
            sticks[kz] = params[kz]->getstickl();
            sticks[nzh] = params[kz]->getstickr();

            // transfer over the theta's
            params[nzh]->split_from(params[kz]);
            params[kz]->split_fix();
            params[nzh]->update_subclusters_sample();
            params[kz]->update_subclusters_sample();

            // do the next ones if there are any
            if (temp_num_splits>0)
            {
               for (k=k+1; k<K; k++)
                  if (do_split[k])
                     break;
            }
            else
               break;
         }
      }

      // correct the labels
      reduction_array<int> NArr(Nthreads, (K+num_splits)*2, 0);
      reduction_array2<double> tArr(Nthreads, (K+num_splits)*2, D, 0);
      reduction_array2<double> TArr(Nthreads, (K+num_splits)*2, D2, 0);

      #pragma omp parallel for
      for (int i=0; i<N; i++) if (do_split[z2k[(int)(phi[i])]])
      {
         int proc = omp_get_thread_num();
         int kz = phi[i];
         int newk;
         if (phi[i]-kz<0.5)
         {
            phi[i] = kz;
            newk = z2k[kz];
         }
         else
         {
            phi[i] = split_into[z2k[kz]];
            newk = split_into_k[z2k[kz]];
         }
         double tphi = (my_rand(rArr[proc])<0.5) ? 0.25 : 0.75;
         phi[i] += tphi;

         // updates stats
         int bin = newk*2 + (int)(tphi*2);
         NArr.reduce_inc(proc, bin);
         tArr.reduce_add(proc, bin, data+(i*D));
         TArr.reduce_add_outerprod(proc, bin, data+i*D);
      }

      // accumulate cluster statistics
      arr(int) fNArr = NArr.final_reduce_add();
      arr(double) ftArr = tArr.final_reduce_add();
      arr(double) fTArr = TArr.final_reduce_add();
      for (int k=0; k<K; k++) if (do_split[k])
      {
         int m = k2z[k];
         params[m]->empty();
         params[m]->get_paramsl()->set_stats(fNArr[k*2],   ftArr+(k*D*2),   fTArr+(k*D2*2));
         params[m]->get_paramsr()->set_stats(fNArr[k*2+1], ftArr+(k*D*2)+D, fTArr+(k*D2*2)+D2);
         params[m]->update_upwards_posterior();
         params[m]->sample_param_subclusters();

         m = split_into[k];
         int newk = split_into_k[k];
         params[m]->empty();
         params[m]->get_paramsl()->set_stats(fNArr[newk*2],   ftArr+(newk*D*2),   fTArr+(newk*D2*2));
         params[m]->get_paramsr()->set_stats(fNArr[newk*2+1], ftArr+(newk*D*2)+D, fTArr+(newk*D2*2)+D2);
         params[m]->update_upwards_posterior();
         params[m]->sample_param_subclusters();
      }
      populate_k2z_z2k();
   }
}




void clusters::propose_random_split_assignments()
{
   // temporary reduction array variables
   reduction_array<int> NArr(Nthreads, K*2, 0);
   reduction_array2<double> tArr(Nthreads, K*2, D, 0);
   reduction_array2<double> TArr(Nthreads, K*2, D2, 0);

   // sample new pi's for the splits
   // store the actual stick breaks for now... will convert to log later
   #pragma omp parallel for
   for (int k=0; k<K; k++)
   {
      int m = k2z[k];
      double stickm = gsl_ran_gamma(rArr[omp_get_thread_num()], 0.5*alpha, 1);
      double stickn = gsl_ran_gamma(rArr[omp_get_thread_num()], 0.5*alpha, 1);
      double total = stickm + stickn;
      stickm /= total;
      stickn /= total;
      params[m]->setrandomstickl(stickm);
      params[m]->setrandomstickr(stickn);
   }

   // loop through points and sample labels
   #pragma omp parallel for schedule(guided)
   for (int i=0; i<N; i++)
   {
      int proc = omp_get_thread_num();
      int m = phi[i];
      int k = z2k[m];
      
      int ri = my_rand(rArr[proc]) < params[m]->getrandomstickl();
      randomSplitIndex[i] = ri;

      // updates stats
      int bin = k*2 + (int)ri;
      NArr.reduce_inc(proc, bin);
      tArr.reduce_add(proc, bin, data+(i*D));
      TArr.reduce_add_outerprod(proc, bin, data+i*D);
   }

   // accumulate cluster statistics
   // convert randomsticks to log(randomsticks)
   arr(int) fNArr = NArr.final_reduce_add();
   arr(double) ftArr = tArr.final_reduce_add();
   arr(double) fTArr = TArr.final_reduce_add();
   for (int k=0; k<K; k++)
   {
      int m = k2z[k];
      params[m]->get_randoml()->set_stats(fNArr[k*2], ftArr+(k*D*2), fTArr+(k*D2*2));
      params[m]->get_randomr()->set_stats(fNArr[k*2+1], ftArr+(k*D*2)+D, fTArr+(k*D2*2)+D2);
      params[m]->update_random_posteriors();
      // convert to log
      params[m]->setrandomstickl( log(params[m]->getrandomstickl()) );
      params[m]->setrandomstickr( log(params[m]->getrandomstickr()) );
   }
}


void clusters::propose_random_splits()
{
   propose_random_split_assignments();

   vector<bool> do_split(K, false);
   int num_splits = 0;
   //#pragma omp parallel for reduction(+:num_splits)
   for (int k=0; k<K; k++)
   {
      int proc = omp_get_thread_num();
      int kz = k2z[k];
      cluster_sampledT<niw_sampled>* paramsk = params[kz];
      int Nk = paramsk->getN();

      // full ratios
      double HR = logalpha + 2*gsl_sf_lngamma(0.5*alpha) - gsl_sf_lngamma(alpha) + gsl_sf_lngamma(alpha + Nk) - gammalnint(Nk) + log(100.0);
      HR += paramsk->data_randomlloglikelihood_marginalized() + paramsk->data_randomrloglikelihood_marginalized();
      HR += -paramsk->data_loglikelihood_marginalized();

      if ((HR>0 || my_rand(rArr[proc]) < exp(HR)))
      {
         do_split[k] = true;
         num_splits++;
      }
   }
   if (num_splits>0)
   {
      //mexPrintf("NumRandomSplits=%d\n", num_splits);
      int temp_num_splits = num_splits;
      // figure out which labels to split into and prepare simple split stuff
      vector<int> split_into(K);
      vector<int> split_into_k(K);
      int k = 0;
      for (k=0; k<K; k++)
         if (do_split[k])
            break;
      for (int nzh=0; nzh<N; nzh++)
      {
         if (params[nzh]==NULL)
         {
            // found an empty one
            int kz = k2z[k];
            split_into[k] = nzh;
            split_into_k[k] = K+(num_splits-temp_num_splits);

            // mark it as unsplittable for now
            splittable[kz] = false;
            likelihoodOld[kz] = -mxGetInf();
            likelihoodDelta[kz] = mxGetInf();
            splittable[nzh] = false;
            likelihoodOld[nzh] = -mxGetInf();
            likelihoodDelta[nzh] = mxGetInf();

            alive_ptrs[nzh] = alive.addNodeEnd(nzh);
            params[nzh] = new cluster_sampledT<niw_sampled>(hyper,alpha);
            temp_num_splits--;

            // sample a new set of pi's
            sticks[kz] = params[kz]->getrandomstickl();
            sticks[nzh] = params[kz]->getrandomstickr();

            // transfer over the theta's
            params[nzh]->split_from_random(params[kz]);
            params[kz]->split_fix_random();
            params[nzh]->update_subclusters_sample();
            params[kz]->update_subclusters_sample();

            // do the next ones if there are any
            if (temp_num_splits>0)
            {
               for (k=k+1; k<K; k++)
                  if (do_split[k])
                     break;
            }
            else
               break;
         }
      }

      // correct the labels
      // temporary reduction array variables
      reduction_array<int> NArr(Nthreads, (K+num_splits)*2, 0);
      reduction_array2<double> tArr(Nthreads, (K+num_splits)*2, D, 0);
      reduction_array2<double> TArr(Nthreads, (K+num_splits)*2, D2, 0);


      #pragma omp parallel for schedule(guided)
      for (int i=0; i<N; i++) if (do_split[z2k[(int)(phi[i])]])
      {
         int proc = omp_get_thread_num();
         int kz = phi[i];
         int newk;
         if (!randomSplitIndex[i])
         {
            phi[i] = kz;
            newk = z2k[kz];
         }
         else
         {
            phi[i] = split_into[z2k[kz]];
            newk = split_into_k[z2k[kz]];
         }
         double tphi = (my_rand(rArr[proc])<0.5) ? 0.25 : 0.75;
         phi[i] += tphi;

         // updates stats
         int bin = newk*2 + (int)(tphi*2);
         NArr.reduce_inc(proc, bin);
         tArr.reduce_add(proc, bin, data+(i*D));
         TArr.reduce_add_outerprod(proc, bin, data+i*D);
      }

      // accumulate cluster statistics
      arr(int) fNArr = NArr.final_reduce_add();
      arr(double) ftArr = tArr.final_reduce_add();
      arr(double) fTArr = TArr.final_reduce_add();
      for (int k=0; k<K; k++) if (do_split[k])
      {
         int m = k2z[k];
         params[m]->empty();
         params[m]->get_paramsl()->set_stats(fNArr[k*2], ftArr+(k*D*2), fTArr+(k*D2*2));
         params[m]->get_paramsr()->set_stats(fNArr[k*2+1], ftArr+(k*D*2)+D, fTArr+(k*D2*2)+D2);
         params[m]->update_upwards_posterior();
         params[m]->sample_param_subclusters();

         m = split_into[k];
         int newk = split_into_k[k];
         params[m]->empty();
         params[m]->get_paramsl()->set_stats(fNArr[newk*2], ftArr+(newk*D*2), fTArr+(newk*D2*2));
         params[m]->get_paramsr()->set_stats(fNArr[newk*2+1], ftArr+(newk*D*2)+D, fTArr+(newk*D2*2)+D2);
         params[m]->update_upwards_posterior();
         params[m]->sample_param_subclusters();
      }
      populate_k2z_z2k();
   }
}

void clusters::propose_random_merges()
{
   vector<int> merge_with(K, -1);
   int numMerges = 0;
   for (int km=0; km<K; km++)
   {
      int zm = k2z[km];
      if (merge_with[km]<0)
         for (int kn=km+1; kn<K; kn++)
         {
            int zn = k2z[kn];
            if ((!useSuperclusters || superclusters[km]==superclusters[kn]) && merge_with[km]<0 && merge_with[kn]<0)
            {
               cluster_sampledT<niw_sampled>* paramsm = params[zm];
               cluster_sampledT<niw_sampled>* paramsn = params[zn];

               int Nm = paramsm->getN();
               int Nn = paramsn->getN();
               int Nkh = Nm+Nn;

               double HR = -logalpha + gsl_sf_lngamma(alpha) - 2*gsl_sf_lngamma(0.5*alpha) + gammalnint(Nkh) - gsl_sf_lngamma(Nkh+alpha) - log(100.0);
               HR += paramsm->data_loglikelihood_marginalized_testmerge(paramsn);
               HR += -paramsm->data_loglikelihood_marginalized() - paramsn->data_loglikelihood_marginalized();
               //mexPrintf("%e\n", HR);
               //mexPrintf("%d\t%d\tHR merge = %e \t %e\n", Nm,Nn,HR, paramsm->data_loglikelihood_marginalized_testmerge(paramsn)-paramsm->data_loglikelihood_marginalized()-paramsn->data_loglikelihood_marginalized());

               if ((HR>0 || my_rand(rand_gen) < exp(HR)))
               {
                  merge_with[km] = km;
                  merge_with[kn] = km;
                  numMerges++;

                  // move on to the next one
                  break;
               }
            }
         }
   }
   if (numMerges>0)
   {
      //mexPrintf("NumRandomMerges=%d\n",numMerges);
      // fix the phi's
      #pragma omp parallel for
      for (int i=0; i<N; i++)
      {
         int zi = phi[i];
         int ki = z2k[zi];
         if (merge_with[ki]>=0)
            phi[i] = k2z[merge_with[ki]] + ((merge_with[ki]==ki) ? 0.25 : 0.75);
      }
      for (int kn=0; kn<K; kn++) if (merge_with[kn]>=0 && merge_with[kn]!=kn)
      {
         int zn = k2z[kn];
         int km = merge_with[kn];
         int zm = k2z[km];

         // mark them as done
         merge_with[km] = -1;
         merge_with[kn] = -1;

         splittable[zm] = false;
         likelihoodOld[zm] = -mxGetInf();
         likelihoodDelta[zm] = mxGetInf();
         params[zm]->merge_with(params[zn]);
         params[zm]->setstickl(sticks[zm]);
         params[zm]->setstickr(sticks[zn]);

         sticks[zm] = logsumexp(sticks[zm], sticks[zn]);
         sticks[zn] = -mxGetInf();

         // sample a new set of parameters for the highest level
         //params[zm]->sample_highest();

         alive.deleteNode(alive_ptrs[zn]);
         alive_ptrs[zn] = NULL;
         delete params[zn];
         params[zn] = NULL;
         numMerges--;
         if (numMerges==0)
            break;
      }
      //mexPrintf("aa\n");drawnow();
      populate_k2z_z2k();
   }
}


double clusters::joint_loglikelihood()
{
   linkedListNode<int>* node = alive.getFirst();
   double loglikelihood = gsl_sf_lngamma(alpha) - gsl_sf_lngamma(alpha+N);
   while (node!=NULL)
   {
      int m = node->getData();
      if (!params[m]->isempty())
         loglikelihood += logalpha + gsl_sf_lngamma(params[m]->getN()) + params[m]->data_loglikelihood_marginalized();
      node = node->getNext();
   }
   return loglikelihood;
}



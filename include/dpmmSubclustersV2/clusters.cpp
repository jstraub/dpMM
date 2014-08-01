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

#include "clusters.h"
#include "reduction_array.h"
#include "reduction_array2.h"
#include "linear_algebra.h"
#include "sample_categorical.h"

#ifndef pi
#define pi 3.14159265
#endif

// --------------------------------------------------------------------------
// -- clusters
// --   constructor; initializes to empty
// --------------------------------------------------------------------------
clusters::clusters() :
   N(0), params(0), sticks(0), k2z(0), z2k(0), Nthreads(0), probsArr(0),
   rArr(0), superclusters(0), supercluster_labels(0), supercluster_labels_count(0),
   likelihoodOld(0), likelihoodDelta(0), splittable(0), randomSplitIndex(0)
{
   D2 = D*D;
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
         params[i] = new cluster_sampledT<normalmean_sampled>(*(that.params[i]));

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

   maxK = that.maxK;
}

// --------------------------------------------------------------------------
// -- clusters
// --   constructor; intializes to all the values given
// --------------------------------------------------------------------------
clusters::clusters(int _N, int _D, int _X, int _Y, double* _data, double* _phi,
   normalmean_sampled &_hyper, const mxArray* inparams) :
   N(_N), D(_D), X(_X), Y(_Y), data(_data), phi(_phi), hyper(_hyper),
   superclusters(0), supercluster_labels(0), supercluster_labels_count(0),
   meanImage(_N*_D, 0), meanSubImage(_N*_D, 0)
{
   alpha = getInput<double>(getField(inparams,0,"alpha"));
   logalpha = log(alpha);
   Nthreads = getInput<double>(getField(inparams,0,"Mproc"));
   useSuperclusters = getInput<bool>(getField(inparams,0,"useSuperclusters"));
   always_splittable = getInput<bool>(getField(inparams,0,"always_splittable"));
   maxK = getInput<double>(getField(inparams,0,"maxK"));
   closestIndices = getArrayInput<unsigned long>(getField(inparams,0,"closestIndices"));
   omask = getArrayInput<bool>(getField(inparams,0,"omask"));

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


   rand_gen = initialize_gsl_rand(mx_rand());

   probsArr.resize(Nthreads);
   rArr.resize(Nthreads);
   for (int t=0; t<Nthreads; t++)
   {
      probsArr[t].resize(N);
      rArr[t] = initialize_gsl_rand(mx_rand());
   }


   // get the blur kernel stuff
   const mxArray* _blurKernels = getField(inparams,0,"blurKernels");
   const mxArray* _blurKernelOthers = getField(inparams,0,"blurKernelOthers");
   int numBlurKernels = mxGetNumberOfElements(_blurKernels);
   blurKernels.resize(numBlurKernels);
   blurKernelOthers.resize(numBlurKernels);
   for (int b=0; b<numBlurKernels; b++)
   {
      const mxArray* _blurKernel = mxGetCell(_blurKernels, b);
      int L = mxGetNumberOfElements(_blurKernel);
      arr(double) _blurKernelArr = getArrayInput<double>(_blurKernel);

      const mxArray* _blurKernelOther = mxGetCell(_blurKernelOthers, b);
      arr(double) _blurKernelOtherArr = getArrayInput<double>(_blurKernelOther);

      blurKernels[b].resize(L);
      blurKernelOthers[b].resize(L);
      for (int l=0; l<L; l++)
      {
         blurKernels[b][l] = _blurKernelArr[l];
         blurKernelOthers[b][l] = _blurKernelOtherArr[l];
      }
   }
   blurKernelNum = getInput<double>(getField(inparams,0,"blurKernelNum"));

   MRFw = getInput<double>(getField(inparams,0,"MRFw"));
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
void clusters::initialize(const mxArray* cluster_params)
{
   // assume that the phi's are compressed between [0,K-1)
   K = mxGetNumberOfElements(cluster_params);
   for (int k=0; k<K; k++)
   {
      int m = getInput<double>(mxGetField(cluster_params,k,"z"));
      alive_ptrs[m] = alive.addNodeEnd(m);
      params[m] = new cluster_sampledT<normalmean_sampled>(hyper,alpha);

      sticks[m] = getInput<double>(mxGetField(cluster_params,k,"logpi"));
      double* mu = getArrayInput<double>(mxGetField(cluster_params,k,"mu"));
      double* mu_l = getArrayInput<double>(mxGetField(cluster_params,k,"mu_l"));
      double* mu_r = getArrayInput<double>(mxGetField(cluster_params,k,"mu_r"));
      double* Sigma = getArrayInput<double>(mxGetField(cluster_params,k,"Sigma"));
      double* Sigma_l = getArrayInput<double>(mxGetField(cluster_params,k,"Sigma_l"));
      double* Sigma_r = getArrayInput<double>(mxGetField(cluster_params,k,"Sigma_r"));
      likelihoodOld[m] = getInput<double>(mxGetField(cluster_params,k,"logsublikelihood"));
      likelihoodDelta[m] = getInput<double>(mxGetField(cluster_params,k,"logsublikelihoodDelta"));
      splittable[m] = getInput<bool>(mxGetField(cluster_params,k,"splittable"));

      params[m]->get_params()->set_normal(mu, Sigma);
      params[m]->get_paramsl()->set_normal(mu_l, Sigma_l);
      params[m]->get_paramsr()->set_normal(mu_r, Sigma_r);
   }
   populate_k2z_z2k();

   // calculate the initial statistics since they aren't explicitly stored
   reduction_array<int> NArr(Nthreads, K*2, 0);
   reduction_array2<double> tArr(Nthreads, K*2, D, 0);
   reduction_array2<double> TArr(Nthreads, K*2, D2, 0);
   #pragma omp parallel for
   for (int i=0; i<N; i++) if (omask[i])
   {
      int m = phi[i];
      int k = z2k[m];
      double tlogp_lr, tlogpl_lr, tlogpr_lr;
      int proc = omp_get_thread_num();

      int bin = k*2 + (int)(2*(phi[i] - m));
      // accumulate the stats
      NArr.reduce_inc(proc, bin);
      tArr.reduce_add(proc, bin, data+(i*D));
      TArr.reduce_add_outerprod(proc, bin, data+i*D);
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

// --------------------------------------------------------------------------
// -- write_output
// --   creates and writes the output cluster structure
// --------------------------------------------------------------------------
void clusters::write_output(mxArray* &plhs, mxArray* &plhs2, mxArray* &plhs3)
{
   linkedListNode<int>* node = alive.getFirst();
   populate_k2z_z2k();

   /*#pragma omp parallel for
   for (int i=0; i<N; i++)
   {
      int m = phi[i];
      int k = z2k[m];
      phi[i] += k-m;
   }*/

   const char* names[12] = {"z","logpi","mu","mu_l","mu_r","Sigma","Sigma_l","Sigma_r","logsublikelihood","logsublikelihoodDelta","splittable","logpz"};
   plhs = mxCreateStructMatrix(K,1,12,names);
   mxArray* cluster_params = plhs;
   for (int k=0; k<K; k++)
   {
      int m = k2z[k];
      if (params[m]->getN()==0)
         mexErrMsgTxt("empty cluster\n");
      mxWriteField(cluster_params, k, "z",     mxCreateScalar((double)m));
      mxWriteField(cluster_params, k, "logpi", mxCreateScalar(sticks[m]));

      mxWriteField(cluster_params, k, "mu"   , mxCreateArray(D,1,params[m]->get_params()->get_mean()));
      mxWriteField(cluster_params, k, "mu_l" , mxCreateArray(D,1,params[m]->get_paramsl()->get_mean()));
      mxWriteField(cluster_params, k, "mu_r" , mxCreateArray(D,1,params[m]->get_paramsr()->get_mean()));

      mxWriteField(cluster_params, k, "Sigma"   , mxCreateArray(D,D,params[m]->get_params()->get_cov()));
      mxWriteField(cluster_params, k, "Sigma_l" , mxCreateArray(D,D,params[m]->get_paramsl()->get_cov()));
      mxWriteField(cluster_params, k, "Sigma_r" , mxCreateArray(D,D,params[m]->get_paramsr()->get_cov()));

      mxWriteField(cluster_params, k, "logsublikelihood",      mxCreateScalar(likelihoodOld[m]));
      mxWriteField(cluster_params, k, "logsublikelihoodDelta", mxCreateScalar(likelihoodDelta[m]));
      mxWriteField(cluster_params, k, "splittable",            mxCreateScalar((bool)splittable[m]));
      //mexPrintf("Writing splittable: %d\n", (int)splittable[m]);

      mxWriteField(cluster_params, k, "logpz",                 mxCreateScalar(params[m]->getlogpz()));
   }

   // modify the blurkernel in inparams directly
   plhs2 = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
   //mexPrintf("Length=%d\n", mxGetNumberOfElements(getField(inparams,0,"blurKernelNum")));
   //arr(double) blurKernelNumPtr = getArrayInput<double>(getField(inparams,0,"blurKernelNum"));
   /*mexPrintf("b=%e\n",blurKernelNumPtr[0]);
   blurKernelNumPtr[0] = blurKernelNum;
   mexPrintf("b=%e\n",blurKernelNumPtr[0]);*/
   arr(double) blurKernelNumPtr = getArrayInput<double>(plhs2);
   blurKernelNumPtr[0] = blurKernelNum;


   plhs3 = mxCreateNumericMatrix(K, K, mxDOUBLE_CLASS, mxREAL);
   arr(double) logpzMtxPtr = getArrayInput<double>(plhs3);
   for (int i=0; i<K*K; i++)
      logpzMtxPtr[i] = logpzMtx[i];
}

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
   populate_k2z_z2k();

   // sample the covariance
   sample_sigma();

   // populate the mapping and the stats for the dirichlet
   for (int k=0; k<K; k++)
   {
      int m = k2z[k];
      sticks[m] = params[m]->getN();
   }

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
   sample_mu();

   /*//mexPrintf("============ SAMPLE PARAMS\n");
   for (int k=0; k<K; k++)
   {
      int m = k2z[k];
      arr(double) mu = params[m]->get_params()->get_mean();
      for (int d=0; d<D; d++)
         mexPrintf("%e\t", mu[d]);
      mexPrintf("------------------\n");
   }*/

   // store the log of the stick lenghts
   for (int k=0; k<K; k++)
   {
      int m = k2z[k];
      sticks[m] = log(sticks[m]) - log(total);
   }
}

void clusters::sample_mu()
{
   int KD = K*D;
   int KD2 = KD*KD;
   vector<double> pLambda(KD2,0);
   vector<double> pLambdatheta(KD,0);
   arr(double) precmu = hyper.get_invDeltah();
   arr(double) precx = hyper.get_prec();
   arr(double) theta = hyper.get_thetah();

   vector<double> tempSpace(N);

   // populate the weights
   vector< vector<double> > weights(K);
   for (int k=0; k<K; k++)
   {
      weights[k].resize(N);
      fill(weights[k].begin(), weights[k].end(), 0);
   }
   for (int i=0; i<N; i++)
   {
      int ci = (omask[i] ? i : closestIndices[i]);
      int m = phi[ci];
      int k = z2k[m];
      weights[k][i] = 1;
   }

   // blur them
   for (int k=0; k<K; k++)
      imfilterSep(weights[k].data(), X, Y, blurKernels[blurKernelNum].data(), blurKernels[blurKernelNum].size(), tempSpace.data());

   // find the sum of weights and sum of cross weights
   //vector<double> sumw(KD,0); // sum_i w_i^k sum_n x_in-g_in * precx_mn
   //vector<double> sumww(K*K,0);

   reduction_array<double> sumwArr(Nthreads, KD, 0);
   reduction_array<double> sumwwArr(Nthreads, K*K, 0);
   #pragma omp parallel for schedule(guided)
   for (int i=0; i<N; i++) if (omask[i])
   {
      int proc = omp_get_thread_num();
      for (int k=0; k<K; k++)
      {
         double wk = weights[k][i];
         if (wk>0)
         {
            for (int m=0; m<D; m++)
            {
               double xhat = 0;
               for (int n=0; n<D; n++)
                  xhat += data[i*D+n] * precx[m*D+n];
               sumwArr.reduce_add(proc, k*D+m, wk * xhat);
               //sumw[k*D+m] += wk * xhat;
            }
            sumwwArr.reduce_add(proc, k*K+k, wk * wk);
            //sumww[k*K+k] += wk * wk;
            for (int l=k+1; l<K; l++)
            {
               sumwwArr.reduce_add(proc, k*K+l, wk * weights[l][i]);
               //sumww[k*K+l] += wk * weights[l][i];
            }
         }
      }
   }
   arr(double) sumw = sumwArr.final_reduce_add();
   arr(double) sumww = sumwwArr.final_reduce_add();
   for (int k=0; k<K; k++)
      for (int l=k+1; l<K; l++)
         sumww[l*K+k] = sumww[k*K+l];

   // populate the posterior hyperparameters
   vector<double> Lambdatheta(D,0);
   for (int m=0; m<D; m++)
      for (int n=0; n<D; n++)
         Lambdatheta[m] += precmu[m*D+n]*theta[n];
   for (int k=0; k<K; k++) for (int m=0; m<D; m++)
   {
      int kmi = k*D+m;
      for (int n=0; n<D; n++)
      {
         int kni = k*D+n;
         pLambda[kmi*KD + kni] = precmu[m*D+n] + precx[m*D+n]*sumww[k*K+k];
         for (int l=k+1; l<K; l++)
         {
            int lni = l*D+n;
            pLambda[kmi*KD + lni] = precx[m*D+n]*sumww[k*K+l];
            pLambda[lni*KD + kmi] = pLambda[kmi*KD + lni];
         }

         pLambdatheta[kmi] = Lambdatheta[m] + sumw[kmi];
      }
   }

   // find the mean
   vector<double> pSigma(pLambda);
   InvMat(pSigma.data(), KD);
   vector<double> ptheta(KD,0);
   for (int kmi=0; kmi<KD; kmi++)
      for (int lni=0; lni<KD; lni++)
         ptheta[kmi] += pSigma[kmi*KD+lni]*pLambdatheta[lni];

   /*for (int k=0; k<K; k++)
   {
      arr(double) mu = params[k2z[k]]->get_params()->get_mean();
      for (int d=0; d<D; d++)
         mexPrintf("%e\t", mu[d]);
   }
   mexPrintf("\n");
   for (int kmi=0; kmi<KD; kmi++)
      mexPrintf("%e\t", ptheta[kmi]);
   mexPrintf("\n");*/

   // sample a mean
   // do the cholesky decomposition
   Eigen::Map<Eigen::MatrixXd> _Delta(pSigma.data(), KD, KD);
   Eigen::LLT<Eigen::MatrixXd> myLlt(_Delta);
   Eigen::MatrixXd chol = myLlt.matrixL();

   vector<double> mu(KD);

   // populate the mean
   for (int kmi=0; kmi<KD; kmi++)
      mu[kmi] = gsl_ran_gaussian(rArr[0],1);
   Eigen::Map<Eigen::VectorXd> emean(mu.data(), KD);
   emean = chol*emean;
   for (int kmi=0; kmi<KD; kmi++)
      mu[kmi] += ptheta[kmi];

   /*for (int kmi=0; kmi<KD; kmi++)
   {
      for (int lni=0; lni<KD; lni++)
         mexPrintf("%e ", chol(kmi,lni));
      mexPrintf("\n");
   }

   mexPrintf("==============\n");
   for (int k=0; k<K; k++)
   {
      arr(double) muk = params[k2z[k]]->get_params()->get_mean();
      for (int d=0; d<D; d++)
         mexPrintf("%e\t", muk[d]);
   }
   mexPrintf("\n");
   for (int kmi=0; kmi<KD; kmi++)
      mexPrintf("%e\t", ptheta[kmi]);
   mexPrintf("\n");
   for (int kmi=0; kmi<KD; kmi++)
      mexPrintf("%e\t", mu[kmi]);
   mexPrintf("\n");

   mexPrintf("----------\n");
   for (int kmi=0; kmi<KD; kmi++)
   {
      for (int lni=0; lni<KD; lni++)
         mexPrintf("%e ", pLambda[kmi*KD+lni]);
      mexPrintf("\n");
   }*/

   // set the means
   for (int k=0; k<K; k++)
   {
      int m = k2z[k];
      params[m]->get_params()->set_mean(mu.data()+k*D);
   }
}

void clusters::sample_sigma()
{
   int numSigmas = hyper.getNumSigmas();
   reduction_array<double> loglikelihoodsArr(Nthreads, numSigmas, 0);
   #pragma omp parallel for
   for (int k=0; k<K; k++)
   {
      int proc = omp_get_thread_num();
      int m = k2z[k];

      for (int numSigma=0; numSigma<numSigmas; numSigma++)
      {
         double loglikelihood = params[m]->get_params()->data_loglikelihood_marginalized(numSigma);
         loglikelihoodsArr.reduce_add(proc, numSigma, loglikelihood);
      }
   }

   arr(double) floglikelihoodsArr = loglikelihoodsArr.final_reduce_add();
   double totalProb = convert_logcategorical(floglikelihoodsArr, numSigmas);
   numSigma = sample_categorical(floglikelihoodsArr, numSigmas, totalProb, rArr[0]);
   setSigmas(numSigma);
}




// --------------------------------------------------------------------------
// -- sample_params
// --   Samples the parameters for each cluster and the mixture weights
// --------------------------------------------------------------------------
void clusters::max_params()
{
   populate_k2z_z2k();

   // sample the covariance
   max_sigma();

   // populate the mapping and the stats for the dirichlet
   for (int k=0; k<K; k++)
   {
      int m = k2z[k];
      sticks[m] = params[m]->getN();
   }

   // posterior
   // sample the cluster parameters and the gamma distributions
   double total = 0;
   #pragma omp parallel for reduction(+:total)
   for (int k=0; k<K; k++)
   {
      int m = k2z[k];
      params[m]->max_param();
      total += sticks[m];
   }

   // store the log of the stick lenghts
   for (int k=0; k<K; k++)
   {
      int m = k2z[k];
      sticks[m] = log(sticks[m]) - log(total);
   }
}

void clusters::max_sigma()
{
   int numSigmas = hyper.getNumSigmas();
   reduction_array<double> loglikelihoodsArr(Nthreads, numSigmas, 0);
   #pragma omp parallel for
   for (int k=0; k<K; k++)
   {
      int proc = omp_get_thread_num();
      int m = k2z[k];

      for (int numSigma=0; numSigma<numSigmas; numSigma++)
      {
         double loglikelihood = params[m]->get_params()->data_loglikelihood_marginalized(numSigma);
         loglikelihoodsArr.reduce_add(proc, numSigma, loglikelihood);
      }
   }

   arr(double) floglikelihoodsArr = loglikelihoodsArr.final_reduce_add();
   numSigma = max_categorical(floglikelihoodsArr, numSigmas);
   setSigmas(numSigma);
}


void clusters::setSigmas(int numSigma)
{
   for (int k=0; k<K; k++)
   {
      int m = k2z[k];
      params[m]->get_params()->setSigmas(numSigma);
      params[m]->get_paramsl()->setSigmas(numSigma);
      params[m]->get_paramsr()->setSigmas(numSigma);
      params[m]->get_randoml()->setSigmas(numSigma);
      params[m]->get_randomr()->setSigmas(numSigma);
   }
   hyper.setSigmas(numSigma);
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




void clusters::sample_blurkernel()
{
   if (blurKernels.size()<=1)
      return;
   //populate_k2z_z2k();
//mexPrintf("a");drawnow();
   populate_meanImage();
   // calculate the loglikelihoods for each blur kernel
//mexPrintf("b");drawnow();
   vector<double> blurredImage(N*D,0);
   vector<double> tempSpace(N*D,0);
   vector<double> zero_mean(D,0);

   double maxProb = -mxGetInf();
//mexPrintf("====================\n");
   arr(double) tprobsArr = probsArr[0].data();
   for (int b=0; b<blurKernels.size(); b++)
   {
      // blur the image
      imfilterColorSep(meanImage.data(), X, Y, blurKernels[b].data(), blurKernels[b].size(), tempSpace.data(), blurredImage.data());

      // calculate the likelihood
      int NArr = 0;
      reduction_array2<double> tArr(Nthreads, 1, D, 0);
      reduction_array2<double> TArr(Nthreads, 1, D2, 0);
      #pragma omp parallel for schedule(guided) reduction(+:NArr)
      for (int i=0; i<N; i++) if (omask[i])
      {
         int m = phi[i];
         int k = z2k[m];
         double tlogp_lr, tlogpl_lr, tlogpr_lr;
         int proc = omp_get_thread_num();

         // accumulate the stats
         NArr++;
         tArr.reduce_add_diff(proc, 0, data+(i*D), blurredImage.data()+(i*D));
         TArr.reduce_add_outerprod_diff(proc, 0, data+i*D, blurredImage.data()+(i*D));
      }
      double* ftArr = tArr.final_reduce_add();
      double* fTArr = TArr.final_reduce_add();

      normalmean_sampled temp(hyper);
      temp.set_mean(zero_mean.data());
      temp.set_stats(NArr, ftArr, fTArr);
      tprobsArr[b] = temp.data_loglikelihood();
      maxProb = max(maxProb, tprobsArr[b]);
      arr(double) s = temp.get_cov();
      //mexPrintf("b=%d \t logp=%e \t %e %e %e %e %e %e\n",b,tprobsArr[b], ftArr[0], ftArr[1], ftArr[2], mu[0], mu[1], mu[2]);
      //mexPrintf("b=%d \t logp=%e \t %e %e %e \n",b,tprobsArr[b], s[0], s[1], s[2]);
   }
//mexPrintf("c");drawnow();
   double totalProb = total_logcategorical(tprobsArr, blurKernels.size(), maxProb);
   blurKernelNum = sample_logcategorical(tprobsArr, blurKernels.size(), maxProb, totalProb, rArr[0]);
   //mexPrintf("b=%d\n", blurKernelNum);
//mexPrintf("d");drawnow();
   // store the conditional distribution in meanImage
   imfilterColorSep(meanImage.data(), X, Y, blurKernels[blurKernelNum].data(), blurKernels[blurKernelNum].size(), tempSpace.data());
   #pragma omp parallel for schedule(guided)
   for (int i=0; i<N; i++)
   {
      if (omask[i])
      {
         for (int d=0; d<D; d++)
            meanImage[i*D+d] = data[i*D+d] - meanImage[i*D+d];
      }
      else
      {
         int ci = closestIndices[i];
         for (int d=0; d<D; d++)
            meanImage[i*D+d] = data[ci*D+d] - meanImage[i*D+d];
      }
   }
//mexPrintf("e");drawnow();
   imfilterColorSep(meanImage.data(), X, Y, blurKernels[blurKernelNum].data(), blurKernels[blurKernelNum].size(), tempSpace.data());
//mexPrintf("f\n");drawnow();
}

void clusters::populate_meanDifference()
{
   populate_meanImage();
   populate_meanSubImage();
   // calculate the loglikelihoods for each blur kernel

   vector<double> tempSpace(N*D,0);

   // store the conditional distribution in meanImage
   imfilterColorSep(meanImage.data(), X, Y, blurKernels[blurKernelNum].data(), blurKernels[blurKernelNum].size(), tempSpace.data());
   imfilterColorSep(meanSubImage.data(), X, Y, blurKernels[blurKernelNum].data(), blurKernels[blurKernelNum].size(), tempSpace.data());
   #pragma omp parallel for schedule(guided)
   for (int i=0; i<N; i++)
   {
      if (omask[i])
      {
         for (int d=0; d<D; d++)
         {
            meanImage[i*D+d] = data[i*D+d] - meanImage[i*D+d];
            meanSubImage[i*D+d] = data[i*D+d] - meanSubImage[i*D+d];
         }
      }
      else
      {
         int ci = closestIndices[i];
         for (int d=0; d<D; d++)
         {
            meanImage[i*D+d] = data[ci*D+d] - meanImage[i*D+d];
            meanSubImage[i*D+d] = data[ci*D+d] - meanSubImage[i*D+d];
         }
      }
   }
   imfilterColorSep(meanImage.data(), X, Y, blurKernels[blurKernelNum].data(), blurKernels[blurKernelNum].size(), tempSpace.data());
   imfilterColorSep(meanSubImage.data(), X, Y, blurKernels[blurKernelNum].data(), blurKernels[blurKernelNum].size(), tempSpace.data());

   // extend phi
   // extend it out
   for (int i=0; i<N; i++) if (!omask[i])
      phi[i] = phi[closestIndices[i]];
}

void clusters::max_blurkernel()
{
   //populate_k2z_z2k();

   populate_meanImage();
   // calculate the loglikelihoods for each blur kernel

   vector<double> blurredImage(N*D,0);
   vector<double> tempSpace(N*D,0);
   vector<double> zero_mean(D,0);

   double maxProb = -mxGetInf();
//mexPrintf("====================\n");
   arr(double) tprobsArr = probsArr[0].data();
   for (int b=0; b<blurKernels.size(); b++)
   {
      // blur the image
      imfilterColorSep(meanImage.data(), X, Y, blurKernels[b].data(), blurKernels[b].size(), tempSpace.data(), blurredImage.data());

      // calculate the likelihood
      int NArr = 0;
      reduction_array2<double> tArr(Nthreads, 1, D, 0);
      reduction_array2<double> TArr(Nthreads, 1, D2, 0);
      #pragma omp parallel for reduction(+:NArr) schedule(guided)
      for (int i=0; i<N; i++) if (omask[i])
      {
         int m = phi[i];
         int k = z2k[m];
         double tlogp_lr, tlogpl_lr, tlogpr_lr;
         int proc = omp_get_thread_num();

         // accumulate the stats
         NArr++;
         tArr.reduce_add_diff(proc, 0, data+(i*D), blurredImage.data()+(i*D));
         TArr.reduce_add_outerprod_diff(proc, 0, data+i*D, blurredImage.data()+(i*D));
      }
      double* ftArr = tArr.final_reduce_add();
      double* fTArr = TArr.final_reduce_add();

      normalmean_sampled temp(hyper);
      temp.set_mean(zero_mean.data());
      temp.set_stats(NArr, ftArr, fTArr);
      tprobsArr[b] = temp.data_loglikelihood();
      maxProb = max(maxProb, tprobsArr[b]);
      arr(double) s = temp.get_cov();
      //mexPrintf("b=%d \t logp=%e \t %e %e %e %e %e %e\n",b,tprobsArr[b], ftArr[0], ftArr[1], ftArr[2], mu[0], mu[1], mu[2]);
      //mexPrintf("b=%d \t logp=%e \t %e %e %e \n",b,tprobsArr[b], s[0], s[1], s[2]);
   }
   blurKernelNum = max_categorical(tprobsArr, blurKernels.size());
   //mexPrintf("b=%d\n", blurKernelNum);

   // store the conditional distribution in meanImage + the middle term
   imfilterColorSep(meanImage.data(), X, Y, blurKernels[blurKernelNum].data(), blurKernels[blurKernelNum].size(), tempSpace.data());
}


void clusters::populate_meanImage()
{
   #pragma omp parallel for schedule(guided)
   for (int i=0; i<N; i++) if (omask[i])
   {
      int m = phi[i];
      int k = z2k[m];
      //if (params[m]==NULL)
         //mexErrMsgTxt("uhoh");
      arr(double) mu = params[m]->get_params()->get_mean();
      for (int d=0; d<D; d++)
         meanImage[i*D+d] = mu[d];
   }
   // extend it out
   #pragma omp parallel for schedule(guided)
   for (int i=0; i<N; i++) if (!omask[i])
   {
      int ci = closestIndices[i];
      for (int d=0; d<D; d++)
         meanImage[i*D+d] = meanImage[ci*D+d];
   }
}
void clusters::populate_meanSubImage()
{
   #pragma omp parallel for schedule(guided)
   for (int i=0; i<N; i++) if (omask[i])
   {
      int m = phi[i];
      int k = z2k[m];
      arr(double) mu;
      if (phi[i]-m-0.5<0)
         mu = params[m]->get_paramsl()->get_mean();
      else
         mu = params[m]->get_paramsr()->get_mean();
      for (int d=0; d<D; d++)
         meanSubImage[i*D+d] = mu[d];
   }
   // extend it out
   #pragma omp parallel for schedule(guided)
   for (int i=0; i<N; i++) if (!omask[i])
   {
      int ci = closestIndices[i];
      for (int d=0; d<D; d++)
         meanSubImage[i*D+d] = meanSubImage[ci*D+d];
   }
}

void clusters::copy_meanImage()
{
   mexPrintf("============\n");
   for (int k=0; k<K; k++)
   {
      int m = k2z[k];
      arr(double) mu = params[m]->get_params()->get_mean();
      for (int d=0; d<D; d++)
         mexPrintf("%e\t", mu[d]);
      mexPrintf("\n");
      mu = params[m]->get_paramsl()->get_mean();
      for (int d=0; d<D; d++)
         mexPrintf("%e\t", mu[d]);
      mexPrintf("\n");
      mu = params[m]->get_paramsr()->get_mean();
      for (int d=0; d<D; d++)
         mexPrintf("%e\t", mu[d]);
      mexPrintf("\n");
      mexPrintf("------------------\n");
   }
   for (int i=0; i<N; i++)
   {
      phi[i] = meanImage[i*D];
   }
}

// --------------------------------------------------------------------------
// -- sample_labels
// --   Samples the label assignments for each data point
// --------------------------------------------------------------------------
void clusters::sample_labels()
{
   linkedListNode<int>* node;

   // =================================================================================
   // propose local label changes within superclusters
   // =================================================================================

   // temporary reduction array variables
   reduction_array<int> NArr(Nthreads, K*2, 0);
   reduction_array2<double> tArr(Nthreads, K*2, D, 0);
   reduction_array2<double> TArr(Nthreads, K*2, D2, 0);
   reduction_array<double> likelihoodArr(Nthreads, K, 0);

   reduction_array<double> tempData(Nthreads, D, 0);

   //arr(double) blurKernel = blurKernels[blurKernelNum].data();
   //int blurKernelL = blurKernels[blurKernelNum].size();
   double blurScale = 1;//blurKernel[blurKernelL/2];

   // loop through points and sample labels
   int k0 = z2k[(int)phi[0]];
   #pragma omp parallel for schedule(guided)
   for (int i=0; i<N; i++) if (omask[i])
   {
      int proc = omp_get_thread_num();
      double* tprobabilities = probsArr[proc].data();
      double tphi;
      double phii = phi[i];
      int k;

      arr(double) ttempdata = tempData.getThreadArray(proc);
      memcpy(ttempdata, data+i*D, sizeof(double)*D);
      for (int d=0; d<D; d++)
         ttempdata[d] = (ttempdata[d]-meanImage[i*D+d]) / blurScale;

      // find the distribution over possible ones
      if (K==1)
      {
         k = k0;
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
            //double prob = params[m]->get_params()->predictive_loglikelihood(data+i*D) + sticks[m];
            double prob = params[m]->get_params()->predictive_loglikelihood(ttempdata) + sticks[m];
            maxProb = max(maxProb, prob);
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
         for (int k2=0; k2<K; k2++)
         {
            int m = k2z[k2];

            // find the probability of the data belonging to this component
            //double prob = params[m]->get_params()->predictive_loglikelihood(data+i*D) + sticks[m];
            double prob = params[m]->get_params()->predictive_loglikelihood(ttempdata) + sticks[m];
            maxProb = max(maxProb, prob);
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
      mexPrintf("Deleted\n");

      //for (int k=0; k<K; k++)
         //mexPrintf("%d %d\n", fNArr[k*2], fNArr[k*2+1]);


      populate_k2z_z2k();
      // resample superclusters because the mapping is all broken
      sample_superclusters();
   }
}


// --------------------------------------------------------------------------
// -- sample_labels2
// --   Samples the label assignments for each data point
// --------------------------------------------------------------------------
void clusters::sample_labels2()
{
   linkedListNode<int>* node;

   // =================================================================================
   // propose local label changes within superclusters
   // =================================================================================

   // temporary reduction array variables
   reduction_array<int> NArr(Nthreads, K*2, 0);
   reduction_array2<double> tArr(Nthreads, K*2, D, 0);
   reduction_array2<double> TArr(Nthreads, K*2, D2, 0);
   reduction_array<double> likelihoodArr(Nthreads, K, 0);
   reduction_array<double> likelihoodzArr(Nthreads, K, 0);
   reduction_array<double> likelihoodzMtx(Nthreads, K*K, 0);


   // loop through points and sample labels
   int k0 = z2k[(int)phi[0]];
   #pragma omp parallel for schedule(guided)
   for (int i=0; i<N; i++) if (omask[i])
   {
      int proc = omp_get_thread_num();
      double* tprobabilities = probsArr[proc].data();
      double tphi;
      double phii = phi[i];
      int k;
      // find the distribution over possible ones
      double maxProb = -mxGetInf();


      if (K==1)
      {
         k = k0;
      }
      else if (useSuperclusters)
      {
         int ki = z2k[(int)phii];
         int sci = superclusters[ki];
         for (int k2i=0; k2i<supercluster_labels_count[sci]; k2i++)
         {
            int k2 = supercluster_labels[sci*K + k2i];
            int m = k2z[k2];

            // find the probability of the data belonging to this component
            double prob = params[m]->get_params()->predictive_loglikelihood(data+i*D) + sticks[m];
            maxProb = max(maxProb, prob);
            tprobabilities[k2i] = prob;
         }

         // sample a new cluster label
         double totalProb = total_logcategorical(tprobabilities, supercluster_labels_count[sci], maxProb);
         int k2i = sample_logcategorical(tprobabilities, supercluster_labels_count[sci], maxProb, totalProb, rArr[proc]);
         k = supercluster_labels[sci*K+k2i];
      }
      else
      {
         for (int k2=0; k2<K; k2++)
         {
            int m = k2z[k2];

            // find the probability of the data belonging to this component
            double prob = params[m]->get_params()->predictive_loglikelihood(data+i*D) + sticks[m];
            //double prob = params[m]->get_params()->predictive_loglikelihood(data+i*D) + sticks[m];
            maxProb = max(maxProb, prob);
            tprobabilities[k2] = prob;
         }

         // sample a new cluster label
         double totalProb = total_logcategorical(tprobabilities, K, maxProb);
         k = sample_logcategorical(tprobabilities, K, maxProb, totalProb, rArr[proc]);
      }
      int m = k2z[k];

      double loglikelihood, lognorm;
      //double logpl = params[m]->get_paramsl()->predictive_loglikelihoodScaleMean(ttempdata, blurScale);
      //double logpr = params[m]->get_paramsr()->predictive_loglikelihoodScaleMean(ttempdata, blurScale);
      double logpl = params[m]->get_paramsl()->predictive_loglikelihood(data+i*D);
      double logpr = params[m]->get_paramsr()->predictive_loglikelihood(data+i*D);
      tphi = params[m]->sample_subcluster_label(logpl, logpr, rArr[proc], loglikelihood, lognorm);

      // update changes
      likelihoodArr.reduce_add(proc, k, loglikelihood);
      likelihoodzArr.reduce_add(proc, k, loglikelihood-lognorm);

      // update stats
      phi[i] = m + tphi;

      // accumulate
      int bin = k*2 + (int)(tphi*2);
      NArr.reduce_inc(proc, bin);
      tArr.reduce_add(proc, bin, data+(i*D));
      TArr.reduce_add_outerprod(proc, bin, data+i*D);

      double pz = tprobabilities[k];
      for (int k2=0; k2<K; k2++)
      {
        // TODO looks like we are just caring about the normalizer?!
         double temp = pz;
         if (k!=k2)
            temp = logsumexp(pz, tprobabilities[k2]);
            //temp = logsumexp(pz, tprobabilities[k2]);
         likelihoodzMtx.reduce_add(proc, k*K + k2, temp);
      }
   }

   arr(double) flikelihoodzArr = likelihoodzArr.final_reduce_add();
   arr(double) flikelihoodzMtx = likelihoodzMtx.final_reduce_add();
   //TODO need sth equivalent
   logpzMtx.resize(K*K);
   memcpy(logpzMtx.data(), flikelihoodzMtx, sizeof(double)*K*K);

   // accumulate cluster statistics
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
      //mexPrintf("splittable[%d]=%d\n", k, (int)splittable[m]);
      likelihoodDelta[m] = newlikelihoodDelta;
      likelihoodOld[m] = (flikelihood[k]/params[m]->getN());

      //TODO need sth equivalent
      params[m]->setlogpz(flikelihoodzArr[k]);
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
      mexPrintf("Deleted\n");
      vector<int> oldz2k(z2k);
      int oldK = K;

      populate_k2z_z2k();
      // resample superclusters because the mapping is all broken
      sample_superclusters();

      vector<double> oldlogpzMtx = logpzMtx;
      logpzMtx.resize(K*K);
      for (int k1=0; k1<K; k1++) for (int k2=0; k2<K; k2++)
      {
         int m1 = k2z[k1];
         int m2 = k2z[k2];
         int oldk1 = oldz2k[m1];
         int oldk2 = oldz2k[m2];
         logpzMtx[k1*K + k2] = oldlogpzMtx[oldk1*oldK + oldk2];
      }
   }
}





// --------------------------------------------------------------------------
// -- sample_labels2
// --   Samples the label assignments for each data point
// --------------------------------------------------------------------------
void clusters::sample_labels3()
{
   linkedListNode<int>* node;

   // =================================================================================
   // propose local label changes within superclusters
   // =================================================================================

   // temporary reduction array variables
   reduction_array<int> NArr(Nthreads, K*2, 0);
   reduction_array2<double> tArr(Nthreads, K*2, D, 0);
   reduction_array2<double> TArr(Nthreads, K*2, D2, 0);
   reduction_array<double> likelihoodArr(Nthreads, K, 0);
   reduction_array<double> likelihoodzArr(Nthreads, K, 0);
   reduction_array<double> likelihoodzMtx(Nthreads, K*K, 0);

   reduction_array<double> tempData(Nthreads, D, 0);
   reduction_array<double> tempSubData(Nthreads, D, 0);

   arr(double) blurKernel = blurKernels[blurKernelNum].data();
   int blurKernelL = blurKernels[blurKernelNum].size();
   double blurSumf2 = 0;
   for (int x=0; x<blurKernelL; x++) for (int y=0; y<blurKernelL; y++)
   {
      double fxy = blurKernels[blurKernelNum][x]*blurKernels[blurKernelNum][y];
      blurSumf2 += fxy*fxy;
   }
   //mexPrintf("blurKernel=%d \t blurKernelL=%d \t blurSumf2=%e\n", blurKernelNum, blurKernelL, blurSumf2);
   blurSumf2 = 1.0/blurSumf2;


   // loop through points and sample labels
   int k0 = z2k[(int)phi[0]];
   for (int it=0; it<17; it++)
   {
      populate_meanDifference();
      #pragma omp parallel for schedule(guided)
      for (int i=0; i<N; i++) if (omask[i])
      {
         int proc = omp_get_thread_num();
         //if ( (int)(my_rand(rArr[proc])*17)!=0 )
         if ( i%17!=it )
            continue;
         double* tprobabilities = probsArr[proc].data();
         double tphi;
         double phii = phi[i];
         int k;
         // find the distribution over possible ones
         double maxProb = -mxGetInf();


         arr(double) ttempdata = tempData.getThreadArray(proc);
         arr(double) curmean = params[(int)phii]->get_params()->get_mean();
         //memcpy(ttempdata, data+i*D, sizeof(double)*D);
         for (int d=0; d<D; d++)
            ttempdata[d] = (meanImage[i*D+d]*blurSumf2 + curmean[d]);

         arr(double) ttempsubdata = tempSubData.getThreadArray(proc);
         int mi = phii;
         arr(double) cursubmean;
         if (phii-mi-0.5<0)
            cursubmean = params[mi]->get_paramsl()->get_mean();
         else
            cursubmean = params[mi]->get_paramsr()->get_mean();
         for (int d=0; d<D; d++)
            ttempsubdata[d] = (meanSubImage[i*D+d]*blurSumf2 + cursubmean[d]);

         /*if (i%10000==0)
         {
            mexPrintf("-----------------\n");
            mexPrintf("%e %e\n", data[i*D], ttempdata[0]/blurScale);
            mexPrintf("%e %e\n", ttempdata[0], curmean[0]*blurScale);
         }*/

         if (K==1)
         {
            k = k0;
         }
         else if (useSuperclusters)
         {
            int ki = z2k[(int)phii];
            int sci = superclusters[ki];
            for (int k2i=0; k2i<supercluster_labels_count[sci]; k2i++)
            {
               int k2 = supercluster_labels[sci*K + k2i];
               int m = k2z[k2];

               // find the probability of the data belonging to this component
               double prob = params[m]->get_params()->predictive_loglikelihoodScalePrec(ttempdata, blurSumf2) + sticks[m];
               maxProb = max(maxProb, prob);
               tprobabilities[k2i] = prob;
            }

            // sample a new cluster label
            double totalProb = total_logcategorical(tprobabilities, supercluster_labels_count[sci], maxProb);
            int k2i = sample_logcategorical(tprobabilities, supercluster_labels_count[sci], maxProb, totalProb, rArr[proc]);
            k = supercluster_labels[sci*K+k2i];
         }
         else
         {
            for (int k2=0; k2<K; k2++)
            {
               int m = k2z[k2];

               // find the probability of the data belonging to this component
               double prob = params[m]->get_params()->predictive_loglikelihoodScalePrec(ttempdata, blurSumf2) + sticks[m];

               if (m!=(int)(phi[i+1]))
                  prob += -MRFw;
               if (m!=(int)(phi[i-1]))
                  prob += -MRFw;
               if (m!=(int)(phi[i+X]))
                  prob += -MRFw;
               if (m!=(int)(phi[i-X]))
                  prob += -MRFw;

               //double prob = params[m]->get_params()->predictive_loglikelihood(data+i*D) + sticks[m];
               maxProb = max(maxProb, prob);
               tprobabilities[k2] = prob;
            }

            // sample a new cluster label
            double totalProb = total_logcategorical(tprobabilities, K, maxProb);
            k = sample_logcategorical(tprobabilities, K, maxProb, totalProb, rArr[proc]);
         }
         int m = k2z[k];

         double loglikelihood, lognorm;
         double logpl = params[m]->get_paramsl()->predictive_loglikelihoodScalePrec(ttempsubdata, blurSumf2);
         double logpr = params[m]->get_paramsr()->predictive_loglikelihoodScalePrec(ttempsubdata, blurSumf2);
         //double logpl = params[m]->get_paramsl()->predictive_loglikelihood(data+i*D);
         //double logpr = params[m]->get_paramsr()->predictive_loglikelihood(data+i*D);

         if (i+1<X && omask[i+1] && m==(int)(phi[i+1]))
         {
            if (phi[i+1]-m-0.5>0)
               logpl += -MRFw/1;
            else
               logpr += -MRFw/1;
         }
         if (i-1>=0 && omask[i-1] && m==(int)(phi[i-1]))
         {
            if (phi[i-1]-m-0.5>0)
               logpl += -MRFw/1;
            else
               logpr += -MRFw/1;
         }
         if (i/X+1<Y && omask[i+X] && m==(int)(phi[i+X]))
         {
            if (phi[i+X]-m-0.5>0)
               logpl += -MRFw/1;
            else
               logpr += -MRFw/1;
         }
         if (i/X-1>=0 && omask[i-X] && m==(int)(phi[i-X]))
         {
            if (phi[i-X]-m-0.5>0)
               logpl += -MRFw/1;
            else
               logpr += -MRFw/1;
         }

         tphi = params[m]->sample_subcluster_label(logpl, logpr, rArr[proc], loglikelihood, lognorm);

         // update changes
         likelihoodArr.reduce_add(proc, k, loglikelihood);
         likelihoodzArr.reduce_add(proc, k, loglikelihood-lognorm);

         // update stats
         phi[i] = m + tphi;

         // accumulate
         int bin = k*2 + (int)(tphi*2);
         NArr.reduce_inc(proc, bin);
         tArr.reduce_add(proc, bin, data+(i*D));
         TArr.reduce_add_outerprod(proc, bin, data+i*D);

         double pz = tprobabilities[k];
         for (int k2=0; k2<K; k2++)
         {
            double temp = pz;
            if (k!=k2)
               temp = logsumexp(pz, tprobabilities[k2]);
               //temp = logsumexp(pz, tprobabilities[k2]);
            likelihoodzMtx.reduce_add(proc, k*K + k2, temp);
         }
      }
   }

   /*#pragma omp parallel for schedule(guided)
   for (int i=0; i<N; i++) if (omask[i])
   {
      int proc = omp_get_thread_num();
      int m = phi[i];
      int k = z2k[m];
      int bin = k*2 + (int)((phi[i]-m)*2);
      NArr.reduce_inc(proc, bin);
      tArr.reduce_add(proc, bin, data+(i*D));
      TArr.reduce_add_outerprod(proc, bin, data+i*D);
   }*/

   arr(double) flikelihoodzArr = likelihoodzArr.final_reduce_add();
   arr(double) flikelihoodzMtx = likelihoodzMtx.final_reduce_add();
   logpzMtx.resize(K*K);
   memcpy(logpzMtx.data(), flikelihoodzMtx, sizeof(double)*K*K);

   // accumulate cluster statistics
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
      //mexPrintf("k=%d \t %e %e\n", k, likelihoodOld[m], flikelihood[k]/params[m]->getN());
      //mexPrintf("splittable[%d]=%d\n", k, (int)splittable[m]);
      likelihoodDelta[m] = newlikelihoodDelta;
      likelihoodOld[m] = (flikelihood[k]/params[m]->getN());

      params[m]->setlogpz(flikelihoodzArr[k]);
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
      mexPrintf("Deleted\n");
      vector<int> oldz2k(z2k);
      int oldK = K;

      //for (int k=0; k<K; k++)
         //mexPrintf("%d %d\n", fNArr[k*2], fNArr[k*2+1]);

      populate_k2z_z2k();
      // resample superclusters because the mapping is all broken
      sample_superclusters();

      vector<double> oldlogpzMtx = logpzMtx;
      logpzMtx.resize(K*K);
      for (int k1=0; k1<K; k1++) for (int k2=0; k2<K; k2++)
      {
         int m1 = k2z[k1];
         int m2 = k2z[k2];
         int oldk1 = oldz2k[m1];
         int oldk2 = oldz2k[m2];
         logpzMtx[k1*K + k2] = oldlogpzMtx[oldk1*oldK + oldk2];
      }
   }
}





// --------------------------------------------------------------------------
// -- sample_labels2
// --   Samples the label assignments for each data point
// --------------------------------------------------------------------------
void clusters::max_labels3()
{
   linkedListNode<int>* node;

   // =================================================================================
   // propose local label changes within superclusters
   // =================================================================================

   // temporary reduction array variables
   reduction_array<int> NArr(Nthreads, K*2, 0);
   reduction_array2<double> tArr(Nthreads, K*2, D, 0);
   reduction_array2<double> TArr(Nthreads, K*2, D2, 0);
   reduction_array<double> likelihoodArr(Nthreads, K, 0);
   reduction_array<double> likelihoodzArr(Nthreads, K, 0);
   reduction_array<double> likelihoodzMtx(Nthreads, K*K, 0);

   reduction_array<double> tempData(Nthreads, D, 0);
   reduction_array<double> tempSubData(Nthreads, D, 0);

   arr(double) blurKernel = blurKernels[blurKernelNum].data();
   int blurKernelL = blurKernels[blurKernelNum].size();
   double blurSumf2 = 0;
   for (int x=0; x<blurKernelL; x++) for (int y=0; y<blurKernelL; y++)
   {
      double fxy = blurKernels[blurKernelNum][x]*blurKernels[blurKernelNum][y];
      blurSumf2 += fxy*fxy;
   }
   //mexPrintf("blurKernel=%d \t blurKernelL=%d \t blurSumf2=%e\n", blurKernelNum, blurKernelL, blurSumf2);
   blurSumf2 = 1.0/blurSumf2;


   // loop through points and sample labels
   int k0 = z2k[(int)phi[0]];
   for (int it=0; it<17; it++)
   {
      populate_meanDifference();
      #pragma omp parallel for schedule(guided)
      for (int i=0; i<N; i++) if (omask[i])
      {
         int proc = omp_get_thread_num();
         //if ( (int)(my_rand(rArr[proc])*17)!=0 )
         if ( i%17!=it )
            continue;
         double* tprobabilities = probsArr[proc].data();
         double tphi;
         double phii = phi[i];
         int k;
         // find the distribution over possible ones
         double maxProb = -mxGetInf();


         arr(double) ttempdata = tempData.getThreadArray(proc);
         arr(double) curmean = params[(int)phii]->get_params()->get_mean();
         //memcpy(ttempdata, data+i*D, sizeof(double)*D);
         for (int d=0; d<D; d++)
            ttempdata[d] = (meanImage[i*D+d]*blurSumf2 + curmean[d]);

         arr(double) ttempsubdata = tempSubData.getThreadArray(proc);
         int mi = phii;
         arr(double) cursubmean;
         if (phii-mi-0.5<0)
            cursubmean = params[mi]->get_paramsl()->get_mean();
         else
            cursubmean = params[mi]->get_paramsr()->get_mean();
         for (int d=0; d<D; d++)
            ttempsubdata[d] = (meanSubImage[i*D+d]*blurSumf2 + cursubmean[d]);

         /*if (i%10000==0)
         {
            mexPrintf("-----------------\n");
            mexPrintf("%e %e\n", data[i*D], ttempdata[0]/blurScale);
            mexPrintf("%e %e\n", ttempdata[0], curmean[0]*blurScale);
         }*/

         if (K==1)
         {
            k = k0;
         }
         else if (useSuperclusters)
         {
            int ki = z2k[(int)phii];
            int sci = superclusters[ki];
            for (int k2i=0; k2i<supercluster_labels_count[sci]; k2i++)
            {
               int k2 = supercluster_labels[sci*K + k2i];
               int m = k2z[k2];

               // find the probability of the data belonging to this component
               double prob = params[m]->get_params()->predictive_loglikelihoodScalePrec(ttempdata, blurSumf2) + sticks[m];
               maxProb = max(maxProb, prob);
               tprobabilities[k2i] = prob;
            }

            // sample a new cluster label
            double totalProb = total_logcategorical(tprobabilities, supercluster_labels_count[sci], maxProb);
            int k2i = max_categorical(tprobabilities, supercluster_labels_count[sci], maxProb, totalProb, rArr[proc]);
            k = supercluster_labels[sci*K+k2i];
         }
         else
         {
            for (int k2=0; k2<K; k2++)
            {
               int m = k2z[k2];

               // find the probability of the data belonging to this component
               double prob = params[m]->get_params()->predictive_loglikelihoodScalePrec(ttempdata, blurSumf2) + sticks[m];

               if (m!=(int)(phi[i+1]))
                  prob += -MRFw;
               if (m!=(int)(phi[i-1]))
                  prob += -MRFw;
               if (m!=(int)(phi[i+X]))
                  prob += -MRFw;
               if (m!=(int)(phi[i-X]))
                  prob += -MRFw;

               //double prob = params[m]->get_params()->predictive_loglikelihood(data+i*D) + sticks[m];
               maxProb = max(maxProb, prob);
               tprobabilities[k2] = prob;
            }

            // sample a new cluster label
            double totalProb = total_logcategorical(tprobabilities, K, maxProb);
            k = max_categorical(tprobabilities, K, maxProb, totalProb, rArr[proc]);
         }
         int m = k2z[k];

         double loglikelihood, lognorm;
         double logpl = params[m]->get_paramsl()->predictive_loglikelihoodScalePrec(ttempsubdata, blurSumf2);
         double logpr = params[m]->get_paramsr()->predictive_loglikelihoodScalePrec(ttempsubdata, blurSumf2);
         //double logpl = params[m]->get_paramsl()->predictive_loglikelihood(data+i*D);
         //double logpr = params[m]->get_paramsr()->predictive_loglikelihood(data+i*D);

         if (i+1<X && omask[i+1] && m==(int)(phi[i+1]))
         {
            if (phi[i+1]-m-0.5>0)
               logpl += -MRFw/1;
            else
               logpr += -MRFw/1;
         }
         if (i-1>=0 && omask[i-1] && m==(int)(phi[i-1]))
         {
            if (phi[i-1]-m-0.5>0)
               logpl += -MRFw/1;
            else
               logpr += -MRFw/1;
         }
         if (i/X+1<Y && omask[i+X] && m==(int)(phi[i+X]))
         {
            if (phi[i+X]-m-0.5>0)
               logpl += -MRFw/1;
            else
               logpr += -MRFw/1;
         }
         if (i/X-1>=0 && omask[i-X] && m==(int)(phi[i-X]))
         {
            if (phi[i-X]-m-0.5>0)
               logpl += -MRFw/1;
            else
               logpr += -MRFw/1;
         }

         tphi = params[m]->max_subcluster_label(logpl, logpr, rArr[proc], loglikelihood, lognorm);

         // update changes
         likelihoodArr.reduce_add(proc, k, loglikelihood);
         likelihoodzArr.reduce_add(proc, k, loglikelihood-lognorm);

         // update stats
         phi[i] = m + tphi;

         // accumulate
         int bin = k*2 + (int)(tphi*2);
         NArr.reduce_inc(proc, bin);
         tArr.reduce_add(proc, bin, data+(i*D));
         TArr.reduce_add_outerprod(proc, bin, data+i*D);

         double pz = tprobabilities[k];
         for (int k2=0; k2<K; k2++)
         {
            double temp = pz;
            if (k!=k2)
               temp = logsumexp(pz, tprobabilities[k2]);
               //temp = logsumexp(pz, tprobabilities[k2]);
            likelihoodzMtx.reduce_add(proc, k*K + k2, temp);
         }
      }
   }

   /*#pragma omp parallel for schedule(guided)
   for (int i=0; i<N; i++) if (omask[i])
   {
      int proc = omp_get_thread_num();
      int m = phi[i];
      int k = z2k[m];
      int bin = k*2 + (int)((phi[i]-m)*2);
      NArr.reduce_inc(proc, bin);
      tArr.reduce_add(proc, bin, data+(i*D));
      TArr.reduce_add_outerprod(proc, bin, data+i*D);
   }*/

   arr(double) flikelihoodzArr = likelihoodzArr.final_reduce_add();
   arr(double) flikelihoodzMtx = likelihoodzMtx.final_reduce_add();
   logpzMtx.resize(K*K);
   memcpy(logpzMtx.data(), flikelihoodzMtx, sizeof(double)*K*K);

   // accumulate cluster statistics
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
      if (newlikelihoodDelta<=0)// && likelihoodDelta[m]<0)
         splittable[m] = true;
      //mexPrintf("k=%d \t %e %e\n", k, likelihoodOld[m], flikelihood[k]/params[m]->getN());
      //mexPrintf("splittable[%d]=%d\n", k, (int)splittable[m]);
      likelihoodDelta[m] = newlikelihoodDelta;
      likelihoodOld[m] = (flikelihood[k]/params[m]->getN());

      params[m]->setlogpz(flikelihoodzArr[k]);
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
      mexPrintf("Deleted\n");
      vector<int> oldz2k(z2k);
      int oldK = K;

      //for (int k=0; k<K; k++)
         //mexPrintf("%d %d\n", fNArr[k*2], fNArr[k*2+1]);

      populate_k2z_z2k();
      // resample superclusters because the mapping is all broken
      sample_superclusters();

      vector<double> oldlogpzMtx = logpzMtx;
      logpzMtx.resize(K*K);
      for (int k1=0; k1<K; k1++) for (int k2=0; k2<K; k2++)
      {
         int m1 = k2z[k1];
         int m2 = k2z[k2];
         int oldk1 = oldz2k[m1];
         int oldk2 = oldz2k[m2];
         logpzMtx[k1*K + k2] = oldlogpzMtx[oldk1*oldK + oldk2];
      }
   }
}











bool clusters::isBlurring()
{
   arr(double) blurKernel = blurKernels[blurKernelNum].data();
   int blurKernelL = blurKernels[blurKernelNum].size();
   int nnz = 0;
   for (int x=0; x<blurKernelL; x++)
   {
      nnz += (blurKernels[blurKernelNum][x]!=0);
   }
   return nnz>1;
}

// --------------------------------------------------------------------------
// -- sample_labels2
// --   Samples the label assignments for each data point
// --------------------------------------------------------------------------
void clusters::sample_labels3_noblur(double weight)
{
   linkedListNode<int>* node;

   // =================================================================================
   // propose local label changes within superclusters
   // =================================================================================

   // temporary reduction array variables
   reduction_array<int> NArr(Nthreads, K*2, 0);
   reduction_array2<double> tArr(Nthreads, K*2, D, 0);
   reduction_array2<double> TArr(Nthreads, K*2, D2, 0);
   reduction_array<double> likelihoodArr(Nthreads, K, 0);
   reduction_array<double> likelihoodzArr(Nthreads, K, 0);
   reduction_array<double> likelihoodzMtx(Nthreads, K*K, 0);

   // loop through points and sample labels
   int k0 = z2k[(int)phi[0]];
   #pragma omp parallel for schedule(guided)
   for (int i=0; i<N; i++) if (omask[i])
   {
      int proc = omp_get_thread_num();
      //if ( (int)(my_rand(rArr[proc])*17)!=0 )

      double* tprobabilities = probsArr[proc].data();
      double tphi;
      double phii = phi[i];
      int k;
      // find the distribution over possible ones
      double maxProb = -mxGetInf();


      /*if (i%10000==0)
      {
         mexPrintf("-----------------\n");
         mexPrintf("%e %e\n", data[i*D], ttempdata[0]/blurScale);
         mexPrintf("%e %e\n", ttempdata[0], curmean[0]*blurScale);
      }*/

      if (K==1)
      {
         k = k0;
      }
      else if (useSuperclusters)
      {
         int ki = z2k[(int)phii];
         int sci = superclusters[ki];
         for (int k2i=0; k2i<supercluster_labels_count[sci]; k2i++)
         {
            int k2 = supercluster_labels[sci*K + k2i];
            int m = k2z[k2];

            // find the probability of the data belonging to this component
            double prob = params[m]->get_params()->predictive_loglikelihood(data+i*D) + sticks[m];
            maxProb = max(maxProb, prob);
            tprobabilities[k2i] = prob;
         }

         // sample a new cluster label
         double totalProb = total_logcategorical(tprobabilities, supercluster_labels_count[sci], maxProb);
         int k2i = sample_logcategorical(tprobabilities, supercluster_labels_count[sci], maxProb, totalProb, rArr[proc]);
         k = supercluster_labels[sci*K+k2i];
      }
      else
      {
         for (int k2=0; k2<K; k2++)
         {
            int m = k2z[k2];

            // find the probability of the data belonging to this component
            double prob = params[m]->get_params()->predictive_loglikelihood(data+i*D) + sticks[m];

            if (m!=(int)(phi[i+1]))
               prob += -MRFw;
            if (m!=(int)(phi[i-1]))
               prob += -MRFw;
            if (m!=(int)(phi[i+X]))
               prob += -MRFw;
            if (m!=(int)(phi[i-X]))
               prob += -MRFw;

            //double prob = params[m]->get_params()->predictive_loglikelihood(data+i*D) + sticks[m];
            maxProb = max(maxProb, prob);
            tprobabilities[k2] = prob;
         }

         // sample a new cluster label
         double totalProb = total_logcategorical(tprobabilities, K, maxProb);
         k = sample_logcategorical(tprobabilities, K, maxProb, totalProb, rArr[proc]);
      }
      int m = k2z[k];

      double loglikelihood, lognorm;
      double logpl = params[m]->get_paramsl()->predictive_loglikelihood(data+i*D);
      double logpr = params[m]->get_paramsr()->predictive_loglikelihood(data+i*D);
      //double logpl = params[m]->get_paramsl()->predictive_loglikelihood(data+i*D);
      //double logpr = params[m]->get_paramsr()->predictive_loglikelihood(data+i*D);

      if (i+1<X && omask[i+1] && m==(int)(phi[i+1]))
      {
         if (phi[i+1]-m-0.5>0)
            logpl += -MRFw * weight;
         else
            logpr += -MRFw * weight;
      }
      if (i-1>=0 && omask[i-1] && m==(int)(phi[i-1]))
      {
         if (phi[i-1]-m-0.5>0)
            logpl += -MRFw * weight;
         else
            logpr += -MRFw * weight;
      }
      if (i/X+1<Y && omask[i+X] && m==(int)(phi[i+X]))
      {
         if (phi[i+X]-m-0.5>0)
            logpl += -MRFw * weight;
         else
            logpr += -MRFw * weight;
      }
      if (i/X-1>=0 && omask[i-X] && m==(int)(phi[i-X]))
      {
         if (phi[i-X]-m-0.5>0)
            logpl += -MRFw * weight;
         else
            logpr += -MRFw * weight;
      }

      tphi = params[m]->sample_subcluster_label(logpl, logpr, rArr[proc], loglikelihood, lognorm);

      // update changes
      likelihoodArr.reduce_add(proc, k, loglikelihood);
      likelihoodzArr.reduce_add(proc, k, loglikelihood-lognorm);

      // update stats
      phi[i] = m + tphi;

      // accumulate
      int bin = k*2 + (int)(tphi*2);
      NArr.reduce_inc(proc, bin);
      tArr.reduce_add(proc, bin, data+(i*D));
      TArr.reduce_add_outerprod(proc, bin, data+i*D);

      double pz = tprobabilities[k];
      for (int k2=0; k2<K; k2++)
      {
         double temp = pz;
         if (k!=k2)
            temp = logsumexp(pz, tprobabilities[k2]);
            //temp = logsumexp(pz, tprobabilities[k2]);
         likelihoodzMtx.reduce_add(proc, k*K + k2, temp);
      }
   }

   /*#pragma omp parallel for schedule(guided)
   for (int i=0; i<N; i++) if (omask[i])
   {
      int proc = omp_get_thread_num();
      int m = phi[i];
      int k = z2k[m];
      int bin = k*2 + (int)((phi[i]-m)*2);
      NArr.reduce_inc(proc, bin);
      tArr.reduce_add(proc, bin, data+(i*D));
      TArr.reduce_add_outerprod(proc, bin, data+i*D);
   }*/

   arr(double) flikelihoodzArr = likelihoodzArr.final_reduce_add();
   arr(double) flikelihoodzMtx = likelihoodzMtx.final_reduce_add();
   logpzMtx.resize(K*K);
   memcpy(logpzMtx.data(), flikelihoodzMtx, sizeof(double)*K*K);

   // accumulate cluster statistics
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
      //mexPrintf("k=%d \t %e %e\n", k, likelihoodOld[m], flikelihood[k]/params[m]->getN());
      //mexPrintf("splittable[%d]=%d\n", k, (int)splittable[m]);
      likelihoodDelta[m] = newlikelihoodDelta;
      likelihoodOld[m] = (flikelihood[k]/params[m]->getN());

      params[m]->setlogpz(flikelihoodzArr[k]);
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
      mexPrintf("Deleted\n");
      vector<int> oldz2k(z2k);
      int oldK = K;

      //for (int k=0; k<K; k++)
         //mexPrintf("%d %d\n", fNArr[k*2], fNArr[k*2+1]);

      populate_k2z_z2k();
      // resample superclusters because the mapping is all broken
      sample_superclusters();

      vector<double> oldlogpzMtx = logpzMtx;
      logpzMtx.resize(K*K);
      for (int k1=0; k1<K; k1++) for (int k2=0; k2<K; k2++)
      {
         int m1 = k2z[k1];
         int m2 = k2z[k2];
         int oldk1 = oldz2k[m1];
         int oldk2 = oldz2k[m2];
         logpzMtx[k1*K + k2] = oldlogpzMtx[oldk1*oldK + oldk2];
      }
   }
}
// --------------------------------------------------------------------------
// -- sample_labels2
// --   Samples the label assignments for each data point
// --------------------------------------------------------------------------
void clusters::max_labels3_noblur(double weight)
{
   linkedListNode<int>* node;

   // =================================================================================
   // propose local label changes within superclusters
   // =================================================================================

   // temporary reduction array variables
   reduction_array<int> NArr(Nthreads, K*2, 0);
   reduction_array2<double> tArr(Nthreads, K*2, D, 0);
   reduction_array2<double> TArr(Nthreads, K*2, D2, 0);
   reduction_array<double> likelihoodArr(Nthreads, K, 0);
   reduction_array<double> likelihoodzArr(Nthreads, K, 0);
   reduction_array<double> likelihoodzMtx(Nthreads, K*K, 0);

   // loop through points and sample labels
   int k0 = z2k[(int)phi[0]];
   #pragma omp parallel for schedule(guided)
   for (int i=0; i<N; i++) if (omask[i])
   {
      int proc = omp_get_thread_num();
      //if ( (int)(my_rand(rArr[proc])*17)!=0 )

      double* tprobabilities = probsArr[proc].data();
      double tphi;
      double phii = phi[i];
      int k;
      // find the distribution over possible ones
      double maxProb = -mxGetInf();


      /*if (i%10000==0)
      {
         mexPrintf("-----------------\n");
         mexPrintf("%e %e\n", data[i*D], ttempdata[0]/blurScale);
         mexPrintf("%e %e\n", ttempdata[0], curmean[0]*blurScale);
      }*/

      if (K==1)
      {
         k = k0;
      }
      else if (useSuperclusters)
      {
         int ki = z2k[(int)phii];
         int sci = superclusters[ki];
         for (int k2i=0; k2i<supercluster_labels_count[sci]; k2i++)
         {
            int k2 = supercluster_labels[sci*K + k2i];
            int m = k2z[k2];

            // find the probability of the data belonging to this component
            double prob = params[m]->get_params()->predictive_loglikelihood(data+i*D) + sticks[m];
            maxProb = max(maxProb, prob);
            tprobabilities[k2i] = prob;
         }

         // sample a new cluster label
         double totalProb = total_logcategorical(tprobabilities, supercluster_labels_count[sci], maxProb);
         int k2i = max_categorical(tprobabilities, supercluster_labels_count[sci], maxProb, totalProb, rArr[proc]);
         k = supercluster_labels[sci*K+k2i];
      }
      else
      {
         for (int k2=0; k2<K; k2++)
         {
            int m = k2z[k2];

            // find the probability of the data belonging to this component
            double prob = params[m]->get_params()->predictive_loglikelihood(data+i*D) + sticks[m];

            if (m!=(int)(phi[i+1]))
               prob += -MRFw;
            if (m!=(int)(phi[i-1]))
               prob += -MRFw;
            if (m!=(int)(phi[i+X]))
               prob += -MRFw;
            if (m!=(int)(phi[i-X]))
               prob += -MRFw;

            //double prob = params[m]->get_params()->predictive_loglikelihood(data+i*D) + sticks[m];
            maxProb = max(maxProb, prob);
            tprobabilities[k2] = prob;
         }

         // sample a new cluster label
         double totalProb = total_logcategorical(tprobabilities, K, maxProb);
         k = max_categorical(tprobabilities, K, maxProb, totalProb, rArr[proc]);
      }
      int m = k2z[k];

      double loglikelihood, lognorm;
      double logpl = params[m]->get_paramsl()->predictive_loglikelihood(data+i*D);
      double logpr = params[m]->get_paramsr()->predictive_loglikelihood(data+i*D);
      //double logpl = params[m]->get_paramsl()->predictive_loglikelihood(data+i*D);
      //double logpr = params[m]->get_paramsr()->predictive_loglikelihood(data+i*D);

      if (i+1<X && omask[i+1] && m==(int)(phi[i+1]))
      {
         if (phi[i+1]-m-0.5>0)
            logpl += -MRFw * weight;
         else
            logpr += -MRFw * weight;
      }
      if (i-1>=0 && omask[i-1] && m==(int)(phi[i-1]))
      {
         if (phi[i-1]-m-0.5>0)
            logpl += -MRFw * weight;
         else
            logpr += -MRFw * weight;
      }
      if (i/X+1<Y && omask[i+X] && m==(int)(phi[i+X]))
      {
         if (phi[i+X]-m-0.5>0)
            logpl += -MRFw * weight;
         else
            logpr += -MRFw * weight;
      }
      if (i/X-1>=0 && omask[i-X] && m==(int)(phi[i-X]))
      {
         if (phi[i-X]-m-0.5>0)
            logpl += -MRFw * weight;
         else
            logpr += -MRFw * weight;
      }

      tphi = params[m]->max_subcluster_label(logpl, logpr, rArr[proc], loglikelihood, lognorm);

      // update changes
      likelihoodArr.reduce_add(proc, k, loglikelihood);
      likelihoodzArr.reduce_add(proc, k, loglikelihood-lognorm);

      // update stats
      phi[i] = m + tphi;

      // accumulate
      int bin = k*2 + (int)(tphi*2);
      NArr.reduce_inc(proc, bin);
      tArr.reduce_add(proc, bin, data+(i*D));
      TArr.reduce_add_outerprod(proc, bin, data+i*D);

      double pz = tprobabilities[k];
      for (int k2=0; k2<K; k2++)
      {
         double temp = pz;
         if (k!=k2)
            temp = logsumexp(pz, tprobabilities[k2]);
            //temp = logsumexp(pz, tprobabilities[k2]);
         likelihoodzMtx.reduce_add(proc, k*K + k2, temp);
      }
   }

   /*#pragma omp parallel for schedule(guided)
   for (int i=0; i<N; i++) if (omask[i])
   {
      int proc = omp_get_thread_num();
      int m = phi[i];
      int k = z2k[m];
      int bin = k*2 + (int)((phi[i]-m)*2);
      NArr.reduce_inc(proc, bin);
      tArr.reduce_add(proc, bin, data+(i*D));
      TArr.reduce_add_outerprod(proc, bin, data+i*D);
   }*/

   arr(double) flikelihoodzArr = likelihoodzArr.final_reduce_add();
   arr(double) flikelihoodzMtx = likelihoodzMtx.final_reduce_add();
   logpzMtx.resize(K*K);
   memcpy(logpzMtx.data(), flikelihoodzMtx, sizeof(double)*K*K);

   // accumulate cluster statistics
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
      if (newlikelihoodDelta<=0)// && likelihoodDelta[m]<0)
         splittable[m] = true;
      //mexPrintf("k=%d \t %e %e\n", k, likelihoodOld[m], flikelihood[k]/params[m]->getN());
      //mexPrintf("splittable[%d]=%d\n", k, (int)splittable[m]);
      likelihoodDelta[m] = newlikelihoodDelta;
      likelihoodOld[m] = (flikelihood[k]/params[m]->getN());

      params[m]->setlogpz(flikelihoodzArr[k]);
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
      mexPrintf("Deleted\n");
      vector<int> oldz2k(z2k);
      int oldK = K;

      //for (int k=0; k<K; k++)
         //mexPrintf("%d %d\n", fNArr[k*2], fNArr[k*2+1]);

      populate_k2z_z2k();
      // resample superclusters because the mapping is all broken
      sample_superclusters();

      vector<double> oldlogpzMtx = logpzMtx;
      logpzMtx.resize(K*K);
      for (int k1=0; k1<K; k1++) for (int k2=0; k2<K; k2++)
      {
         int m1 = k2z[k1];
         int m2 = k2z[k2];
         int oldk1 = oldz2k[m1];
         int oldk2 = oldz2k[m2];
         logpzMtx[k1*K + k2] = oldlogpzMtx[oldk1*oldK + oldk2];
      }
   }
}



void clusters::max_labels2()
{
   linkedListNode<int>* node;

   // =================================================================================
   // propose local label changes within superclusters
   // =================================================================================

   // temporary reduction array variables
   reduction_array<int> NArr(Nthreads, K*2, 0);
   reduction_array2<double> tArr(Nthreads, K*2, D, 0);
   reduction_array2<double> TArr(Nthreads, K*2, D2, 0);
   reduction_array<double> likelihoodArr(Nthreads, K, 0);
   reduction_array<double> likelihoodzArr(Nthreads, K, 0);
   reduction_array<double> likelihoodzMtx(Nthreads, K*K, 0);

   // loop through points and sample labels
   int k0 = z2k[(int)phi[0]];
   #pragma omp parallel for schedule(guided)
   for (int i=0; i<N; i++) if (omask[i])
   {
      int proc = omp_get_thread_num();
      double* tprobabilities = probsArr[proc].data();
      double tphi;
      double phii = phi[i];
      int k;
      // find the distribution over possible ones
      double maxProb = -mxGetInf();
      if (K==1)
      {
         k = k0;
      }
      else if (useSuperclusters)
      {
         int ki = z2k[(int)phii];
         int sci = superclusters[ki];
         for (int k2i=0; k2i<supercluster_labels_count[sci]; k2i++)
         {
            int k2 = supercluster_labels[sci*K + k2i];
            int m = k2z[k2];

            // find the probability of the data belonging to this component
            double prob = params[m]->get_params()->predictive_loglikelihood(data+i*D) + sticks[m];
            maxProb = max(maxProb, prob);
            tprobabilities[k2i] = prob;
         }

         // sample a new cluster label
         int k2i = max_categorical(tprobabilities, supercluster_labels_count[sci]);
         k = supercluster_labels[sci*K+k2i];
      }
      else
      {
         for (int k2=0; k2<K; k2++)
         {
            int m = k2z[k2];

            // find the probability of the data belonging to this component
            double prob = params[m]->get_params()->predictive_loglikelihood(data+i*D) + sticks[m];
            maxProb = max(maxProb, prob);
            tprobabilities[k2] = prob;
         }

         // sample a new cluster label
         k = max_categorical(tprobabilities, K);
      }
      int m = k2z[k];

      double loglikelihood, lognorm;
      double logpl = params[m]->get_paramsl()->predictive_loglikelihood(data+i*D);
      double logpr = params[m]->get_paramsr()->predictive_loglikelihood(data+i*D);
      //tphi = params[m]->sample_subcluster_label(logpl, logpr, rArr[proc], loglikelihood, lognorm);
      tphi = params[m]->max_subcluster_label(logpl, logpr, rArr[proc], loglikelihood, lognorm);

      // update changes
      likelihoodArr.reduce_add(proc, k, loglikelihood);
      likelihoodzArr.reduce_add(proc, k, loglikelihood-lognorm);

      // update stats
      phi[i] = m + tphi;

      // accumulate
      int bin = k*2 + (int)(tphi*2);
      NArr.reduce_inc(proc, bin);
      tArr.reduce_add(proc, bin, data+(i*D));
      TArr.reduce_add_outerprod(proc, bin, data+i*D);

      double pz = tprobabilities[k];
      for (int k2=0; k2<K; k2++)
      {
         double temp = pz;
         if (k!=k2)
            temp = logsumexp(pz, tprobabilities[k2]);
            //temp = logsumexp(pz, tprobabilities[k2]);
         likelihoodzMtx.reduce_add(proc, k*K + k2, temp);
      }
   }

   arr(double) flikelihoodzArr = likelihoodzArr.final_reduce_add();
   arr(double) flikelihoodzMtx = likelihoodzMtx.final_reduce_add();
   logpzMtx.resize(K*K);
   memcpy(logpzMtx.data(), flikelihoodzMtx, sizeof(double)*K*K);

   // accumulate cluster statistics
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
      if (newlikelihoodDelta<=0)// && likelihoodDelta[m]<0)
         splittable[m] = true;
      //mexPrintf("splittable[%d]=%d\n", k, (int)splittable[m]);
      likelihoodDelta[m] = newlikelihoodDelta;
      likelihoodOld[m] = (flikelihood[k]/params[m]->getN());

      params[m]->setlogpz(flikelihoodzArr[k]);
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
      mexPrintf("Deleted\n");
      vector<int> oldz2k(z2k);
      int oldK = K;

      populate_k2z_z2k();
      // resample superclusters because the mapping is all broken
      sample_superclusters();

      vector<double> oldlogpzMtx = logpzMtx;
      logpzMtx.resize(K*K);
      for (int k1=0; k1<K; k1++) for (int k2=0; k2<K; k2++)
      {
         int m1 = k2z[k1];
         int m2 = k2z[k2];
         int oldk1 = oldz2k[m1];
         int oldk2 = oldz2k[m2];
         logpzMtx[k1*K + k2] = oldlogpzMtx[oldk1*oldK + oldk2];
      }
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

               cluster_sampledT<normalmean_sampled>* paramsm = params[zm];
               cluster_sampledT<normalmean_sampled>* paramsn = params[zn];

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
      mexPrintf("NumMerges=%d\n",numMerges);
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
   for (int k=0; k<K; k++) if (K + num_splits < maxK)
   {
      // check to see if the subclusters have converged
      if (always_splittable || splittable[k2z[k]])//paramsChanges[k2z[k]]/((double)params[k2z[k]]->getN())<=0.1)
      {
         int proc = omp_get_thread_num();
         int kz = k2z[k];
         cluster_sampledT<normalmean_sampled>* paramsk = params[kz];
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

      #pragma omp parallel for schedule(guided)
      for (int i=0; i<N; i++)  if (omask[i] && do_reset[z2k[(int)(phi[i])]])
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
      mexPrintf("NumSplits=%d\n", num_splits);
      int temp_num_splits = num_splits;
      // figure out which labels to split into and prepare simple split stuff
      vector<int> split_into(K);
      vector<int> split_into_k(K);
      int k = 0;
      for (k; k<K; k++)
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
            params[nzh] = new cluster_sampledT<normalmean_sampled>(hyper,alpha);
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

      #pragma omp parallel for schedule(guided)
      for (int i=0; i<N; i++) if (omask[i] && do_split[z2k[(int)(phi[i])]])
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
   }
   if (num_resets>0 || num_splits>0)
      populate_k2z_z2k();
}



// --------------------------------------------------------------------------
// -- propose_merges
// --   Samples the label assignments for each data point
// --------------------------------------------------------------------------
void clusters::propose_merges2()
{
   vector<int> merge_with(K, -1);
   int numMerges = 0;
   for (int km=0; km<K; km++) if (always_splittable || splittable[k2z[km]])
   {
      int zm = k2z[km];
      if (merge_with[km]<0)
         for (int kn=km+1; kn<K; kn++) if (always_splittable || splittable[k2z[kn]])
         {
            int zn = k2z[kn];
            if ((!useSuperclusters || superclusters[km]==superclusters[kn]) && merge_with[km]<0 && merge_with[kn]<0)
            {
               int Nm = params[zm]->getN();
               int Nn = params[zn]->getN();
               int Nkh = Nm+Nn;

               cluster_sampledT<normalmean_sampled>* paramsm = params[zm];
               cluster_sampledT<normalmean_sampled>* paramsn = params[zn];

               //mexPrintf("============\n");
               double HR = -logalpha - myloggamma(Nm) - myloggamma(Nn) + myloggamma(Nkh);
               //double HR = -logalpha - gsl_sf_lngamma(Nm) - gsl_sf_lngamma(Nn) + gsl_sf_lngamma(Nkh);
               //mexPrintf("1 : %e\n", HR);
               HR += paramsm->data_loglikelihood_marginalized_testmerge(paramsn);
               //mexPrintf("2 : %e\n", HR);
               HR += -paramsm->data_loglikelihood_marginalized() - paramsn->data_loglikelihood_marginalized();
               //mexPrintf("3 : %e\n", HR);
               // TODO need to agregate those when I sample assignment to (normal) clusters
               HR += logpzMtx[km*K+km] + logpzMtx[kn*K+kn] - logpzMtx[km*K+kn] - logpzMtx[kn*K+km];
               //HR += -paramsm->logmu_prior() - paramsn->logmu_prior() - paramsm->data_loglikelihood() - paramsn->data_loglikelihood();

               //mexPrintf("4 : %e\n", HR);

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
      mexPrintf("NumMerges=%d\n",numMerges);drawnow();
      // fix the phi's
      #pragma omp parallel for schedule(guided)
      for (int i=0; i<N; i++) if (omask[i])
      {
         int zi = phi[i];
         int ki = z2k[zi];
         if (merge_with[ki]>=0)
         {
            phi[i] = k2z[merge_with[ki]] + ((merge_with[ki]==ki) ? 0.25 : 0.75);
         }
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
void clusters::propose_splits2()
{
   vector<bool> do_split(K, false);
   vector<bool> do_reset(K, false);
   int num_splits = 0;
   int num_resets = 0;
   //#pragma omp parallel for reduction(+:num_splits)
   //mexPrintf("====================\n");

   for (int k=0; k<K; k++) if (K + num_splits < maxK)
   {
      // check to see if the subclusters have converged
      if (always_splittable || splittable[k2z[k]])//paramsChanges[k2z[k]]/((double)params[k2z[k]]->getN())<=0.1)
      {
         int kz = k2z[k];
         cluster_sampledT<normalmean_sampled>* paramsk = params[kz];
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
        // TODO need to agregate those when I sample assignments to subclusters 
         HR += -paramsk->getlogpz();
         //HR += paramsk->logmul_prior() + paramsk->logmur_prior() - paramsk->logmu_prior();
         //HR += paramsk->data_lloglikelihood() + paramsk->data_rloglikelihood() - paramsk->data_loglikelihood();
         //mexPrintf("HR=%e\n", HR);

         if ((HR>0 || my_rand(rArr[0]) < exp(HR)))
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

      #pragma omp parallel for schedule(guided)
      for (int i=0; i<N; i++) if (omask[i] && do_reset[z2k[(int)(phi[i])]])
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
         splittable[m] = false;
         likelihoodOld[m] = -mxGetInf();
         likelihoodDelta[m] = mxGetInf();
      }
   }


   if (num_splits>0)
   {
      mexPrintf("NumSplits=%d\n", num_splits);drawnow();
      int temp_num_splits = num_splits;
      // figure out which labels to split into and prepare simple split stuff
      vector<int> split_into(K);
      vector<int> split_into_k(K);
      int k = 0;
      for (k; k<K; k++)
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
            params[nzh] = new cluster_sampledT<normalmean_sampled>(hyper,alpha);
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

      #pragma omp parallel for schedule(guided)
      for (int i=0; i<N; i++) if (omask[i] && do_split[z2k[(int)(phi[i])]])
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
   }

   if (num_resets>0 || num_splits>0)
      populate_k2z_z2k();
}












double clusters::calc_sigma_merge(int k1, int k2)
{
   int numSigmas = hyper.getNumSigmas();
   
   double totalLogNew = -mxGetInf();
   double totalLogOld = -mxGetInf();

   int m1 = k2z[k1];
   int m2 = k2z[k2];

   for (int numSigma=0; numSigma<numSigmas; numSigma++)
   {
      double logNew = 0;
      double logOld = 0;
      for (int k=0; k<K; k++)
      {
         int m = k2z[k];
         double tempNew, tempOld;
         if (k==k1)
         {
            tempNew = params[m1]->get_params()->data_loglikelihood_marginalized_testmerge(numSigma, params[m2]->get_params());
            tempOld = params[m1]->get_params()->data_loglikelihood_marginalized(numSigma) + params[m2]->get_params()->data_loglikelihood_marginalized(numSigma);
         }
         else if (k!=k2)
         {
            tempNew = params[m]->get_params()->data_loglikelihood_marginalized(numSigma);
            tempOld = tempNew;
         }

         logNew += tempNew;
         logOld += tempOld;
      }
      totalLogNew = logsumexp(totalLogNew, logNew);
      totalLogOld = logsumexp(totalLogOld, logOld);
   }

   return totalLogNew-totalLogOld;
}

void clusters::propose_merges3()
{
   vector<int> merge_with(K, -1);
   int numMerges = 0;
   for (int km=0; km<K; km++) if (always_splittable || splittable[k2z[km]])
   {
      int zm = k2z[km];
      if (merge_with[km]<0)
         for (int kn=km+1; kn<K; kn++) if (always_splittable || splittable[k2z[kn]])
         {
            int zn = k2z[kn];
            if ((!useSuperclusters || superclusters[km]==superclusters[kn]) && merge_with[km]<0 && merge_with[kn]<0)
            {
               int Nm = params[zm]->getN();
               int Nn = params[zn]->getN();
               int Nkh = Nm+Nn;

               cluster_sampledT<normalmean_sampled>* paramsm = params[zm];
               cluster_sampledT<normalmean_sampled>* paramsn = params[zn];

               //mexPrintf("============\n");
               double HR = -logalpha - myloggamma(Nm) - myloggamma(Nn) + myloggamma(Nkh);
               //double HR = -logalpha - gsl_sf_lngamma(Nm) - gsl_sf_lngamma(Nn) + gsl_sf_lngamma(Nkh);
               //mexPrintf("1 : %e\n", HR);
               HR += calc_sigma_merge(km,kn);
               //mexPrintf("3 : %e\n", HR);
               HR += logpzMtx[km*K+km] + logpzMtx[kn*K+kn] - logpzMtx[km*K+kn] - logpzMtx[kn*K+km];
               //HR += -paramsm->logmu_prior() - paramsn->logmu_prior() - paramsm->data_loglikelihood() - paramsn->data_loglikelihood();

               //mexPrintf("4 : %e\n", HR);

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
      mexPrintf("NumMerges3=%d\n",numMerges);drawnow();
      // fix the phi's
      #pragma omp parallel for schedule(guided)
      for (int i=0; i<N; i++) if (omask[i])
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
   sample_sigma();
}







double clusters::calc_sigma_split(int splitk)
{
   int numSigmas = hyper.getNumSigmas();
   
   double totalLogNew = -mxGetInf();
   double totalLogOld = -mxGetInf();

   for (int numSigma=0; numSigma<numSigmas; numSigma++)
   {
      double logNew = 0;
      double logOld = 0;
      for (int k=0; k<K; k++)
      {
         int m = k2z[k];
         double tempNew, tempOld;
         if (k==splitk)
         {
            tempNew = params[m]->get_paramsl()->data_loglikelihood_marginalized(numSigma) + params[m]->get_paramsr()->data_loglikelihood_marginalized(numSigma);
            tempOld = params[m]->get_params()->data_loglikelihood_marginalized(numSigma);
         }
         else
         {
            tempNew = params[m]->get_params()->data_loglikelihood_marginalized(numSigma);
            tempOld = tempNew;
         }

         logNew += tempNew;
         logOld += tempOld;
      }
      totalLogNew = logsumexp(totalLogNew, logNew);
      totalLogOld = logsumexp(totalLogOld, logOld);
   }

   return totalLogNew-totalLogOld;

   /*for (int k=0; k<K; k++)
   {
      int m = k2z[splitk];
      if (k==splitk)
      {
         for (int numSigma=0; numSigma<numSigmas; numSigma++)
         {
            totalLogratio += params[m]->get_paramsl()->data_loglikelihood_marginalized(numSigma) + params[m]->get_paramsr()->data_loglikelihood_marginalized(numSigma);
            totalLogratio -= params[m]->get_params()->data_loglikelihood_marginalized(numSigma);
         }
      }
      else
      {
         for (int numSigma=0; numSigma<numSigmas; numSigma++)
         {
            totalLogratio += params[m]->get_paramsl()->data_loglikelihood_marginalized(numSigma) + params[m]->get_paramsr()->data_loglikelihood_marginalized(numSigma);
            totalLogratio -= params[m]->get_params()->data_loglikelihood_marginalized(numSigma);
         }
      }
   }
   return totalLogratio;*/
}
// --------------------------------------------------------------------------
// -- propose_splits
// --   Samples the label assignments for each data point
// --------------------------------------------------------------------------
void clusters::propose_splits3()
{
   vector<bool> do_split(K, false);
   vector<bool> do_reset(K, false);
   int num_splits = 0;
   int num_resets = 0;
   //#pragma omp parallel for reduction(+:num_splits)
   //mexPrintf("====================\n");

   for (int k=0; k<K; k++) if (K + num_splits < maxK)
   {
      // check to see if the subclusters have converged
      if (always_splittable || splittable[k2z[k]])//paramsChanges[k2z[k]]/((double)params[k2z[k]]->getN())<=0.1)
      {
         int proc = omp_get_thread_num();
         int kz = k2z[k];
         cluster_sampledT<normalmean_sampled>* paramsk = params[kz];
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
         //mexPrintf("%e\t",HR);
         HR += calc_sigma_split(k);
         //mexPrintf("%e\t",HR);
         HR += -paramsk->getlogpz();
         //mexPrintf("%e\n",HR);
         drawnow();

         if ((HR>0 || my_rand(rArr[proc]) < exp(HR)))
         {
            do_split[k] = true;
            num_splits++;
            break;
         }
      }
   }



   if (num_resets>0)
   {
      // correct the labels
      reduction_array<int> NArr(Nthreads, K*2, 0);
      reduction_array2<double> tArr(Nthreads, K*2, D, 0);
      reduction_array2<double> TArr(Nthreads, K*2, D2, 0);

      #pragma omp parallel for schedule(guided)
      for (int i=0; i<N; i++) if (omask[i] && do_reset[z2k[(int)(phi[i])]])
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
         splittable[m] = false;
         likelihoodOld[m] = -mxGetInf();
         likelihoodDelta[m] = mxGetInf();
      }
   }


   if (num_splits>0)
   {
      mexPrintf("NumSplits3=%d\n", num_splits);drawnow();
      int temp_num_splits = num_splits;
      // figure out which labels to split into and prepare simple split stuff
      vector<int> split_into(K);
      vector<int> split_into_k(K);
      int k = 0;
      for (k; k<K; k++)
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
            params[nzh] = new cluster_sampledT<normalmean_sampled>(hyper,alpha);
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

      #pragma omp parallel for schedule(guided)
      for (int i=0; i<N; i++) if (omask[i] && do_split[z2k[(int)(phi[i])]])
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
   }
   if (num_resets>0 || num_splits>0)
      populate_k2z_z2k();
   sample_sigma();
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
   for (int i=0; i<N; i++) if (omask[i])
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
      cluster_sampledT<normalmean_sampled>* paramsk = params[kz];
      int Nk = paramsk->getN();
      if (paramsk->getrandomlN()<=0 || paramsk->getrandomrN()<=0)
         continue;

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
      mexPrintf("NumRandomSplits=%d\n", num_splits);
      int temp_num_splits = num_splits;
      // figure out which labels to split into and prepare simple split stuff
      vector<int> split_into(K);
      vector<int> split_into_k(K);
      int k = 0;
      for (k; k<K; k++)
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
            params[nzh] = new cluster_sampledT<normalmean_sampled>(hyper,alpha);
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
      for (int i=0; i<N; i++) if (omask[i] && do_split[z2k[(int)(phi[i])]])
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

         int m1 = k2z[k];
         int m2 = split_into[k];
         /*mexPrintf("%d %d %d -\t- %d %d %d -\t- %d %d %d %d\n",
            params[m1]->getrandomlN(), params[m1]->getrandomrN(), params[m1]->getN(),
            params[m2]->getrandomlN(), params[m2]->getrandomrN(), params[m2]->getN(),
            fNArr[k*2], fNArr[k*2+1], fNArr[newk*2], fNArr[newk*2+1]);*/
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
               cluster_sampledT<normalmean_sampled>* paramsm = params[zm];
               cluster_sampledT<normalmean_sampled>* paramsn = params[zn];

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
      mexPrintf("NumRandomMerges=%d\n",numMerges);
      // fix the phi's
      #pragma omp parallel for schedule(guided)
      for (int i=0; i<N; i++) if (omask[i])
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


void clusters::checkData()
{
   mexPrintf("Checking..\n");
   vector<int> counts(N,0);
   int total = 0;
   for (int i=0; i<N; i++) if (omask[i])
   {
      counts[(int)phi[i]]++;
      total++;
   }

   int Knew=0;
   for (int i=0; i<N; i++)
   {
      if (params[i]!=NULL)
      {
         Knew++;
         if (params[i]->getN() != counts[i])
            mexErrMsgTxt("err1\n");
         if (params[i]->getN() == 0)
            mexErrMsgTxt("err3\n");
         total -= params[i]->getN();
      }
      else if (counts[i]>0)
         mexErrMsgTxt("err2\n");
   }

   linkedListNode<int>* node = alive.getFirst();
   while (node!=NULL)
   {
      int m = node->getData();
      if (params[m]==NULL)
      {
         mexErrMsgTxt("error3\n");
      }
      if (params[m]->getN()<=0)
         mexErrMsgTxt("error4\n");
      node = node->getNext();  
   }

   if (K!=Knew)
      mexErrMsgTxt("err5\n");
   if (K!=alive.getLength())
      mexErrMsgTxt("err5.5\n");

   for (int k=0; k<K; k++)
   {
      if (k != z2k[k2z[k]])
         mexErrMsgTxt("err6\n");

      if (params[k2z[k]]==NULL)
         mexErrMsgTxt("err7\n");
   }

   if (total!=0)
      mexErrMsgTxt("err4\n");
}

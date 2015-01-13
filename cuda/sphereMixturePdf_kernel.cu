/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */
 

#include "cuda_global.h"

// executions per thread
#define N_PER_T 32
#define BLOCK_SIZE 256
#define K_MAX 256/4 // for 3D this is a max due to current 
                    // caching implementation


template<typename T>
__global__ void sphereGMMPdf_3D(T *d_p, T *d_Rnorths, T * d_q, 
    T *d_invSigmas, T *d_logNormalizer, T *d_logPi, T* d_logPdf, 
    uint32_t K, uint32_t N) 
{
  __shared__ T p[DIM*K_MAX];
  __shared__ T logNorm[K_MAX];

  const int tid = threadIdx.x;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;

  // caching 
  if(tid < DIM*K) p[tid] = d_p[tid];
  if(DIM*K <= tid && tid < DIM*K+K) 
    logNorm[tid-DIM*K] = d_logNormalizer[tid-DIM*K];
  __syncthreads(); // make sure that ys have been cached

  for(uint32_t id=idx*N_PER_T; id<min(N,(idx+1)*N_PER_T); ++id)
  {
    T q[DIM];
    T x[DIM]; 
    q[0] = d_q[id*DIM+0]; 
    q[1] = d_q[id*DIM+1]; 
    q[2] = d_q[id*DIM+2];
    T pdf[K_MAX]; 
    for(uint32_t k=0; k<K; ++k)
    {
      T *pk = p+k*DIM;
      Log_p<T>(pk,q,x);
      T *d_Rnorthsk = d_Rnorths+k*(DIM-1)*DIM;
      T xNorth[DIM-1];
      xNorth[0] = d_Rnorthsk[0]*x[0] + d_Rnorthsk[2]*x[1] + d_Rnorthsk[4]*x[2];
      xNorth[1] = d_Rnorthsk[1]*x[0] + d_Rnorthsk[3]*x[1] + d_Rnorthsk[5]*x[2];
      T *invSk = d_invSigmas + (DIM-1)*(DIM-1)*k;
      pdf[k] = d_logPi[k] + logNorm[k] - 0.5*bTAb_2D(invSk,xNorth);
//      d_logPdf[id+N*k] = - 0.5*bTAb_2D(invSk,xNorth);
    }
#pragma unroll
    for(uint32_t k=0; k<K; ++k)
      d_logPdf[id+N*k] = pdf[k];

//    T maxPdf = pdf[0];
//#pragma unroll
//    for(uint32_t k=1; k<K; ++k)
//      if(maxPdf < pdf[k]) maxPdf = pdf[k];
//    T logsumexp = exp(pdf[0]-maxPdf);
//#pragma unroll
//    for(uint32_t k=1; k<K; ++k)
//      logsumexp += exp(pdf[k]-maxPdf);
//    logsumexp = log(logsumexp)+maxPdf;
//#pragma unroll
//    for(uint32_t k=0; k<K; ++k)
//      d_logPdf[id+N*k] = pdf[k]-logsumexp;
//    // TODO: it is now actually returning logPdfs! may break some stuff
//      d_logPdf[id+N*k] = exp(pdf[k]-logsumexp);
  }
}

template<typename T>
__global__ void sphereGMMPdf(T *d_p, T *d_Rnorths, T * d_q,
    T *d_invSigmas, T *d_logNormalizer, T *d_logPi, 
    T* d_logPdf, uint32_t K, uint32_t N) 
{
  __shared__ T p[DIM*K_MAX];
  __shared__ T logNorm[K_MAX];

  const int tid = threadIdx.x;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;

  // caching 
  if(tid < DIM*K) p[tid] = d_p[tid];
  if(DIM*K <= tid && tid < DIM*K+K) logNorm[tid-DIM*K] = d_logNormalizer[tid-DIM*K];
  __syncthreads(); // make sure that ys have been cached

  for(uint32_t id=idx*N_PER_T; id<min(N,(idx+1)*N_PER_T); ++id)
  {
    T q[DIM];
    T x[DIM]; 
#pragma unroll
    for(uint32_t i=0; i<DIM; ++i)
      q[i] = d_q[id*DIM+i]; 
    T pdf[K_MAX]; 
    for(uint32_t k=0; k<K; ++k)
    {
      T *pk = p+k*DIM;
      Log_p<T>(pk,q,x);
      T *d_Rnorthsk = d_Rnorths+k*(DIM-1)*DIM;
      T xNorth[DIM-1];
      // TODO
      xNorth[0] = d_Rnorthsk[0]*x[0] + d_Rnorthsk[2]*x[1] + d_Rnorthsk[4]*x[2];  
      xNorth[1] = d_Rnorthsk[1]*x[0] + d_Rnorthsk[3]*x[1] + d_Rnorthsk[5]*x[2];  
      T *invSk = d_invSigmas + (DIM-1)*(DIM-1)*k;
      pdf[k] = d_logPi[k] + logNorm[k] - 0.5* bTAb_2D(invSk,xNorth);
    }
    T maxPdf = pdf[0];
#pragma unroll
    for(uint32_t k=1; k<K; ++k)
      if(maxPdf < pdf[k]) maxPdf = pdf[k];

    T logsumexp = exp(pdf[0]-maxPdf);
#pragma unroll
    for(uint32_t k=1; k<K; ++k)
      logsumexp += exp(pdf[k]-maxPdf);
    logsumexp = log(logsumexp)+maxPdf;
//#pragma unroll
//    for(uint32_t k=0; k<K; ++k)
//      pdf[k] = exp(pdf[k]-logsumexp);
#pragma unroll
    for(uint32_t k=0; k<K; ++k)
      d_logPdf[id+N*k] = pdf[k] -logsumexp;
    // TODO: it is now actually returning logPdfs! may break some stuff
//      d_logPdf[id+N*k] = exp(pdf[k] -logsumexp);
  }
};

extern void sphereGMMPdf(double *d_p, double *d_Rnorths, double * d_q, 
    double *d_invSigmas, 
    double *d_logNormalizers, double *d_logPi, double* d_logPdf, uint32_t N, 
    uint32_t K)
{
  assert(K <= K_MAX);
  assert(BLOCK_SIZE > K*(DIM+1));

  dim3 threads(BLOCK_SIZE,1,1);
  dim3 blocks(N/(BLOCK_SIZE*N_PER_T)+(N%(BLOCK_SIZE*N_PER_T)>0?1:0),1,1);
  if(DIM ==3)
    sphereGMMPdf_3D<double><<<blocks,threads>>>(d_p,d_Rnorths,d_q,d_invSigmas,
        d_logNormalizers,d_logPi,d_logPdf,K,N);
  else
    sphereGMMPdf<double><<<blocks,threads>>>(d_p,d_Rnorths,d_q,d_invSigmas,
        d_logNormalizers,d_logPi,d_logPdf,K,N);
  checkCudaErrors(cudaDeviceSynchronize());
};


extern void sphereGMMPdf(float *d_p, float *d_Rnorths, float * d_q, 
    float *d_invSigmas, 
    float *d_logNormalizers, float *d_logPi, float* d_logPdf, uint32_t N, 
    uint32_t K)
{
  assert(K <= K_MAX);
  assert(BLOCK_SIZE > K*(DIM+1));

  dim3 threads(BLOCK_SIZE,1,1);
  dim3 blocks(N/(BLOCK_SIZE*N_PER_T)+(N%(BLOCK_SIZE*N_PER_T)>0?1:0),1,1);
  if(DIM ==3)
    sphereGMMPdf_3D<float><<<blocks,threads>>>(d_p,d_Rnorths,d_q,d_invSigmas,
        d_logNormalizers,d_logPi,d_logPdf,K,N);
  else
    sphereGMMPdf<float><<<blocks,threads>>>(d_p,d_Rnorths,d_q,d_invSigmas,
        d_logNormalizers,d_logPi,d_logPdf,K,N);
  checkCudaErrors(cudaDeviceSynchronize());
};

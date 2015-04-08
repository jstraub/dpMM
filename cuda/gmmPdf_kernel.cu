/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */
 

#include <nvidia/cuda_global.h>

// executions per thread
#define N_PER_T 32
#define BLOCK_SIZE 256
#define K 6


template<typename T>
__global__ void gmmPdf_3D(T * d_x, T *d_invSigmas, T *d_logNormalizer, 
    T *d_logPi, T* d_logPdf, uint32_t N) 
{
  __shared__ T logNorm[K];

  const int tid = threadIdx.x;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;

  // caching 
  if(tid < K) logNorm[tid] = d_logNormalizer[tid];
  __syncthreads(); // make sure that ys have been cached

  for(uint32_t id=idx*N_PER_T; id<min(N,(idx+1)*N_PER_T); ++id)
  {
    T x[DIM]; 
    x[0] = d_x[id*DIM];
    x[1] = d_x[id*DIM+1];
    x[2] = d_x[id*DIM+2];
    T pdf[K]; 
    for(uint32_t k=0; k<K; ++k)
    {
      T *invSk = d_invSigmas + (DIM-1)*(DIM-1)*k;
      pdf[k] = d_logPi[k] + logNorm[k] - 0.5*bTAb_2D(invSk,x);
//      d_logPdf[id+N*k] = - 0.5*bTAb_2D(invSk,xNorth);
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
#pragma unroll
    for(uint32_t k=0; k<K; ++k)
      d_logPdf[id+N*k] = exp(pdf[k]-logsumexp);
  }
}

template<typename T>
__global__ void gmmPdf(T * d_x,
    T *d_invSigmas, T *d_logNormalizer, T *d_logPi, 
    T* d_logPdf, uint32_t N) 
{
  __shared__ T logNorm[K];

  const int tid = threadIdx.x;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;

  // caching 
  if(tid < K) logNorm[tid] = d_logNormalizer[tid];
  __syncthreads(); // make sure that ys have been cached

  for(uint32_t id=idx*N_PER_T; id<min(N,(idx+1)*N_PER_T); ++id)
  {
    T x[DIM]; 
#pragma unroll
    for(uint32_t i=0; i<DIM; ++i)
      x[i] = d_x[id*DIM+i]; 
    T pdf[K]; 
    for(uint32_t k=0; k<K; ++k)
    {
      T *invSk = d_invSigmas + (DIM-1)*(DIM-1)*k;
      //TODO for nD bTAb_2D?
      pdf[k] = d_logPi[k] + logNorm[k] - 0.5* bTAb_2D(invSk,x);
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
      d_logPdf[id+N*k] = exp(pdf[k] -logsumexp);
  }
};

extern void gmmPdf(double * d_x, double *d_invSigmas, 
    double *d_logNormalizers, double *d_logPi, double* d_logPdf, uint32_t N, 
    uint32_t K_)
{
  assert(K_==K);
  assert(BLOCK_SIZE > K*(DIM+1));

  dim3 threads(BLOCK_SIZE,1,1);
  dim3 blocks(N/(BLOCK_SIZE*N_PER_T)+(N%(BLOCK_SIZE*N_PER_T)>0?1:0),1,1);
  if(DIM ==3)
    gmmPdf_3D<double><<<blocks,threads>>>(d_x,d_invSigmas,
        d_logNormalizers,d_logPi,d_logPdf,N);
  else
    gmmPdf<double><<<blocks,threads>>>(d_x,d_invSigmas,
        d_logNormalizers,d_logPi,d_logPdf,N);
  checkCudaErrors(cudaDeviceSynchronize());
};


extern void gmmPdf(float * d_x, float *d_invSigmas, 
    float *d_logNormalizers, float *d_logPi, float* d_logPdf, uint32_t N, 
    uint32_t K_)
{
  assert(K_==K);
  assert(BLOCK_SIZE > K*(DIM+1));

  dim3 threads(BLOCK_SIZE,1,1);
  dim3 blocks(N/(BLOCK_SIZE*N_PER_T)+(N%(BLOCK_SIZE*N_PER_T)>0?1:0),1,1);
  if(DIM ==3)
    gmmPdf_3D<float><<<blocks,threads>>>(d_x,d_invSigmas,
        d_logNormalizers,d_logPi,d_logPdf,N);
  else
    gmmPdf<float><<<blocks,threads>>>(d_x,d_invSigmas,
        d_logNormalizers,d_logPi,d_logPdf,N);
  checkCudaErrors(cudaDeviceSynchronize());
};

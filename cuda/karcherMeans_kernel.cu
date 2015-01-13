/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */
 

#include "cuda_global.h"
#include <stdio.h>

// executions per thread
#define N_PER_T 16
#define BLOCK_SIZE 256
#define K 6

template<typename T, uint32_t BLK_SIZE>
__global__ void karcherMean_kernel_3D(T *d_p, T *d_q, uint32_t *z, 
    T *mu_karch, uint32_t N)
{
  __shared__ T p[DIM*K];
  // one J per column; BLK_SIZE columns; per column first 3 first col of J, 
  // second 3 columns second cols of J 
  // forth row is number of associated points
  __shared__ T mu[BLK_SIZE*(DIM+1)*K];
  //__shared__ T mu[BLK_SIZE*(DIM+1)*K];

  const int tid = threadIdx.x;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;


  // caching 
  if(tid < DIM*K) p[tid] = d_p[tid];
#pragma unroll
  for(int s=0; s<K*(DIM+1); ++s) {
    // this is almost certainly bad ordering
    mu[tid*K*(DIM+1)+s] = 0.0;
  }
  __syncthreads(); // make sure that ys have been cached


  for(int id=idx*N_PER_T; id<min(N,(idx+1)*N_PER_T); ++id)
  {
    uint32_t k = z[id];
//    if(k<K)
//    {
    T *pk = p+k*DIM;
    T x[DIM]; 
    T q[DIM];
    q[0] = d_q[id*DIM+0]; 
    q[1] = d_q[id*DIM+1]; 
    q[2] = d_q[id*DIM+2];
    // project into tangent space
    Log_p<T>(pk,q,x);
    //      mu[tid*K*(DIM+1)] = T(tid);
    //      return;
    //TODO somewhere in here is a bug
    mu[tid*K*(DIM+1)+(DIM+1)*k+0] += x[0];
    mu[tid*K*(DIM+1)+(DIM+1)*k+1] += x[1];
    mu[tid*K*(DIM+1)+(DIM+1)*k+2] += x[2];
    mu[tid*K*(DIM+1)+(DIM+1)*k+3] += 1.0f;
//    }
  }

//  T *muTid = mu + tid*K*(DIM+1);

  __syncthreads(); //sync the threads
#pragma unroll
  for(int s=(BLK_SIZE)/2; s>1; s>>=1) {
    if(tid < s)
    {
      const int si = s*K*(DIM+1);
      const int tidk = tid*K*(DIM+1);
      // sum the whole K*(DIM+1) matrices up
#pragma unroll
      for( int i=0; i<K*(DIM+1); ++i) {
//        muTid[i] += muTid[si+i];
        mu[tidk+i] += mu[si+tidk+i];
      }
    }
    __syncthreads();
  }
  if(tid<K*(DIM+1)) {
    // sum the last two remaining matrixes directly into global memory
    atomicAdd_<T>(&mu_karch[tid],mu[tid]+mu[tid+K*(DIM+1)]);
//    atomicAdd(&mu_karch[tid],mu[tid]+mu[tid+K*(DIM+1)]);
  }
}

//TODO
template<typename T, uint32_t BLK_SIZE>
__global__ void karcherMean_kernel(T *d_p, T *d_q, uint32_t *z, 
    T *mu_karch, uint32_t N) //, T *N)
{
  __shared__ T p[DIM*K];
  // one J per column; BLK_SIZE columns; per column first 3 first col of J, 
  // second 3 columns second cols of J 
  // forth row is number of associated points
  __shared__ T mu[BLK_SIZE*(DIM+1)*K];

  const int tid = threadIdx.x;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;

  // caching 
  if(tid < DIM*K) p[tid] = d_p[tid];
#pragma unroll
  for(int s=0; s<K*(DIM+1); ++s) {
    // this is almost certainly bad ordering
    mu[tid*K*(DIM+1)+s] = 0.0f;
  }
  __syncthreads(); // make sure that ys have been cached

  for(int id=idx*N_PER_T; id<min(N,(idx+1)*N_PER_T); ++id)
  {
      uint32_t k = z[id];
      T *pk = p+k*DIM;
      T q[DIM];
      T x[DIM];
#pragma unroll
      for(int i=1; i<DIM; ++i)
        q[i] = d_q[id*DIM+i]; 
      // project into tangent space
      Log_p<T>(pk,q,x);
#pragma unroll
      for(int i=0; i<DIM; ++i)
        mu[tid*K*(DIM+1)+(DIM+1)*k+i] += x[i]; //(q[i] - pk[i]*dot)*sinc;
      mu[tid*K*(DIM+1)+(DIM+1)*k+DIM] += 1.0f;
  }

  __syncthreads(); //sync the threads
#pragma unroll
  for(int s=(BLK_SIZE)/2; s>1; s>>=1) {
    if(tid < s)
    {
      const uint32_t si = s*K*(DIM+1);
      const uint32_t tidk = tid*K*(DIM+1);
      // sum the whole K*(DIM+1) matrices up
#pragma unroll
      for(int k=0; k<K*(DIM+1); ++k) {
        mu[tidk+k] += mu[si+k];
      }
    }
    __syncthreads();
  }
  if(tid<K*(DIM+1)) {
    // sum the last two remaining matrixes directly into global memory
    atomicAdd_<T>(&mu_karch[tid],mu[tid]+mu[tid+K*(DIM+1)]);
  }
}

extern void meanInTpS2_gpu(double *d_p, double *d_mu_karch, double *d_q, 
    uint32_t *d_z , uint32_t N, uint32_t K_)
{
  const uint32_t BLK_SIZE = BLOCK_SIZE/2;
  assert(K_==K);
  assert(BLK_SIZE > K*(DIM+1));

//  printf("meanInTpS2_gpu<double>\n");

  dim3 threads(BLK_SIZE,1,1);
  dim3 blocks(N/(BLK_SIZE*N_PER_T)+(N%(BLK_SIZE*N_PER_T)>0?1:0),1,1);
  if(DIM ==3)
  {
//    double* dbg;
//    checkCudaErrors(cudaMalloc((void **)&dbg, N*sizeof(double))); 
    karcherMean_kernel_3D<double,BLK_SIZE><<<blocks,threads>>>(d_p,d_q, d_z, 
        d_mu_karch,N);
  }
  else
    karcherMean_kernel<double,BLK_SIZE><<<blocks,threads>>>(d_p,d_q, d_z, 
        d_mu_karch,N);
  checkCudaErrors(cudaDeviceSynchronize());
};


extern void meanInTpS2_gpu(float *d_p, float *d_mu_karch, float *d_q, 
    uint32_t *d_z , uint32_t N, uint32_t K_)
{
  const uint32_t BLK_SIZE = BLOCK_SIZE/2;
  assert(K_==K);
  assert(BLK_SIZE > K*(DIM+1));
//  printf("meanInTpS2_gpu<float> N=%d BLK_SIZE=%d \n",N,BLK_SIZE);

  dim3 threads(BLK_SIZE,1,1);
  dim3 blocks(N/(BLK_SIZE*N_PER_T)+(N%(BLK_SIZE*N_PER_T)>0?1:0),1,1);
  if(DIM ==3)
  {
//    float* dbg;
//    checkCudaErrors(cudaMalloc((void **)&dbg, N*sizeof(float))); 
    karcherMean_kernel_3D<float,BLK_SIZE><<<blocks,threads>>>(d_p,d_q, d_z,
        d_mu_karch,N);
//    float* h_dbg;
//    h_dbg = (float*) malloc(N*sizeof(float));
//    checkCudaErrors(cudaMemcpy(h_dbg,dbg, N*sizeof(float), cudaMemcpyDeviceToHost));
//    for(uint32_t i=0; i<N; ++i)
//      printf("%lf ",h_dbg[i]);
  }else
    karcherMean_kernel<float,BLK_SIZE><<<blocks,threads>>>(d_p,d_q, d_z, 
        d_mu_karch,N);
  checkCudaErrors(cudaDeviceSynchronize());
};


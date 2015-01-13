/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */
 

#include "cuda_global.h"

// executions per thread
#define N_PER_T 32
#define BLOCK_SIZE 256
//#define K 6
#define SS_DIM ((DIM-1)+1+(DIM-1)*(DIM-1))

template<typename T, uint32_t K, uint32_t BLK_SIZE>
__global__ void sufficientStatistics_kernel(T *d_x,
    uint32_t *z, uint32_t N, uint32_t k0, T *SSs) 
{
  // sufficient statistics for whole blocksize
  // 2 (x in TpS @north) + 1 (count) + 4 (outer product in TpS @north)
  // all fo that times 6 for the different axes
  __shared__ T xSSs[BLK_SIZE*SS_DIM*K];

  //const int tid = threadIdx.x;
  const int tid = threadIdx.x;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;

  // caching 
#pragma unroll
  for(int s=0; s< K*SS_DIM; ++s) {
    // this is almost certainly bad ordering
    xSSs[tid*K*SS_DIM+s] = 0.0f;
  }
  __syncthreads(); // make sure that ys have been cached

  for(int id=idx*N_PER_T; id<min(N,(idx+1)*N_PER_T); ++id)
  {
    int32_t k = z[id]-k0;
    if(0 <= k && k < K)
    {
      // copy q into local memory
      T x[DIM-1];
      x[0] = d_x[id*(DIM-1)+0]; 
      x[1] = d_x[id*(DIM-1)+1]; 
      // input sufficient statistics
      xSSs[tid*SS_DIM*K+k*SS_DIM+0] += x[0];
      xSSs[tid*SS_DIM*K+k*SS_DIM+1] += x[1];
      xSSs[tid*SS_DIM*K+k*SS_DIM+2] += x[0]*x[0];
      xSSs[tid*SS_DIM*K+k*SS_DIM+3] += x[1]*x[0];
      xSSs[tid*SS_DIM*K+k*SS_DIM+4] += x[0]*x[1];
      xSSs[tid*SS_DIM*K+k*SS_DIM+5] += x[1]*x[1];
      xSSs[tid*SS_DIM*K+k*SS_DIM+6] += 1.0f;
    }
  }

  // old reduction.....
  __syncthreads(); //sync the threads
#pragma unroll
  for(int s=(BLK_SIZE)/2; s>1; s>>=1) {
    if(tid < s)
    {
      const uint32_t si = s*K*SS_DIM;
      const uint32_t tidk = tid*K*SS_DIM;
#pragma unroll
      for( int k=0; k<K*SS_DIM; ++k) {
        xSSs[tidk+k] += xSSs[si+tidk+k];
      }
    }
    __syncthreads();
  }
  if(tid < K*SS_DIM) {
    // sum the last two remaining matrixes directly into global memory
    atomicAdd_<T>(&SSs[tid],xSSs[tid]+xSSs[tid+K*SS_DIM]);
  }
}

extern void sufficientStatistics_gpu( double *d_x, uint32_t *d_z , uint32_t N, 
    uint32_t k0, uint32_t K, double *d_SSs)
{
  const uint32_t BLK_SIZE = BLOCK_SIZE/2;
  assert(BLK_SIZE > DIM*K+DIM*(DIM-1)*K);

  dim3 threads(BLK_SIZE,1,1);
  dim3 blocks(N/(BLK_SIZE*N_PER_T)+(N%(BLK_SIZE*N_PER_T)>0?1:0),1,1);
  if(K == 1){
    sufficientStatistics_kernel<double,1,BLK_SIZE><<<blocks,threads>>>(
        d_x, d_z,N,k0,d_SSs);
  }else if(K==2){
    sufficientStatistics_kernel<double,2,BLK_SIZE><<<blocks,threads>>>(
        d_x, d_z,N,k0,d_SSs);
  }else if(K==3){
    sufficientStatistics_kernel<double,3,BLK_SIZE><<<blocks,threads>>>(
        d_x, d_z,N,k0,d_SSs);
  }else if(K==4){
    sufficientStatistics_kernel<double,4,BLK_SIZE><<<blocks,threads>>>(
        d_x, d_z,N,k0,d_SSs);
  }else if(K==5){
    sufficientStatistics_kernel<double,5,BLK_SIZE><<<blocks,threads>>>(
        d_x, d_z,N,k0,d_SSs);
  }else if(K==6){
    sufficientStatistics_kernel<double,6,BLK_SIZE><<<blocks,threads>>>(
        d_x, d_z,N,k0,d_SSs);
  }else{
    assert(false);
  }
  checkCudaErrors(cudaDeviceSynchronize());
};

extern void sufficientStatistics_gpu(float *d_x, uint32_t *d_z, 
    uint32_t N, uint32_t k0, uint32_t K, float *d_SSs)
{
  const uint32_t BLK_SIZE = BLOCK_SIZE;
  assert(BLK_SIZE > DIM*K+DIM*(DIM-1)*K);

  dim3 threads(BLK_SIZE,1,1);
  dim3 blocks(N/(BLK_SIZE*N_PER_T)+(N%(BLK_SIZE*N_PER_T)>0?1:0),1,1);
  if(K == 1){
    sufficientStatistics_kernel<float,1,BLK_SIZE><<<blocks,threads>>>(
        d_x, d_z,N,k0,d_SSs);
  }else if(K==2){
    sufficientStatistics_kernel<float,2,BLK_SIZE><<<blocks,threads>>>(
        d_x, d_z,N,k0,d_SSs);
  }else if(K==3){
    sufficientStatistics_kernel<float,3,BLK_SIZE><<<blocks,threads>>>(
        d_x, d_z,N,k0,d_SSs);
  }else if(K==4){
    sufficientStatistics_kernel<float,4,BLK_SIZE><<<blocks,threads>>>(
        d_x, d_z,N,k0,d_SSs);
  }else if(K==5){
    sufficientStatistics_kernel<float,5,BLK_SIZE><<<blocks,threads>>>(
        d_x, d_z,N,k0,d_SSs);
  }else if(K==6){
    sufficientStatistics_kernel<float,6,BLK_SIZE><<<blocks,threads>>>(
        d_x, d_z,N,k0,d_SSs);
  }else{
    assert(false);
  }
  checkCudaErrors(cudaDeviceSynchronize());
};

template<typename T, uint32_t K, uint32_t BLK_SIZE>
__global__ void sufficientStatisticsOnTpS2_kernel(T *d_p, T *Rnorths, T *d_q,
    uint32_t *z, uint32_t N, uint32_t k0, T *SSs) 
{
  //const uint32_t SS_DIM = ((DIM-1)+1+(DIM-1)*(DIM-1));
  __shared__ T p[DIM*K];
  // sufficient statistics for whole blocksize
  // 2 (x in TpS @north) + 1 (count) + 4 (outer product in TpS @north)
  // all fo that times 6 for the different axes
  __shared__ T xSSs[BLK_SIZE*SS_DIM*K];
//  __shared__ T sRnorths[DIM*(DIM-1)*K];

  //const int tid = threadIdx.x;
  const int tid = threadIdx.x;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;

  // caching 
  if(tid < DIM*K) p[tid] = d_p[tid];
//  if(DIM*K <= tid && tid < DIM*K+DIM*(DIM-1)*K) 
//    sRnorths[tid-DIM*K] = Rnorths[tid-DIM*K];
#pragma unroll
  for(int s=0; s< K*SS_DIM; ++s) {
    // this is almost certainly bad ordering
    xSSs[tid*K*SS_DIM+s] = 0.0f;
  }
  __syncthreads(); // make sure that ys have been cached

  for(int id=idx*N_PER_T; id<min(N,(idx+1)*N_PER_T); ++id)
  {
    int32_t k = z[id]-k0;
    if(0 <= k && k < K)
    {
      T *pk = p+k*DIM;
      // copy q into local memory
      T q[DIM];
      q[0] = d_q[id*DIM+0]; 
      q[1] = d_q[id*DIM+1]; 
      q[2] = d_q[id*DIM+2];
      T x[DIM]; 
      // transform to TpS^2
      Log_p<T>(pk,q,x);
      // rotate up to north pole
      T *sRnorthsk = Rnorths+k*(DIM-1)*DIM;
      T xNorth[2];
      xNorth[0] = sRnorthsk[0]*x[0] + sRnorthsk[2]*x[1] + sRnorthsk[4]*x[2];  
      xNorth[1] = sRnorthsk[1]*x[0] + sRnorthsk[3]*x[1] + sRnorthsk[5]*x[2];  
      // input sufficient statistics
      xSSs[tid*SS_DIM*K+k*SS_DIM+0] += xNorth[0];
      xSSs[tid*SS_DIM*K+k*SS_DIM+1] += xNorth[1];
      xSSs[tid*SS_DIM*K+k*SS_DIM+2] += xNorth[0]*xNorth[0];
      xSSs[tid*SS_DIM*K+k*SS_DIM+3] += xNorth[1]*xNorth[0];
      xSSs[tid*SS_DIM*K+k*SS_DIM+4] += xNorth[0]*xNorth[1];
      xSSs[tid*SS_DIM*K+k*SS_DIM+5] += xNorth[1]*xNorth[1];
      xSSs[tid*SS_DIM*K+k*SS_DIM+6] += 1.0f;
    }
  }

  // old reduction.....
  __syncthreads(); //sync the threads
#pragma unroll
  for(int s=(BLK_SIZE)/2; s>1; s>>=1) {
    if(tid < s)
    {
      const uint32_t si = s*K*SS_DIM;
      const uint32_t tidk = tid*K*SS_DIM;
#pragma unroll
      for( int k=0; k<K*SS_DIM; ++k) {
        xSSs[tidk+k] += xSSs[si+tidk+k];
      }
    }
    __syncthreads();
  }
  if(tid < K*SS_DIM) {
    // sum the last two remaining matrixes directly into global memory
    atomicAdd_<T>(&SSs[tid],xSSs[tid]+xSSs[tid+K*SS_DIM]);
  }
}

extern void sufficientStatisticsOnTpS2_gpu(double *d_p, 
  double *d_Rnorths, double *d_q, uint32_t *d_z , uint32_t N, uint32_t k0, 
  uint32_t K, double *d_SSs)
{
  const uint32_t BLK_SIZE = BLOCK_SIZE/2;
  assert(BLK_SIZE > DIM*K+DIM*(DIM-1)*K);

  dim3 threads(BLK_SIZE,1,1);
  dim3 blocks(N/(BLK_SIZE*N_PER_T)+(N%(BLK_SIZE*N_PER_T)>0?1:0),1,1);
  if(K == 1){
    sufficientStatisticsOnTpS2_kernel<double,1,BLK_SIZE><<<blocks,threads>>>(
        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
  }else if(K==2){
    sufficientStatisticsOnTpS2_kernel<double,2,BLK_SIZE><<<blocks,threads>>>(
        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
  }else if(K==3){
    sufficientStatisticsOnTpS2_kernel<double,3,BLK_SIZE><<<blocks,threads>>>(
        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
  }else if(K==4){
    sufficientStatisticsOnTpS2_kernel<double,4,BLK_SIZE><<<blocks,threads>>>(
        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
  }else if(K==5){
    sufficientStatisticsOnTpS2_kernel<double,5,BLK_SIZE><<<blocks,threads>>>(
        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
  }else if(K==6){
    sufficientStatisticsOnTpS2_kernel<double,6,BLK_SIZE><<<blocks,threads>>>(
        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
//  }else if(K==7){
//    sufficientStatisticsOnTpS2_kernel<double,7,BLK_SIZE><<<blocks,threads>>>(
//        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
//  }else if(K==8){
//    sufficientStatisticsOnTpS2_kernel<double,8,BLK_SIZE><<<blocks,threads>>>(
//        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
//  }else if(K==9){
//    sufficientStatisticsOnTpS2_kernel<double,9,BLK_SIZE><<<blocks,threads>>>(
//        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
//  }else if(K==10){
//    sufficientStatisticsOnTpS2_kernel<double,10,BLK_SIZE><<<blocks,threads>>>(
//        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
//  }else if(K==11){
//    sufficientStatisticsOnTpS2_kernel<double,11,BLK_SIZE><<<blocks,threads>>>(
//        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
//  }else if(K==12){
//    sufficientStatisticsOnTpS2_kernel<double,12,BLK_SIZE><<<blocks,threads>>>(
//        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
//  }else if(K==13){
//    sufficientStatisticsOnTpS2_kernel<double,13,BLK_SIZE><<<blocks,threads>>>(
//        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
//  }else if(K==14){
//    sufficientStatisticsOnTpS2_kernel<double,14,BLK_SIZE><<<blocks,threads>>>(
//        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
//  }else if(K==15){
//    sufficientStatisticsOnTpS2_kernel<double,15,BLK_SIZE><<<blocks,threads>>>(
//        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
//  }else if(K==16){
//    sufficientStatisticsOnTpS2_kernel<double,16,BLK_SIZE><<<blocks,threads>>>(
//        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
//  }else if(K==17){
//    sufficientStatisticsOnTpS2_kernel<double,17,BLK_SIZE><<<blocks,threads>>>(
//        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
//  }else if(K==18){
//    sufficientStatisticsOnTpS2_kernel<double,18,BLK_SIZE><<<blocks,threads>>>(
//        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
//  }else if(K==19){
//    sufficientStatisticsOnTpS2_kernel<double,19,BLK_SIZE><<<blocks,threads>>>(
//        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
//  }else if(K==20){
//    sufficientStatisticsOnTpS2_kernel<double,20,BLK_SIZE><<<blocks,threads>>>(
//        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
//  }else if(K==21){
//    sufficientStatisticsOnTpS2_kernel<double,21,BLK_SIZE><<<blocks,threads>>>(
//        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
//  }else if(K==22){
//    sufficientStatisticsOnTpS2_kernel<double,22,BLK_SIZE><<<blocks,threads>>>(
//        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
//  }else if(K==23){
//    sufficientStatisticsOnTpS2_kernel<double,23,BLK_SIZE><<<blocks,threads>>>(
//        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
//  }else if(K==24){
//    sufficientStatisticsOnTpS2_kernel<double,24,BLK_SIZE><<<blocks,threads>>>(
//        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
  }else{
    assert(false);
  }
  checkCudaErrors(cudaDeviceSynchronize());
};

extern void sufficientStatisticsOnTpS2_gpu(float *d_p, 
  float *d_Rnorths, float *d_q, uint32_t *d_z , uint32_t N, uint32_t k0, 
  uint32_t K, float *d_SSs)
{
  const uint32_t BLK_SIZE = BLOCK_SIZE;
  assert(BLK_SIZE > DIM*K+DIM*(DIM-1)*K);

  dim3 threads(BLK_SIZE,1,1);
  dim3 blocks(N/(BLK_SIZE*N_PER_T)+(N%(BLK_SIZE*N_PER_T)>0?1:0),1,1);
  if(K == 1){
    sufficientStatisticsOnTpS2_kernel<float,1,BLK_SIZE><<<blocks,threads>>>(
        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
  }else if(K==2){
    sufficientStatisticsOnTpS2_kernel<float,2,BLK_SIZE><<<blocks,threads>>>(
        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
  }else if(K==3){
    sufficientStatisticsOnTpS2_kernel<float,3,BLK_SIZE><<<blocks,threads>>>(
        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
  }else if(K==4){
    sufficientStatisticsOnTpS2_kernel<float,4,BLK_SIZE><<<blocks,threads>>>(
        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
  }else if(K==5){
    sufficientStatisticsOnTpS2_kernel<float,5,BLK_SIZE><<<blocks,threads>>>(
        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
  }else if(K==6){
    sufficientStatisticsOnTpS2_kernel<float,6,BLK_SIZE><<<blocks,threads>>>(
        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
//  }else if(K==7){
//    sufficientStatisticsOnTpS2_kernel<float,7,BLK_SIZE><<<blocks,threads>>>(
//        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
//  }else if(K==8){
//    sufficientStatisticsOnTpS2_kernel<float,8,BLK_SIZE><<<blocks,threads>>>(
//        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
//  }else if(K==9){
//    sufficientStatisticsOnTpS2_kernel<float,9,BLK_SIZE><<<blocks,threads>>>(
//        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
//  }else if(K==10){
//    sufficientStatisticsOnTpS2_kernel<float,10,BLK_SIZE><<<blocks,threads>>>(
//        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
//  }else if(K==11){
//    sufficientStatisticsOnTpS2_kernel<float,11,BLK_SIZE><<<blocks,threads>>>(
//        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
//  }else if(K==12){
//    sufficientStatisticsOnTpS2_kernel<float,12,BLK_SIZE><<<blocks,threads>>>(
//        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
//  }else if(K==13){
//    sufficientStatisticsOnTpS2_kernel<float,13,BLK_SIZE><<<blocks,threads>>>(
//        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
//  }else if(K==14){
//    sufficientStatisticsOnTpS2_kernel<float,14,BLK_SIZE><<<blocks,threads>>>(
//        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
//  }else if(K==15){
//    sufficientStatisticsOnTpS2_kernel<float,15,BLK_SIZE><<<blocks,threads>>>(
//        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
//  }else if(K==16){
//    sufficientStatisticsOnTpS2_kernel<float,16,BLK_SIZE><<<blocks,threads>>>(
//        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
//  }else if(K==17){
//    sufficientStatisticsOnTpS2_kernel<float,17,BLK_SIZE><<<blocks,threads>>>(
//        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
//  }else if(K==18){
//    sufficientStatisticsOnTpS2_kernel<float,18,BLK_SIZE><<<blocks,threads>>>(
//        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
//  }else if(K==19){
//    sufficientStatisticsOnTpS2_kernel<float,19,BLK_SIZE><<<blocks,threads>>>(
//        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
//  }else if(K==20){
//    sufficientStatisticsOnTpS2_kernel<float,20,BLK_SIZE><<<blocks,threads>>>(
//        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
//  }else if(K==21){
//    sufficientStatisticsOnTpS2_kernel<float,21,BLK_SIZE><<<blocks,threads>>>(
//        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
//  }else if(K==22){
//    sufficientStatisticsOnTpS2_kernel<float,22,BLK_SIZE><<<blocks,threads>>>(
//        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
//  }else if(K==23){
//    sufficientStatisticsOnTpS2_kernel<float,23,BLK_SIZE><<<blocks,threads>>>(
//        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
//  }else if(K==24){
//    sufficientStatisticsOnTpS2_kernel<float,24,BLK_SIZE><<<blocks,threads>>>(
//        d_p,d_Rnorths,d_q, d_z,N,k0,d_SSs);
  }else{
    assert(false);
  }
  checkCudaErrors(cudaDeviceSynchronize());
};

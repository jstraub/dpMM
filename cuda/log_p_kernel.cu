/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */
 

#include "cuda_global.h"

template<typename T>
__global__ void Log_p_kernel_noBuff(T *p, T *q, T *Rs, uint32_t *z, 
  uint32_t K, uint32_t N, T *x)
{
  const uint32_t idx=threadIdx.x + blockIdx.x*blockDim.x;

  if(idx < N)
  {
    T xD_i[DIM];
    uint32_t k=z[idx];
    T *pk = p+k*DIM;
    T *q_i = q+idx*DIM;
//    // precompute dot, sinc
//    T qiTp = q_i[0]*p_i[0];
//    #pragma unroll
//    for(uint32_t i=1; i<DIM; ++i)
//      qiTp += q_i[i]*p_i[i]; 
//    T dot = max(-1.0f,min(1.0f,qiTp));
//    T sinc = acosf(dot);
//    if(sinc <1e-7)
//      sinc = 1.0f;
//    else
//      sinc = sinc/sin(sinc);
//    // compute Log_p
//    #pragma unroll
//    for(uint32_t i=0; i<DIM; ++i)
//      xD_i[i] = (q_i[i] - p_i[i]*dot)*sinc; 
    Log_p<T>(pk,q_i,xD_i);
    T *x_i = x+idx*(DIM-1);
    T *R = Rs+k*DIM*(DIM-1); // last row of Rs is not transmitted, since we dont need
    // rotate to north
    #pragma unroll
    for(uint32_t i=0; i<DIM-1; ++i)
    {
      x_i[i] = R[i]*xD_i[0]; 
      #pragma unroll
      for(uint32_t j=1; j<DIM; ++j)
        x_i[i] += xD_i[j]*R[i+(DIM-1)*j]; 
    }
  }
}

template<typename T>
__global__ void Log_p_kernel_3D(T *p, T *q, T *Rs, uint32_t *z, 
  uint32_t K, uint32_t N, T *x)
{
  const uint32_t idx=threadIdx.x + blockIdx.x*blockDim.x;

  if(idx < N)
  {
    T xD_i[3];
    uint32_t k=z[idx];
    T *pk = p+k*3;
    T *q_i = q+idx*3;
    Log_p<T>(pk,q_i,xD_i);
    T *x_i = x+idx*(2);
    T *R = Rs+k*3*(2); // last row of Rs is not transmitted, since we dont need
    // rotate to north
    x_i[0] = R[0]*xD_i[0] + R[2]*xD_i[1] + R[4]*xD_i[2];
    x_i[1] = R[1]*xD_i[0] + R[3]*xD_i[1] + R[5]*xD_i[2];
  }
}

/*
 * assumes p,q,x are column major
 */
template<typename T>
__global__ void Log_p_kernel(T *p, T *q, T *Rs, uint32_t *z, 
  uint32_t K, uint32_t N, T *x)
{
  const uint32_t idx=threadIdx.x + blockIdx.x*blockDim.x;

  if(idx < N)
  {
    T p_i[DIM];
    T q_i[DIM];
    T xD_i[DIM];
    T x_i[DIM-1];

    T R[DIM*(DIM-1)];
    uint32_t k=z[idx];
    T *pp_i = p+k*DIM;
    T *qq_i = q+idx*DIM;
    T *Rk = Rs+k*DIM*(DIM-1); // last row of Rs is not transmitted, since we dont need

    #pragma unroll
    for(uint32_t i=0; i<DIM; ++i)
    {
      p_i[i]=*(pp_i++);
      q_i[i]=*(qq_i++);
    }
    #pragma unroll
    for(uint32_t i=0; i<DIM*(DIM-1); ++i)
      R[i] = *(Rk++);
    // precompute dot, sinc
    T qiTp = q_i[0]*p_i[0];
    #pragma unroll
    for(uint32_t i=1; i<DIM; ++i)
      qiTp += q_i[i]*p_i[i]; 
    T dot = max(-1.0f,min(1.0f,qiTp));
    T sinc = acosf(dot);
    if(sinc <1e-7)
      sinc = 1.0f;
    else
      sinc = sinc/sin(sinc);
    // compute Log_p
    #pragma unroll
    for(uint32_t i=0; i<DIM; ++i)
      xD_i[i] = (q_i[i] - p_i[i]*dot)*sinc; 
    // rotate to north
    #pragma unroll
    for(uint32_t i=0; i<DIM-1; ++i)
    {
      x_i[i] = R[i]*xD_i[0]; 
      #pragma unroll
      for(uint32_t j=1; j<DIM; ++j)
        x_i[i] += xD_i[j]*R[i+(DIM-1)*j]; 
    }
    // write to memory
    T *xx_i = x+idx*(DIM-1);
    #pragma unroll
    for(uint32_t i=0; i<DIM-1; ++i)
      *(xx_i++) = x_i[i];
  }
}


extern void Log_p_gpu(double *p, double *q, double *Rs, uint32_t *z, 
  uint32_t K, uint32_t N, double *x)
{
  dim3 threads(256,1,1);
  dim3 blocks(N/256+(N%256>0?1:0), 1,1);
  if(DIM ==3)
    // interestingly not faster than nDIM
    Log_p_kernel_3D<double><<<blocks,threads>>>(p,q,Rs,z,K,N,x);
  else
    Log_p_kernel_noBuff<double><<<blocks,threads>>>(p,q,Rs,z,K,N,x);
  checkCudaErrors(cudaDeviceSynchronize());
}

extern void Log_p_gpu(float *p, float *q, float *Rs, uint32_t *z, 
  uint32_t K, uint32_t N, float *x)
{
  dim3 threads(256,1,1);
  dim3 blocks(N/256+(N%256>0?1:0), 1,1);
  if(DIM ==3)
    // interestingly not faster than nDIM
    Log_p_kernel_3D<float><<<blocks,threads>>>(p,q,Rs,z,K,N,x);
  else
    Log_p_kernel_noBuff<float><<<blocks,threads>>>(p,q,Rs,z,K,N,x);
  checkCudaErrors(cudaDeviceSynchronize());
}

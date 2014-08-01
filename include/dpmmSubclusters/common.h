#pragma once

#include <stdio.h>
#include <stdlib.h>

#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"
#include "gsl/gsl_permutation.h"
#include "gsl/gsl_cdf.h"

#ifdef DEBUG_ARR                                                                
   #define arr(type)    array<type>                                             
#else                                                                           
   #define arr(type)    type*                                                   
#endif

#define mexPrintf printf
#define mexErrMsgTxt printf

#ifndef PI
#define PI 3.14159265
#endif

#ifndef NDEBUG
// DEBUG
  inline int omp_get_thread_num()
  {return 0;};
#endif

template <typename T>                                                        
T* allocate_memory(unsigned int length)                                      
{                                                                            
  return new T[length];                                                     
  //return (T*)mxCalloc(length, sizeof(T));                                 
}                                                                            
template <typename T>                                                        
T* allocate_memory(unsigned int length, T value)                             
{                                                                            
  T* ptr = new T[length];                                                   
  //T* ptr = (T*)mxCalloc(length, sizeof(T));                               
  for (int i=length-1; i>=0; i--)                                           
    ptr[i] = value;                                                        
  return ptr;                                                               
}                                                                            
template <typename T>                                                        
void deallocate_memory(T* ptr)                                               
{                                                                            
  delete[] ptr;                                                             
  //mxFree(ptr);                                                            
}                                                                            
template <typename T>                                                        
void copy_memory(T* ptr1, const T* ptr2, size_t num)                         
{                                                                            
  memcpy(ptr1, ptr2, num);                                                  
}                                                                            
//template <typename T>                                                        
//void copy_memory(T* ptr1, const array<T> ptr2, size_t num)                   
//{                                                                            
//  memcpy(ptr1, ptr2.getData(), num);                                        
//}                                                                            
template <typename T>                                                        
void set_memory(T* ptr1, const T value, size_t num)                          
{                                                                            
  memset(ptr1, value, num);                                                 
}

inline double my_rand(gsl_rng* r)                                               
{                                                                               
   return gsl_rng_uniform_pos(r);                                               
}                                                                               
inline unsigned long int my_randi(gsl_rng* r)                                   
{                                                                               
   return gsl_rng_get(r);                                                       
}

// TODO: used to initialize random number generators - so probably do something 
// different here
inline int mx_rand(void)
{ return 0;};
inline gsl_rng* initialize_gsl_rand()
{
   const gsl_rng_type * T;
   gsl_rng * r;
   gsl_rng_env_setup();

   T = gsl_rng_default;
   r = gsl_rng_alloc (T);
   gsl_rng_set(r, (unsigned long)(mx_rand()*pow(2,32)));
   return r;
}
inline gsl_rng* initialize_gsl_rand(int seed)
{
   const gsl_rng_type * T;
   gsl_rng * r;
   gsl_rng_env_setup();

   T = gsl_rng_default;
   r = gsl_rng_alloc (T);
   gsl_rng_set(r, (unsigned long)(seed));
   return r;
}
inline gsl_rng* initialize_gsl_rand(double seed)
{
   const gsl_rng_type * T;
   gsl_rng * r;
   gsl_rng_env_setup();

   T = gsl_rng_default;
   r = gsl_rng_alloc (T);
   gsl_rng_set(r, (unsigned long)(seed*pow(2,32)));
   return r;
}

inline double mxGetInf()
{ return 1.0/0.0;}

// =============================================================================
// == reduction_array2.h
// == --------------------------------------------------------------------------
// == A reduction array class that can be used with OpenMP. Current OpenMP
// == software only allows the reduction of an array into a scalar. This class
// == allows one to reduce an array into an array. Also, since the code here
// == implements a reduction similar to OpenMP without actually using OpenMP's
// == reduction clause, the reduction_array2 class allows users to define their
// == own reduction functions and reductions on user-specified classes.
// ==
// == An array of size numThreads x D is created. Reduction is performed so
// == that each thread only has write accesses a unique spot in the array.
// ==
// == Notation is as follows:
// ==   numThreads - The number of OpenMP threads that will be running
// ==   D - The dimension of the array to reduce to (if scalar, should be 1)
// ==
// == General usage:
// ==   (1) Before the parallel environment, initialize the reduction array:
// ==     >> reduction_array2<double> my_arr(numThreads,D,initial_value);
// ==
// ==   (2) Inside the parallel for loop, you can reduce with data x with:
// ==     >> my_arr.reduce_XXX(omp_get_thread_num(), bin, x);
// ==       "XXX" can be the predefined "add", "multiply", "max", or "min"
// ==       or you can specify a function pointer to your own function with
// ==     >> my_arr.reduce_function(omp_get_thread_num(), bin, x, func_ptr);
// ==
// ==   (3) After the parallel environment, reduce on the separate threads:
// ==     >> double* output = my_arr.final_reduce_XXX();
// ==       Again, "XXX" can be any of the predefined functions, or you can use
// ==     >> double* output = my_arr.final_reduce_function(func_ptr);
// ==
// ==   (4) output now is an array of length D with the reduced values.
// ==       Do *NOT* attempt to deallocate output or my_arr as they have their
// ==       own destructors.
// ==
// == Notes:
// ==   (1) Because of possible "false sharing", the array size here is actually
// ==       numThreads x D x cache_line. We pad the array so that different
// ==       threads will not access the same cache_line. If the cache line is
// ==       not large enough, please increase ie manually.
// == --------------------------------------------------------------------------
// == Written by Jason Chang 04-14-2013 - jchang7@csail.mit.edu
// =============================================================================

#ifndef _REDUCTION_ARRAY2
#define _REDUCTION_ARRAY2

#include <string.h>
#include <cstdlib>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include "dpmmSubclusters/reduction_array.h"
//#include "array.h"
//#include "helperMEX.h"


#ifndef cache_line
#define cache_line 4
#endif

template <typename T>
class reduction_array2
{
private:
   arr(T) data;
   int numThreads;
   int K;
   int D;

   long offt; // offsets for the threads
   long offk; // offsets for the dimensions

public:
   // --------------------------------------------------------------------------
   // -- reduction_array2
   // --   constructors; initializes the reduction array with a number of
   // -- threads, each with a KxD dimensional vector. The third parameter can be
   // -- specified to give the initializing value.
   // --------------------------------------------------------------------------
   reduction_array2();
   reduction_array2(int thenumThreads, int theK, int theD);
   reduction_array2(int thenumThreads, int theK, int theD, T value);
   virtual ~reduction_array2();

   // --------------------------------------------------------------------------
   // -- init_values
   // --   Initializes all values in the reduciton array to "value"
   // --
   // --   parameters:
   // --     - value : the value to set all elements to
   // --------------------------------------------------------------------------
   void init_values(T value);

   // --------------------------------------------------------------------------
   // -- reduce_XXX
   // --   Performs the reduction "XXX" on the d^th dimension with value
   // --------------------------------------------------------------------------
   void reduce_inc(int t, int k);
   void reduce_add(int t, int k, arr(T) value);

   template <typename T2>
   void reduce_add(int t, int k, long d, T2 value);
   template <typename T2>
   void reduce_add(int t, int k, long* ds, T2* values, int nnz);

   void reduce_add_outerprod(int t, int k, arr(T) value);
   void reduce_multiply(int t, int k, arr(T) value);
   void reduce_min(int t, int k, arr(T) value);
   void reduce_max(int t, int k, arr(T) value);

   // --------------------------------------------------------------------------
   // -- reduce_function
   // --   Performs the reduction specified by the function pointer
   // --------------------------------------------------------------------------
   void reduce_function(int t, int k, arr(T) value, T (*func)(T,T));

   // --------------------------------------------------------------------------
   // -- final_reduce_XXX
   // --   Performs the reduction "XXX" on the threads and returns result
   // --------------------------------------------------------------------------
   arr(T) final_reduce_add();
   arr(T) final_reduce_multiply();
   arr(T) final_reduce_min();
   arr(T) final_reduce_max();

   // --------------------------------------------------------------------------
   // -- final_reduce_function
   // --   The function pointer version of above
   // --------------------------------------------------------------------------
   arr(T) final_reduce_function(T (*func)(T,T));
   
   // --------------------------------------------------------------------------
   // -- final_reduce_ext_XXX
   // --   Performs the reduction "XXX" on the threads into the external array.
   // -- Assumes that ext is already allocated to the correct size.
   // --------------------------------------------------------------------------
   void final_reduce_ext_add(arr(T) ext);
   void final_reduce_ext_multiply(arr(T) ext);

   // --------------------------------------------------------------------------
   // -- collapse_cache_line
   // --   collapses the cache line for the final return
   // --------------------------------------------------------------------------
   void collapse_cache_line();
};


// --------------------------------------------------------------------------
// -- reduction_array2
// --   constructors; initializes the reduction array with a number of
// -- threads, each with a KxD dimensional vector. The third parameter can be
// -- specified to give the initializing value.
// --------------------------------------------------------------------------
template <typename T>
reduction_array2<T>::reduction_array2() :
   numThreads(0), K(0), D(0), data(NULL)
{
}
template <typename T>
reduction_array2<T>::reduction_array2(int thenumThreads, int theK, int theD) :
   numThreads(thenumThreads), K(theK), D(theD)
{
   offk = D + cache_line;
   offt = offk*K;
   data = allocate_memory<T>(numThreads*offt);
}
template <typename T>
reduction_array2<T>::reduction_array2(int thenumThreads, int theK, int theD, T value) :
   numThreads(thenumThreads), K(theK), D(theD)
{
   offk = D + cache_line;
   offt = offk*K;
   data = allocate_memory<T>(numThreads*offt);
   set_memory<T>(data, value, sizeof(T)*numThreads*offt);
}
template <typename T>
reduction_array2<T>::~reduction_array2()
{
   if (data!=NULL)
      deallocate_memory(data);;
}


// --------------------------------------------------------------------------
// -- init_values
// --   Initializes all values in the reduciton array to "value"
// --
// --   parameters:
// --     - value : the value to set all elements to
// --------------------------------------------------------------------------
template <typename T>
void reduction_array2<T>::init_values(T value)
{
   set_memory<T>(data, value, sizeof(T)*numThreads*offt);
}


// --------------------------------------------------------------------------
// -- reduce_XXX
// --   Performs the reduction "XXX" on the d^th dimension with value
// --------------------------------------------------------------------------
template <typename T>
inline void reduction_array2<T>::reduce_inc(int t, int k)
{
   long offset = t*offt + k*offk;
   for (int d=0; d<D; d++)
      ++data[offset+d];
}
template <typename T>
inline void reduction_array2<T>::reduce_add(int t, int k, arr(T) value)
{
   long offset = t*offt + k*offk;
   for (int d=0; d<D; d++)
      data[offset+d] += value[d];
}
template <typename T>
template <typename T2>
inline void reduction_array2<T>::reduce_add(int t, int k, long d, T2 value)
{
   long offset = t*offt + k*offk;
   data[offset+d] += value;
}
template <typename T>
template <typename T2>
inline void reduction_array2<T>::reduce_add(int t, int k, long* ds, T2* values, int nnz)
{
   long offset = t*offt + k*offk;
   for (int di=0; di<nnz; di++)
      data[offset+ds[di]] += values[di];
}
template <typename T>
inline void reduction_array2<T>::reduce_add_outerprod(int t, int k, arr(T) value)
{
   long offset = t*offt + k*offk;
   long sqrtD = sqrt(D);
   for (int d=0; d<D; d++)
      data[offset+d] += value[d/sqrtD]*value[d%sqrtD];
}
template <typename T>
inline void reduction_array2<T>::reduce_multiply(int t, int k, arr(T) value)
{
   long offset = t*offt + k*offk;
   for (int d=0; d<D; d++)
      data[offset+d] *= value[d];
}
template <typename T>
inline void reduction_array2<T>::reduce_min(int t, int k, arr(T) value)
{
   long offset = t*offt + k*offk;
   for (int d=0; d<D; d++)
      data[offset+d] = mymin(data[offset+d], value[d]);
}
template <typename T>
inline void reduction_array2<T>::reduce_max(int t, int k, arr(T) value)
{
   long offset = t*offt + k*offk;
   for (int d=0; d<D; d++)
      data[offset+d] = mymax(data[offset+d], value[d]);
}

// --------------------------------------------------------------------------
// -- reduce_function
// --   Performs the reduction specified by the function pointer
// --------------------------------------------------------------------------
template <typename T>
inline void reduction_array2<T>::reduce_function(int t, int k, arr(T) value, T (*func)(T,T))
{
   long offset = t*offt + k*offk;
   for (int d=0; d<D; d++)
      data[offset+d] = func(data[offset+d], value[d]);
}



// --------------------------------------------------------------------------
// -- final_reduce_XXX
// --   Performs the reduction "XXX" on the threads and returns result
// --------------------------------------------------------------------------
template <typename T>
inline arr(T) reduction_array2<T>::final_reduce_add()
{
   for (int k=0; k<K; k++)
      for (int t=1; t<numThreads; t++)
      {
         long offset0 = k*offk;
         long offset1 = t*offt + k*offk;
         #pragma omp parallel for
         for (int d=0; d<D; d++)
            data[offset0+d] += data[offset1+d];
      }
   collapse_cache_line();
   return data;
}
template <typename T>
inline arr(T) reduction_array2<T>::final_reduce_multiply()
{
   for (int t=1; t<numThreads; t++)
      for (int k=0; k<K; k++)
      {
         long offset0 = k*offk;
         long offset1 = t*offt + k*offk;
         for (int d=0; d<D; d++)
            data[offset0+d] *= data[offset1+d];
      }
   collapse_cache_line();
   return data;
}
template <typename T>
inline arr(T) reduction_array2<T>::final_reduce_min()
{
   for (int t=1; t<numThreads; t++)
      for (int k=0; k<K; k++)
      {
         long offset0 = k*offk;
         long offset1 = t*offt + k*offk;
         for (int d=0; d<D; d++)
            data[offset0+d] = mymin(data[offset0+d],data[offset1+d]);
      }
   collapse_cache_line();
   return data;
}
template <typename T>
inline arr(T) reduction_array2<T>::final_reduce_max()
{
   for (int t=1; t<numThreads; t++)
      for (int k=0; k<K; k++)
      {
         long offset0 = k*offk;
         long offset1 = t*offt + k*offk;
         for (int d=0; d<D; d++)
            data[offset0+d] = mymax(data[offset0+d],data[offset1+d]);
      }
   collapse_cache_line();
   return data;
}

// --------------------------------------------------------------------------
// -- final_reduce_function
// --   The function pointer version of above
// --------------------------------------------------------------------------
template <typename T>
inline arr(T) reduction_array2<T>::final_reduce_function(T (*func)(T,T))
{
   for (int t=1; t<numThreads; t++)
      for (int k=0; k<K; k++)
      {
         long offset0 = k*offk;
         long offset1 = t*offt + k*offk;
         for (int d=0; d<D; d++)
            data[offset0+d] = func(data[offset0+d],data[offset1+d]);
      }
   collapse_cache_line();
   return data;
}

// --------------------------------------------------------------------------
// -- collapse_cache_line
// --   collapses the cache line for the final return
// --------------------------------------------------------------------------
template <typename T>
inline void reduction_array2<T>::collapse_cache_line()
{
   // doesn't need to do anything since cache_line = 1;
   for (int k=1; k<K; k++)
   {
      long offsetOld = k*offk;
      long offsetNew = k*D;
      for (int d=0; d<D; d++)
         data[offsetNew+d] = data[offsetOld+d];
   }
}

// --------------------------------------------------------------------------
// -- final_reduce_ext_XXX
// --   Performs the reduction "XXX" on the threads into the external array.
// -- Assumes that ext is already allocated to the correct size.
// --------------------------------------------------------------------------
template <typename T>
inline void reduction_array2<T>::final_reduce_ext_add(arr(T) ext)
{
   set_memory<T>(ext, 0, sizeof(T)*K*D);
   for (int t=0; t<numThreads; t++)
      for (int k=0; k<K; k++)
      {
         long offset0 = k*D;
         long offset1 = t*offt + k*offk;
         for (int d=0; d<D; d++)
            ext[offset0+d] += data[offset1+d];
      }
}
template <typename T>
inline void reduction_array2<T>::final_reduce_ext_multiply(arr(T) ext)
{
   set_memory<T>(ext, 1, sizeof(T)*K*D);
   for (int t=0; t<numThreads; t++)
      for (int k=0; k<K; k++)
      {
         long offset0 = k*D;
         long offset1 = t*offt + k*offk;
         for (int d=0; d<D; d++)
            ext[offset0+d] *= data[offset1+d];
      }
}

#endif

// =============================================================================
// == reduction_array.h
// == --------------------------------------------------------------------------
// == A reduction array class that can be used with OpenMP. Current OpenMP
// == software only allows the reduction of an array into a scalar. This class
// == allows one to reduce an array into an array. Also, since the code here
// == implements a reduction similar to OpenMP without actually using OpenMP's
// == reduction clause, the reduction_array class allows users to define their
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
// ==     >> reduction_array<double> my_arr(numThreads,D,initial_value);
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

#ifndef _REDUCTION_ARRAY
#define _REDUCTION_ARRAY

#include <string.h>
#include <cstdlib>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
//#include "array.h"
//#include "helperMEX.h"

// this should be the cache line size.  set it to 4, might need to be higher to
// avoid false sharing
#ifndef cache_line
#define cache_line 4
#endif

template <typename T>
inline T mymax(T val1, T val2)
{
   return (val1>val2) ? val1 : val2;
}
template <typename T>
inline T mymin(T val1, T val2)
{
   return (val1<val2) ? val1 : val2;
}

template <typename T>
class reduction_array
{
private:
   arr(T) data;
   int numThreads;
   int D;
   int offt; // offsets for the threads
   int offd; // offsets for the dimensions

public:
   // --------------------------------------------------------------------------
   // -- reduction_array
   // --   constructors; initializes the reduction array with a number of
   // -- threads, each with a D dimensional vector. The third parameter can be
   // -- specified to give the initializing value.
   // --------------------------------------------------------------------------
   reduction_array();
   reduction_array(int thenumThreads, int theD);
   reduction_array(int thenumThreads, int theD, T value);
   virtual ~reduction_array();

   // --------------------------------------------------------------------------
   // -- init_values
   // --   Initializes all values in the reduciton array to "value"
   // --
   // --   parameters:
   // --     - value : the value to set all elements to
   // --------------------------------------------------------------------------
   void init_values(T value);

   // --------------------------------------------------------------------------
   // -- getThreadArray
   // --   Returns the t^th D-dimensional array that is accessed by the t^th
   // -- thread.  Can be used for more complicated reductions
   // --------------------------------------------------------------------------
   arr(T) getThreadArray(const int t);

   // --------------------------------------------------------------------------
   // -- reduce_XXX
   // --   Performs the reduction "XXX" on the d^th dimension with value
   // --------------------------------------------------------------------------
   void reduce_inc(int t, int d);
   void reduce_add(int t, int d, T value);
   void reduce_multiply(int t, int d, T value);
   void reduce_min(int t, int d, T value);
   void reduce_max(int t, int d, T value);

   // --------------------------------------------------------------------------
   // -- reduce_function
   // --   Performs the reduction specified by the function pointer
   // --------------------------------------------------------------------------
   void reduce_function(int t, int d, T value, T (*func)(T,T));

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
// -- reduction_array
// --   constructors; initializes the reduction array with a number of
// -- threads, each with a D dimensional vector. The third parameter can be
// -- specified to give the initializing value.
// --------------------------------------------------------------------------
template <typename T>
reduction_array<T>::reduction_array() :
   numThreads(0), D(0), data(NULL)
{
}
template <typename T>
reduction_array<T>::reduction_array(int thenumThreads, int theD) :
   numThreads(thenumThreads), D(theD)
{
   offt = D*cache_line;
   offd = cache_line;
   data = allocate_memory<T>(numThreads*D*cache_line);
}
template <typename T>
reduction_array<T>::reduction_array(int thenumThreads, int theD, T value) :
   numThreads(thenumThreads), D(theD)
{
   offt = D*cache_line;
   offd = cache_line;
   data = allocate_memory<T>(numThreads*D*cache_line);
   for (int i=0; i<offt*numThreads; i++)
      data[i] = value;
   //memset(data, value, sizeof(T)*numThreads*D*cache_line);
}
template <typename T>
reduction_array<T>::~reduction_array()
{
   if (data!=NULL) deallocate_memory<T>(data);
}


// --------------------------------------------------------------------------
// -- init_values
// --   Initializes all values in the reduciton array to "value"
// --
// --   parameters:
// --     - value : the value to set all elements to
// --------------------------------------------------------------------------
template <typename T>
void reduction_array<T>::init_values(T value)
{
   set_memory<T>(data, value, sizeof(T)*numThreads*D*cache_line);
}

// --------------------------------------------------------------------------
// -- getThreadArray
// --   Returns the t^th D-dimensional array that is accessed by the t^th
// -- thread.  Can be used for more complicated reductions
// --------------------------------------------------------------------------
template <typename T>
arr(T) reduction_array<T>::getThreadArray(const int t)
{
   return data+t*offt;
}


// --------------------------------------------------------------------------
// -- reduce_XXX
// --   Performs the reduction "XXX" on the d^th dimension with value
// --------------------------------------------------------------------------
template <typename T>
inline void reduction_array<T>::reduce_inc(int t, int d)
{
   ++data[t*offt+d*offd];
}
template <typename T>
inline void reduction_array<T>::reduce_add(int t, int d, T value)
{
   data[t*offt+d*offd] += value;
}
template <typename T>
inline void reduction_array<T>::reduce_multiply(int t, int d, T value)
{
   data[t*offt+d*offd] *= value;
}
template <typename T>
inline void reduction_array<T>::reduce_min(int t, int d, T value)
{
   data[t*offt+d*offd] = mymin(data[t*offt+d*offd], value);
}
template <typename T>
inline void reduction_array<T>::reduce_max(int t, int d, T value)
{
   data[t*offt+d*offd] = mymax(data[t*offt+d*offd], value);
}

// --------------------------------------------------------------------------
// -- reduce_function
// --   Performs the reduction specified by the function pointer
// --------------------------------------------------------------------------
template <typename T>
inline void reduction_array<T>::reduce_function(int t, int d, T value, T (*func)(T,T))
{
   data[t*offt+d*offd] = func(data[t*offt+d*offd], value);
}



// --------------------------------------------------------------------------
// -- final_reduce_XXX
// --   Performs the reduction "XXX" on the threads and returns result
// --------------------------------------------------------------------------
template <typename T>
inline arr(T) reduction_array<T>::final_reduce_add()
{
   for (int t=1; t<numThreads; t++)
      for (int d=0; d<D; d++)
         data[d*offd] += data[t*offt+d*offd];
   collapse_cache_line();
   return data;
}
template <typename T>
inline arr(T) reduction_array<T>::final_reduce_multiply()
{
   for (int t=1; t<numThreads; t++)
      for (int d=0; d<D; d++)
         data[d*offd] *= data[t*offt+d*offd];
   collapse_cache_line();
   return data;
}
template <typename T>
inline arr(T) reduction_array<T>::final_reduce_min()
{
   for (int t=1; t<numThreads; t++)
      for (int d=0; d<D; d++)
         data[d*offd] = mymin(data[d*offd],data[t*offt+d*offd]);
   collapse_cache_line();
   return data;
}
template <typename T>
arr(T) reduction_array<T>::final_reduce_max()
{
   for (int t=1; t<numThreads; t++)
      for (int d=0; d<D; d++)
         data[d*offd] = mymax(data[d*offd],data[t*offt+d*offd]);
   collapse_cache_line();
   return data;
}

// --------------------------------------------------------------------------
// -- final_reduce_function
// --   The function pointer version of above
// --------------------------------------------------------------------------
template <typename T>
arr(T) reduction_array<T>::final_reduce_function(T (*func)(T,T))
{
   for (int t=1; t<numThreads; t++)
      for (int d=0; d<D; d++)
         data[d*offd] = func(data[d*offd],data[t*offt+d*offd]);
   collapse_cache_line();
   return data;
}

// --------------------------------------------------------------------------
// -- collapse_cache_line
// --   collapses the cache line for the final return
// --------------------------------------------------------------------------
template <typename T>
void reduction_array<T>::collapse_cache_line()
{
   for (int d=1; d<D; d++)
      data[d] = data[d*offd];
}





// --------------------------------------------------------------------------
// -- final_reduce_ext_XXX
// --   Performs the reduction "XXX" on the threads into the external array.
// -- Assumes that ext is already allocated to the correct size.
// --------------------------------------------------------------------------
template <typename T>
inline void reduction_array<T>::final_reduce_ext_add(arr(T) ext)
{
   set_memory<T>(ext, 0, sizeof(T)*D);
   for (int t=0; t<numThreads; t++)
      for (int d=0; d<D; d++)
         ext[d] += data[t*offt+d*offd];
}
template <typename T>
inline void reduction_array<T>::final_reduce_ext_multiply(arr(T) ext)
{
   set_memory<T>(ext, 1, sizeof(T)*D);
   for (int t=0; t<numThreads; t++)
      for (int d=0; d<D; d++)
         ext[d] *= data[t*offt+d*offd];
}


#endif

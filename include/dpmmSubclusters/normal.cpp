// =============================================================================
// == normal.cpp
// == --------------------------------------------------------------------------
// == A class for a Normal distribution
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

#include "normal.h"

#ifndef pi
#define pi 3.14159265
#endif

#ifndef logpi
#define logpi 1.144729885849
#endif

// --------------------------------------------------------------------------
// -- normal
// --   constructor; initializes to empty
// --------------------------------------------------------------------------
normal::normal() : D(0), D2(0), mean(NULL), cov(NULL), prec(NULL), logDetCov(0)
{
}

// --------------------------------------------------------------------------
// -- normal
// --   copy constructor;
// --------------------------------------------------------------------------
normal::normal(const normal& that)
{
   copy(that);
}
// --------------------------------------------------------------------------
// -- operator=
// --   assignment operator
// --------------------------------------------------------------------------
normal& normal::operator=(const normal& that)
{
   if (this != &that)
   {
      if (D==that.D)
      {
         logDetCov = that.logDetCov;
         memcpy(mean, that.mean, sizeof(double)*D);
         memcpy(cov, that.cov, sizeof(double)*D2);
         memcpy(prec, that.prec, sizeof(double)*D2);
      }
      else
      {
         cleanup();
         copy(that);
      }
   }
   return *this;
}
// --------------------------------------------------------------------------
// -- copy
// --   returns a copy of this
// --------------------------------------------------------------------------
void normal::copy(const normal& that)
{
   D = that.D;
   D2 = that.D2;
   logDetCov = that.logDetCov;
   mean = allocate_memory<double>(D);
   cov = allocate_memory<double>(D2);
   prec = allocate_memory<double>(D2);   
   memcpy(mean, that.mean, sizeof(double)*D);
   memcpy(cov, that.cov, sizeof(double)*D2);
   memcpy(prec, that.prec, sizeof(double)*D2);
}

// --------------------------------------------------------------------------
// -- normal
// --   constructor; intializes to all the values given
// --------------------------------------------------------------------------
normal::normal(int _D)
{
   D = _D;
   D2 = D*D;
   mean = allocate_memory<double>(D);
   cov = allocate_memory<double>(D2);
   prec = allocate_memory<double>(D2);
}

// --------------------------------------------------------------------------
// -- ~normal
// --   destructor
// --------------------------------------------------------------------------
normal::~normal()
{
   cleanup();
}
// --------------------------------------------------------------------------
// -- ~cleanup
// --   deletes all the memory allocated by this
// --------------------------------------------------------------------------
void normal::cleanup()
{
   if (mean)   deallocate_memory(mean);   mean = NULL;
   if (cov)    deallocate_memory(cov);    cov = NULL;
   if (prec)   deallocate_memory(prec);   prec = NULL;
}


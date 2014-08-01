// =============================================================================
// == normal.h
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

#ifndef _NORMAL_H_INCLUDED_
#define _NORMAL_H_INCLUDED_

//#include "matrix.h"
//#include "mex.h"
#include <math.h>
//#include "array.h"

//#include "helperMEX.h"
//#include "debugMEX.h"

#include "dpmmSubclusters/linear_algebra.h"
#include "dpmmSubclusters/myfuncs.h"

#ifndef log2pi
#define log2pi 1.837877066409
#endif

class normal
{
public:
   // instantiated gaussian parameters
   int D;
   int D2;
   arr(double) mean;
   arr(double) cov;
   arr(double) prec;
   double logDetCov;

public:
   // --------------------------------------------------------------------------
   // -- normal
   // --   constructor; initializes to empty
   // --------------------------------------------------------------------------
   normal();
   // --------------------------------------------------------------------------
   // -- normal
   // --   copy constructor;
   // --------------------------------------------------------------------------
   normal(const normal& that);
   // --------------------------------------------------------------------------
   // -- operator=
   // --   assignment operator
   // --------------------------------------------------------------------------
   normal& operator=(const normal& that);
   // --------------------------------------------------------------------------
   // -- copy
   // --   returns a copy of this
   // --------------------------------------------------------------------------
   void copy(const normal& that);
   // --------------------------------------------------------------------------
   // -- normal
   // --   constructor; intializes to all the values given
   // --------------------------------------------------------------------------
   normal(int _D);

   // --------------------------------------------------------------------------
   // -- ~normal
   // --   destructor
   // --------------------------------------------------------------------------
   virtual ~normal();

   // --------------------------------------------------------------------------
   // -- ~cleanup
   // --   deletes all the memory allocated by this
   // --------------------------------------------------------------------------
   virtual void cleanup();

   double predictive_loglikelihood(arr(double) data) const;
   double data_loglikelihood(int N, arr(double) t, arr(double) T, arr(double) tempVec) const;
   double Jdivergence(const normal &other);

   friend class niw_sampled;
   friend class niwSphere_sampled;
   friend class cluster_sampled;
};


inline double normal::predictive_loglikelihood(arr(double) data) const
{
   return -0.5*xmut_A_xmu(data, mean, prec, D) - 0.5*logDetCov - 0.5*D*log2pi;
}


inline double normal::data_loglikelihood(int N, arr(double) t, arr(double) T, arr(double) tempVec) const
{
   if (N==0)
      return 0;
   for (int d=0; d<D; d++)
      tempVec[d] = t[d]/N - mean[d];
   return -0.5*N*D*log2pi - 0.5*N*logDetCov - 0.5*N*(xt_A_x(tempVec,prec,D)) - 0.5*traceAxB(prec,T,D) + 0.5/N*xt_A_x(t,prec,D);
}

inline double normal::Jdivergence(const normal &other)
{
   // ignore the det(cov) because it will be cancelled out
   return 0.5*(traceAxB(prec,other.cov,D) + xmut_A_xmu(mean, other.mean, prec, D) - D)
         +0.5*(traceAxB(other.prec,cov,D) + xmut_A_xmu(other.mean, mean, other.prec, D) - D);
}

#endif

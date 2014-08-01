#ifndef _SAMPLE_CATEGORICAL_H_INCLUDED_
#define _SAMPLE_CATEGORICAL_H_INCLUDED_

#include <algorithm>

#include "tableFuncs.h"

inline double total_logcategorical(arr(double) probabilities, int K, double maxProb)
{
   double totalProb = 0;
   for (int k=0; k<K; k++)
      totalProb += myexpneg(probabilities[k]-maxProb);
   return totalProb;
}
inline int sample_logcategorical(arr(double) probabilities, int K, double maxProb, double totalProb, gsl_rng *r)
{
   double rand_num = my_rand(r) * totalProb;
   double total = 0;
   int k;
   for (k=0; k<K; k++)
   {
      total += myexpneg(probabilities[k]-maxProb);
      if (rand_num<=total)
         break;
   }
   return k;
}

inline double convert_logcategorical(arr(double) probabilities, int K, double maxProb)
{
   double totalProb = 0;
   for (int k=0; k<K; k++)
   {
      probabilities[k] = myexpneg(probabilities[k]-maxProb);
      totalProb += probabilities[k];
   }
   return totalProb;
}

inline double convert_logcategorical(arr(double) probabilities, int K)
{
   double maxProb = probabilities[0];
   for (int k=1; k<K; k++)
      maxProb = std::max(maxProb, probabilities[k]);
   return convert_logcategorical(probabilities, K, maxProb);
}

inline int sample_categorical(arr(double) probabilities, int K, double totalProb, gsl_rng *r)
{
   double rand_num = my_rand(r) * totalProb;
   double total = 0;
   int k;
   for (k=0; k<K; k++)
   {
      total += probabilities[k];
      if (rand_num<=total)
         break;
   }
   return k;
}

inline int max_categorical(arr(double) probabilities, int K)
{
   double maxVal = 0;
   int k;
   for (int a=0; a<K; a++)
   {
      if (probabilities[a]>maxVal)
      {
         maxVal = probabilities[a];
         k = a;
      }
   }
   return k;
}

inline int max_categorical(arr(double) probabilities, int K, double totalProb, gsl_rng *r)
{
   return max_categorical(probabilities, K);
}
#endif

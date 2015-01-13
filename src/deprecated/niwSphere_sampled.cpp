/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */
// =============================================================================
// == niwSphere_sampled.cpp
// == --------------------------------------------------------------------------
// == A class for a Normal Inverse-Wishart distribution
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

#include "niwSphere_sampled.h"

// --------------------------------------------------------------------------
// -- niwSphere_sampled
// --   constructor; initializes to empty
// --------------------------------------------------------------------------
niwSphere_sampled::niwSphere_sampled() :
   D(0), D2(0), kappah(0), nuh(0), thetah(NULL), Deltah(NULL), t(NULL), T(NULL), N(0),
   kappa(0), nu(0), theta(NULL), Delta(NULL), tempVec(NULL), tempMtx(NULL), param(0)
{
   r = initialize_gsl_rand(mx_rand());
}

// --------------------------------------------------------------------------
// -- niwSphere_sampled
// --   copy constructor;
// --------------------------------------------------------------------------
niwSphere_sampled::niwSphere_sampled(const niwSphere_sampled& that)
{
   copy(that);
}
// --------------------------------------------------------------------------
// -- operator=
// --   assignment operator
// --------------------------------------------------------------------------
niwSphere_sampled& niwSphere_sampled::operator=(const niwSphere_sampled& that)
{
   if (this != &that)
   {
      cleanup();
      copy(that);
   }
}
// --------------------------------------------------------------------------
// -- copy
// --   returns a copy of this
// --------------------------------------------------------------------------
void niwSphere_sampled::copy(const niwSphere_sampled& that)
{
   r = initialize_gsl_rand(mx_rand());
   D = that.D;
   D2 = that.D2;
   kappah = that.kappah;
   nuh = that.nuh;
   N = that.N;
   kappa = that.kappa;
   nu = that.nu;
   thetah = allocate_memory<double>(D);
   Deltah = allocate_memory<double>(D2);
   t = allocate_memory<double>(D);
   T = allocate_memory<double>(D2);
   theta = allocate_memory<double>(D);
   Delta = allocate_memory<double>(D2);
   tempVec = allocate_memory<double>(D);
   tempMtx = allocate_memory<double>(D2);
   param = that.param;

   memcpy(thetah, that.thetah, sizeof(double)*D);
   memcpy(t, that.t, sizeof(double)*D);
   memcpy(theta, that.theta, sizeof(double)*D);
   memcpy(Deltah, that.Deltah, sizeof(double)*D2);
   memcpy(T, that.T, sizeof(double)*D2);
   memcpy(Delta, that.Delta, sizeof(double)*D2);
}

// --------------------------------------------------------------------------
// -- niwSphere_sampled
// --   constructor; intializes to all the values given
// --------------------------------------------------------------------------
niwSphere_sampled::niwSphere_sampled(int _D, double _kappah, double _nuh, arr(double) _thetah, arr(double) _Deltah) :
   D(_D), kappah(_kappah), nuh(_nuh), param(_D)
{
   D2 = D*D;
   thetah = allocate_memory<double>(D);
   Deltah = allocate_memory<double>(D2);
   t = allocate_memory<double>(D);
   T = allocate_memory<double>(D2);
   theta = allocate_memory<double>(D);
   Delta = allocate_memory<double>(D2);
   tempVec = allocate_memory<double>(D);
   tempMtx = allocate_memory<double>(D2);

   memcpy(thetah, _thetah, sizeof(double)*D);
   memcpy(Deltah, _Deltah, sizeof(double)*D2);

   r = initialize_gsl_rand(mx_rand());

   clear();

}




// --------------------------------------------------------------------------
// -- ~niwSphere_sampled
// --   destructor
// --------------------------------------------------------------------------
niwSphere_sampled::~niwSphere_sampled()
{
   cleanup();
}
// --------------------------------------------------------------------------
// -- ~cleanup
// --   deletes all the memory allocated by this
// --------------------------------------------------------------------------
void niwSphere_sampled::cleanup()
{
   if (thetah) deallocate_memory(thetah); thetah = NULL;
   if (Deltah) deallocate_memory(Deltah); Deltah = NULL;
   if (t)      deallocate_memory(t);      t = NULL;
   if (T)      deallocate_memory(T);      T = NULL;
   if (theta)  deallocate_memory(theta);  theta = NULL;
   if (Delta)  deallocate_memory(Delta);  Delta = NULL;
   if (tempVec)deallocate_memory(tempVec);tempVec = NULL;
   if (tempMtx)deallocate_memory(tempMtx);tempMtx = NULL;
   if (r)      gsl_rng_free(r);           r = NULL;
}

// --------------------------------------------------------------------------
// -- clear
// --   Empties out the statistics and posterior hyperparameters and the
// -- sampled parameters. This basically deletes all notions of data from
// -- the node.
// --------------------------------------------------------------------------
void niwSphere_sampled::clear()
{
   empty();
   update_posteriors_sample();
}
// --------------------------------------------------------------------------
// -- empty
// --   Empties out the statistics and posterior hyperparameters, but not
// -- the sampled parameters
// --------------------------------------------------------------------------
void niwSphere_sampled::empty()
{
   N = 0;
   memset(t, 0, sizeof(double)*D);
   memset(T, 0, sizeof(double)*D2);
   memset(theta, 0, sizeof(double)*D);
   memset(Delta, 0, sizeof(double)*D2);
   kappa = kappah;
   nu = nuh;
}

bool niwSphere_sampled::isempty() const               { return (N==0); }
int niwSphere_sampled::getN() const                   { return N;}
int niwSphere_sampled::getD() const                   { return D;}
arr(double) niwSphere_sampled::get_mean() const        { return param.mean;}
arr(double) niwSphere_sampled::get_cov() const         { return param.cov;}
arr(double) niwSphere_sampled::get_prec() const        { return param.prec;}

gsl_rng* niwSphere_sampled::get_r()                   { return r;}

void niwSphere_sampled::set_normal(normal &other)
{
   param = other;
}
void niwSphere_sampled::set_normal(arr(double) _mean, arr(double) _cov)
{
   memcpy(param.mean, _mean, sizeof(double)*D);
   memcpy(param.cov, _cov, sizeof(double)*D*D);
   // do the cholesky decomposition
   Eigen::Map<Eigen::MatrixXd> tempCov(param.cov, D, D);
   Eigen::Map<Eigen::MatrixXd> tempPrec(param.prec, D, D);

   param.logDetCov = log(det(_cov,D));
   // convert to a precision
   tempPrec = tempCov.inverse();
}

normal* niwSphere_sampled::get_normal()
{
   return &param;
}

// --------------------------------------------------------------------------
// -- update_posteriors
// --   Updates the posterior hyperparameters
// --------------------------------------------------------------------------
void niwSphere_sampled::update_posteriors()
{
  //TODO
  //assert(false);
  //kappa = kappah + N;
   nu = nuh + N;
//   for (int d=0; d<D; d++)
//      theta[d] = (thetah[d]*kappah + t[d]) / kappa;
   for (int d=0; d<D2; d++)
      Delta[d] = (Deltah[d]*nuh + T[d]) / nu;
}

void niwSphere_sampled::update_posteriors_sample()
{
  update_posteriors();
  sample();
}

// --------------------------------------------------------------------------
// -- add_data
// --   functions to add an observation to the niwSphere_sampled. Updates the sufficient
// -- statistics, posterior hyperparameters, and predictive parameters
// --
// --   parameters:
// --     - data : the new observed data point of size [1 D]
// --------------------------------------------------------------------------
void niwSphere_sampled::add_data_init(arr(double) data)
{
  //TODO
  assert(false);
   // update the sufficient stats and the N
   N++;
   for (int d1=0; d1<D; d1++)
   {
      double temp_data = data[d1];
      t[d1] += temp_data;
      for (int d2=0; d2<D; d2++)
         T[d1+d2*D] += temp_data*data[d2];
   }
}
void niwSphere_sampled::add_data(arr(double) data)
{
  assert(false);
   add_data_init(data);
   update_posteriors();
}
void niwSphere_sampled::merge_with(niwSphere_sampled* &other, bool doSample)
{
  assert(false);
   if (other!=NULL)
      merge_with(*other, doSample);
}
void niwSphere_sampled::merge_with(niwSphere_sampled* &other1, niwSphere_sampled* &other2, bool doSample)
{
  assert(false);
   if (other1==NULL && other2!=NULL)
      merge_with(*other2,doSample);
   else if (other1!=NULL && other2==NULL)
      merge_with(*other1,doSample);
   else if (other1!=NULL && other2!=NULL)
      merge_with(*other1, *other2, doSample);
}
void niwSphere_sampled::merge_with(niwSphere_sampled &other, bool doSample)
{
  //TODO
  assert(false);
   N += other.N;
   for (int d=0; d<D; d++)
      t[d] += other.t[d];
   for (int d=0; d<D2; d++)
      T[d] += other.T[d];
   if (doSample)
      update_posteriors_sample();
   else
      update_posteriors();
}
void niwSphere_sampled::merge_with(niwSphere_sampled &other1, niwSphere_sampled &other2, bool doSample)
{
  //TODO: how to merge two clusters on the sphere when they have different means??
  //assert(false);
  //   for (int d=0; d<D; d++)
  //      t[d] += other1.t[d] + other2.t[d];

  // find karcher mean
  Sphere<double> M;
  Eigen::MatrixXd mus(mu_.rows(),3); mus << mu_,other1.mu_,other2.mu_;
  Eigen::VectorXd w(3); w << N,other1.N,other2.N;
  mu_ = karcherMeanWeighted<Eigen::Dynamic>(mu_,mus,w,100);
  Eigen::MatrixXd mus_p = M.Log_p_north(mu_,mus);

  // outer products
  for (int d=0; d<D2; d++)
      T[d] += other1.T[d] + other2.T[d];
  Eigen::Map<Eigen::MatrixXd> _T(T,D,D);
  _T += w(0)* mus_p.col(0)*mus_p.col(0).transpose();
  _T += w(1)* mus_p.col(1)*mus_p.col(1).transpose();
  _T += w(2)* mus_p.col(2)*mus_p.col(2).transpose();

  // counts
  N += other1.N + other2.N;

  cout<<"N = "<<N<<endl;
  cout<<"T = "<<_T<<endl;
  cout<<"mu = "<<mu_<<endl;

  if (doSample)
    update_posteriors_sample();
  else
    update_posteriors();
}
void niwSphere_sampled::set_stats(int _N, arr(double) _t, arr(double) _T)
{
   //assert(false);
   N = _N;
   //memcpy(t, _t, sizeof(double)*D);
   memcpy(T, _T, sizeof(double)*D2);
}

double niwSphere_sampled::Jdivergence(const niwSphere_sampled &other)
{
  //TODO
  assert(false);
   // ignore the det(cov) because it will be cancelled out
   return param.Jdivergence(other.param);
}


double niwSphere_sampled::data_loglikelihood() const
{
  //TODO
  assert(false);
   return param.data_loglikelihood(N, t, T, tempVec);
}

double niwSphere_sampled::data_loglikelihood_marginalized() const
{
  //TODO
  assert(false);
   /*double val = -0.5*N*D*logpi + mylogmgamma05(nu,D) - mylogmgamma05(nuh,D) + (0.5*nuh)*(D*log(nuh)+log(det(Deltah,D))) - (0.5*nu)*(D*log(nu)+log(det(Delta,D))) + 0.5*D*log(kappah/kappa);
   for (int d1=0; d1<D; d1++)
   {
      for (int d2=0; d2<D; d2++)
         mexPrintf("%e\t", Delta[d1+d2*D]);
      mexPrintf("\n");
   }
   mexPrintf("%e %e %e %e %e %e\n", -0.5*N*D*logpi, mylogmgamma05(nu,D), -mylogmgamma05(nuh,D), (0.5*nuh)*(D*log(nuh)+log(det(Deltah,D))), - (0.5*nu)*(D*log(nu)+log(det(Delta,D))), 0.5*D*log(kappah/kappa));*/

   if (N==0)
      return 0;

   //mexPrintf("%e %e %e %e %e %e\n", -0.5*N*D*logpi, mylogmgamma05(nu,D), -mylogmgamma05(nuh,D), (0.5*nuh)*(D*log(nuh)+log(det(Deltah,D))), log(det(Delta,D)), 0.5*D*log(kappah/kappa));
   return -0.5*N*D*logpi + mylogmgamma05(nu,D) - mylogmgamma05(nuh,D) + (0.5*nuh)*(D*log(nuh)+log(det(Deltah,D))) - (0.5*nu)*(D*log(nu)+log(det(Delta,D))) + 0.5*D*log(kappah/kappa);

/*   // covariance gets divided by N
   // mean is average of data points
   for (int d=0; d<D; d++)
      tempVec[d] = t[d]/N - mean[d];
   for (int d=0; d<D2; d++)
      tempMtx[d] = prec[d] * N;

   return -0.5*xt_A_x(tempVec, tempMtx, D);*/
}


double niwSphere_sampled::data_loglikelihood_marginalized_testmerge(niwSphere_sampled *other) const
{
  //TODO
  assert(false);
   int tN = N + other->N;
   double tkappa = kappa + other->N;
   double tnu = nu + other->N;
   vector<double> ttheta(D);
   vector<double> tDelta(D2);
   for (int d=0; d<D; d++)
      ttheta[d] = (thetah[d]*kappah + t[d] + other->t[d]) / tkappa;
   for (int d=0; d<D2; d++)
      tDelta[d] = (Deltah[d]*nuh + T[d] + other->T[d] + kappah*thetah[d%D]*thetah[d/D] - tkappa*ttheta[d%D]*ttheta[d/D]) / tnu;

   return -0.5*tN*D*logpi + mylogmgamma05(tnu,D) - mylogmgamma05(nuh,D) + (0.5*nuh)*(D*log(nuh)+log(det(Deltah,D))) - (0.5*tnu)*(D*log(tnu)+log(det(tDelta.data(),D))) + 0.5*D*log(kappah/tkappa);
}


void niwSphere_sampled::sample(normal &_param)
{
  //assert(false);
   // do the cholesky decomposition
   Eigen::Map<Eigen::MatrixXd> sigma(Delta, D, D);
   Eigen::LLT<Eigen::MatrixXd> myLlt(sigma);
   Eigen::MatrixXd chol = myLlt.matrixL();

   for (int d=0; d<D; d++)
   {
      _param.cov[d+d*D] = sqrt(gsl_ran_chisq(r, nu-d));
      for (int d2=d+1; d2<D; d2++)
      {
         _param.cov[d2+d*D] = 0;
         _param.cov[d+d2*D] = gsl_ran_gaussian(r, 1);
      }
   }

   // diag * cov^-1
   Eigen::Map<Eigen::MatrixXd> tempCov(_param.cov, D, D);
   Eigen::Map<Eigen::MatrixXd> tempPrec(_param.prec, D, D);
   tempCov = sqrt(nu)*chol*tempCov.inverse();
   tempCov = tempCov*tempCov.transpose();

   // temp now contains the covariance
   myLlt.compute(tempCov);
//   chol = myLlt.matrixL();

//   // populate the mean
//   for (int d=0; d<D; d++)
//      _param.mean[d] = gsl_ran_gaussian(r,1);
//   Eigen::Map<Eigen::VectorXd> emean(_param.mean, D);
//   emean = chol*emean / sqrt(kappa);
//   for (int d=0; d<D; d++)
//      _param.mean[d] += theta[d];

   _param.logDetCov = log(det(_param.cov,D));
   // convert to a precision
   tempPrec = tempCov.inverse();
}


void niwSphere_sampled::sample_scale(normal &_param)
{
  //TODO
  assert(false);
   // do the cholesky decomposition
   Eigen::Map<Eigen::MatrixXd> sigma(Delta, D, D);
   Eigen::LLT<Eigen::MatrixXd> myLlt(sigma);
   Eigen::MatrixXd chol = myLlt.matrixL();

   double scale = 0.001;
   double nnu = nu-scale*N;
   double nkappa = kappa-scale*N;

   for (int d=0; d<D; d++)
   {
      _param.cov[d+d*D] = sqrt(gsl_ran_chisq(r, nnu-d));
      for (int d2=d+1; d2<D; d2++)
      {
         _param.cov[d2+d*D] = 0;
         _param.cov[d+d2*D] = gsl_ran_gaussian(r, 1);
      }
   }

   // diag * cov^-1
   Eigen::Map<Eigen::MatrixXd> tempCov(_param.cov, D, D);
   Eigen::Map<Eigen::MatrixXd> tempPrec(_param.prec, D, D);
   tempCov = sqrt(nnu)*chol*tempCov.inverse();
   tempCov = tempCov*tempCov.transpose();

   // temp now contains the covariance
   myLlt.compute(tempCov);
   chol = myLlt.matrixL();

   // populate the mean
   for (int d=0; d<D; d++)
      _param.mean[d] = gsl_ran_gaussian(r,1);
   Eigen::Map<Eigen::VectorXd> emean(_param.mean, D);
   emean = chol*emean / sqrt(nkappa);
   for (int d=0; d<D; d++)
      _param.mean[d] += theta[d];

   _param.logDetCov = log(det(_param.cov,D));
   // convert to a precision
   tempPrec = tempCov.inverse();
}

void niwSphere_sampled::sample()
{
  //TODO
  //assert(false);
   sample(param);

//   logpmu_prior = 0.5*nuh*D*log(nuh) + 0.5*nuh*log(det(Deltah,D)) - 0.5*nuh*D*log(2) - mylogmgamma05(nuh,D);
//   logpmu_prior += -0.5*(nuh+D+1)*param.logDetCov - 0.5*nuh*traceAxB(Deltah, param.prec, D);
//
//   // log |cov/kappa| = log kappa^D |cov| = D*log(kappa) + logdetcov
//   logpmu_prior += -0.5*D*log(2*pi) - 0.5*param.logDetCov + 0.5*D*log(kappah) - 0.5*kappah*xmut_A_xmu(param.mean, thetah, param.prec, D);
//
//   logpmu_posterior = 0.5*nu*D*log(nu) + 0.5*nu*log(det(Delta,D)) - 0.5*nu*D*log(2) - mylogmgamma05(nu,D);
//   logpmu_posterior += -0.5*(nu+D+1)*param.logDetCov - 0.5*nu*traceAxB(Delta, param.prec, D);
//
//   // log |cov/kappa| = log kappa^D |cov| = D*log(kappa) + logdetcov
//   logpmu_posterior += -0.5*D*log(2*pi) - 0.5*param.logDetCov + 0.5*D*log(kappa) - 0.5*kappa*xmut_A_xmu(param.mean, theta, param.prec, D);
}



double niwSphere_sampled::logmu_posterior(const normal &_param) const
{
  //TODO
  assert(false);
   double logmu_posterior = 0.5*nu*D*log(nu) + 0.5*nu*log(det(Delta,D)) - 0.5*nu*D*log(2) - mylogmgamma05(nu,D);
   logmu_posterior += -0.5*(nu+D+1)*_param.logDetCov - 0.5*nu*traceAxB(Delta, _param.prec, D);

   // log |cov/kappa| = log kappa^D |cov| = D*log(kappa) + logdetcov
   logmu_posterior += -0.5*D*log2pi - 0.5*_param.logDetCov + 0.5*D*log(kappa) - 0.5*kappa*xmut_A_xmu(_param.mean, theta, _param.prec, D);
   return logmu_posterior;
}

double niwSphere_sampled::logmu_posterior() const
{

  assert(false);
   return logmu_posterior(param);
}
double niwSphere_sampled::logmu_prior(const normal &_param) const
{

  assert(false);
   double logmu_prior = 0.5*nuh*D*log(nuh) + 0.5*nuh*log(det(Deltah,D)) - 0.5*nuh*D*log(2) - mylogmgamma05(nuh,D);
   logmu_prior += -0.5*(nuh+D+1)*_param.logDetCov - 0.5*nuh*traceAxB(Deltah, _param.prec, D);

   // log |cov/kappa| = log kappa^D |cov| = D*log(kappa) + logdetcov
   logmu_prior += -0.5*D*log2pi - 0.5*_param.logDetCov + 0.5*D*log(kappah) - 0.5*kappah*xmut_A_xmu(_param.mean, thetah, _param.prec, D);
   return logmu_prior;
}
double niwSphere_sampled::logmu_prior() const
{
  assert(false);
   return logmu_prior(param);
}


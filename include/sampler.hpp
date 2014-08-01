#pragma once

#include <iostream>
#include <time.h>
#include <Eigen/Dense>

#include <boost/random/mersenne_twister.hpp>
//#include <boost/random/uniform_int_distribution.hpp>
//#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_01.hpp>

#include "global.hpp"
#include "cat.hpp"


using namespace Eigen;
using std::cout;
using std::endl;

/* Sampler for CPU */
template<typename T=double>
class Sampler
{
protected:
  boost::mt19937* pRndGen_;
  bool selfManaged_;
  boost::uniform_01<> unif_;

public:
  Sampler(boost::mt19937* pRndGen=NULL);
  virtual ~Sampler();

  virtual void sampleUnif(Matrix<T,Dynamic,1>& r);
  virtual Matrix<T,Dynamic,1> sampleUnif(uint32_t N)
  {
    Matrix<T,Dynamic,1> u(N);
    sampleUnif(u);
    return u;
  };

  virtual void sampleDiscLogPdfUnNormalized(const Matrix<T,Dynamic,Dynamic>& pdfs,
    VectorXu& z);
  virtual VectorXu sampleDiscLogPdfUnNormalized(
    const Matrix<T,Dynamic,Dynamic>& pdfs)
  { 
    VectorXu z(pdfs.rows());
    sampleDiscLogPdfUnNormalized(pdfs,z);
    return z;
  };

  virtual void sampleDiscPdf(T *d_pdfs, const spVectorXu& z){ assert(false);};
  virtual void sampleDiscPdf(const Matrix<T,Dynamic,Dynamic>& pdfs, VectorXu& z);
  virtual VectorXu sampleDiscPdf(const Matrix<T,Dynamic,Dynamic>& pdfs)
  {
    VectorXu z(pdfs.rows());
    sampleDiscPdf(pdfs,z);
    return z;
  };
};

#ifdef CUDA
#include "gpuMatrix.hpp"

extern void choiceMultGpu(double* d_pdf, uint32_t* d_z, uint32_t N, 
    uint32_t M, uint32_t seed);
extern void choiceMultGpu(float* d_pdf, uint32_t* d_z, uint32_t N, 
    uint32_t M, uint32_t seed);
extern void choiceMultLogPdfUnNormalizedGpu(double* d_pdf, uint32_t* d_z, 
  uint32_t N, uint32_t M, uint32_t seed);
extern void choiceMultLogPdfUnNormalizedGpu(float* d_pdf, uint32_t* d_z, 
  uint32_t N, uint32_t M, uint32_t seed);
extern void unifGpu(double* d_u, uint32_t N, uint32_t seed);
extern void unifGpu(float* d_u, uint32_t N, uint32_t seed);

/* Sampler for GPU */
template<typename T=float>
class SamplerGpu : public Sampler<T>
{
  GpuMatrix<T> pdfs_; // one pdf per row
  GpuMatrix<uint32_t> z_; // samples from pdfs
  GpuMatrix<T> r_; // unif random numbers

public:
  SamplerGpu(uint32_t N, uint32_t K, boost::mt19937* pRndGen=NULL);
  ~SamplerGpu();

  void sampleUnif(Matrix<T,Dynamic,1>& r);
  Matrix<T,Dynamic,1> sampleUnif()
  {
    Matrix<T,Dynamic,1> u(r_.rows());
    sampleUnif(u);
    return u;
  };

  virtual void sampleDiscLogPdfUnNormalized(
      const Matrix<T,Dynamic,Dynamic>& pdfs, VectorXu& z);
  /* use pdfs already prestored in GPU memory */
  void sampleDiscPdf(T *d_pdfs, const spVectorXu& z);
//  void sampleDiscPdf(const Matrix<T,Dynamic,Dynamic>& pdfs, const spVectorXu& z);
  void sampleDiscPdf(const Matrix<T,Dynamic,Dynamic>& pdfs, VectorXu& z);
  VectorXu sampleDiscPdf(const Matrix<T,Dynamic,Dynamic>& pdfs)
  {
    VectorXu z(pdfs.rows());
    sampleDiscPdf(pdfs,z);
    return z;
  };
};
#endif


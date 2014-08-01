#pragma once

#include <iostream>
#include <stdint.h>
#include <Eigen/Dense>

#include <cuda_runtime.h>
//#include <helper_functions.h> 
#include <helper_cuda.h> 

#include "global.hpp"

using namespace Eigen;
using boost::shared_ptr;
using std::cout;
using std::endl;

template <class T>
struct GpuMatrix
{

  GpuMatrix(uint32_t rows, uint32_t cols=1);
  GpuMatrix(const Matrix<T,Dynamic,Dynamic> & data);
  GpuMatrix(const Matrix<T,Dynamic,1> & data);
  GpuMatrix(const shared_ptr<Matrix<T,Dynamic,Dynamic> >& data);
  GpuMatrix(const shared_ptr<Matrix<T,Dynamic,1> > & data);
  ~GpuMatrix();

  void set(const Matrix<T,Dynamic,Dynamic>& A);
  void set(const Matrix<T,Dynamic,1>& A);
  void set(const shared_ptr<Matrix<T,Dynamic,Dynamic> >& A);
  void set(const shared_ptr<Matrix<T,Dynamic,1> >& A);
  void setZero();

  void get(Matrix<T,Dynamic,Dynamic>& A);
  void get(Matrix<T,Dynamic,1>& A);
  void get(const shared_ptr<Matrix<T,Dynamic,Dynamic> >& A);
  void get(const shared_ptr<Matrix<T,Dynamic,1> >& A);
  Matrix<T,Dynamic,Dynamic> get(void)
  {
    Matrix<T,Dynamic,Dynamic> d(rows_,cols_);
    this->get(d); return d;
  };

  void resize(uint32_t rows, uint32_t cols);

  uint32_t rows(){return rows_;};
  uint32_t cols(){return cols_;};
  T* data(){ assert(initialized_); return data_;};
  bool isInit(){return initialized_;};

  void print(){cout<<rows_<<";"<<cols_<<" init="<<(initialized_?'y':'n')<<endl;};

private: 
  uint32_t rows_;
  uint32_t cols_;
  T * data_;
  bool initialized_;
};

// --------------------------------- impl -------------------------------------

template <class T>
GpuMatrix<T>::GpuMatrix(uint32_t rows, uint32_t cols)
  : rows_(rows), cols_(cols), initialized_(false)
{
//  cout<<rows_<<"x"<<cols_<<"="<<rows_*cols_<<endl;
  checkCudaErrors(cudaMalloc((void **)&data_, rows_*cols_*sizeof(T))); 
};

template <class T>
GpuMatrix<T>::GpuMatrix(const Matrix<T,Dynamic,Dynamic> & data)
  : rows_(data.rows()), cols_(data.cols()), initialized_(false)
{
//  cout<<rows_<<"x"<<cols_<<"="<<rows_*cols_<<endl;
  checkCudaErrors(cudaMalloc((void **)&data_, rows_*cols_*sizeof(T))); 
  set(data);
};

template <class T>
GpuMatrix<T>::GpuMatrix(const Matrix<T,Dynamic,1> & data)
  : rows_(data.rows()), cols_(1), initialized_(false)
{
//  cout<<rows_<<"x"<<cols_<<"="<<rows_*cols_<<endl;
  checkCudaErrors(cudaMalloc((void **)&data_, rows_*cols_*sizeof(T))); 
  set(data);
};

template <class T>
GpuMatrix<T>::GpuMatrix(const shared_ptr<Matrix<T,Dynamic,Dynamic> > & data)
  : rows_(data->rows()), cols_(data->cols()), initialized_(false)
{
//  cout<<rows_<<"x"<<cols_<<"="<<rows_*cols_<<endl;
  checkCudaErrors(cudaMalloc((void **)&data_, rows_*cols_*sizeof(T))); 
  set(data);
};

template <class T>
GpuMatrix<T>::GpuMatrix(const shared_ptr<Matrix<T,Dynamic,1> > & data)
  : rows_(data->rows()), cols_(1), initialized_(false)
{
//  cout<<rows_<<"x"<<cols_<<"="<<rows_*cols_<<endl;
  checkCudaErrors(cudaMalloc((void **)&data_, rows_*cols_*sizeof(T))); 
  set(data);
};

template <class T>
GpuMatrix<T>::~GpuMatrix()
{
  checkCudaErrors(cudaFree(data_));
};

template <class T>
void GpuMatrix<T>::resize(uint32_t rows, uint32_t cols)
{
  if((rows != rows_)||(cols != cols_))
  { 
    rows_ = rows;
    cols_ = cols;
    checkCudaErrors(cudaFree(data_));
    checkCudaErrors(cudaMalloc((void **)&data_, rows_*cols_*sizeof(T))); 
  } 
};

//setters 
template <class T>
void GpuMatrix<T>::set(const Matrix<T,Dynamic,Dynamic>& A)
{
  resize(A.rows(),A.cols());
  assert(A.cols() == cols_);
  assert(A.rows() == rows_);
  checkCudaErrors(cudaMemcpy(data_, A.data(), cols_*rows_* sizeof(T),
        cudaMemcpyHostToDevice));
  initialized_ = true;
};

template <class T>
void GpuMatrix<T>::set(const Matrix<T,Dynamic,1>& A)
{
  resize(A.rows(),A.cols());
  assert(A.cols() == cols_);
  assert(A.rows() == rows_);
  checkCudaErrors(cudaMemcpy(data_, A.data(), cols_*rows_* sizeof(T),
        cudaMemcpyHostToDevice));
  initialized_ = true;
};

template <class T>
void GpuMatrix<T>::set(const shared_ptr<Matrix<T,Dynamic,Dynamic> >& A)
{
  resize(A->rows(),A->cols());
  assert(A->cols() == cols_);
  assert(A->rows() == rows_);
  checkCudaErrors(cudaMemcpy(data_, A->data(), cols_*rows_* sizeof(T),
        cudaMemcpyHostToDevice));
  initialized_ = true;
};

template <class T>
void GpuMatrix<T>::set(const shared_ptr<Matrix<T,Dynamic,1> >& A)
{
  resize(A->rows(),A->cols());
  assert(A->cols() == cols_);
  assert(A->rows() == rows_);
  checkCudaErrors(cudaMemcpy(data_, A->data(), cols_*rows_* sizeof(T),
        cudaMemcpyHostToDevice));
  initialized_ = true;
};

template <class T>
void GpuMatrix<T>::setZero()
{
  Matrix<T,Dynamic,Dynamic> A = Matrix<T,Dynamic,Dynamic>::Zero(rows_,cols_);
  checkCudaErrors(cudaMemcpy(data_, A.data(), cols_*rows_* sizeof(T),
        cudaMemcpyHostToDevice));
  initialized_ = true;
};

 // getters
template <class T>
void GpuMatrix<T>::get(Matrix<T,Dynamic,Dynamic>& A)
{
  assert(A.cols() == cols_);
  assert(A.rows() == rows_);
  checkCudaErrors(cudaMemcpy(A.data(),data_, cols_*rows_*sizeof(T),
                cudaMemcpyDeviceToHost));
};

template <class T>
void GpuMatrix<T>::get(Matrix<T,Dynamic,1>& A)
{
  assert(A.cols() == cols_);
  assert(A.rows() == rows_);
  checkCudaErrors(cudaMemcpy(A.data(),data_, cols_*rows_*sizeof(T),
                cudaMemcpyDeviceToHost));
};

template <class T>
void GpuMatrix<T>::get(const shared_ptr<Matrix<T,Dynamic,Dynamic> >& A)
{
  assert(A->cols() == cols_);
  assert(A->rows() == rows_);
  checkCudaErrors(cudaMemcpy(A->data(),data_, cols_*rows_*sizeof(T),
                cudaMemcpyDeviceToHost));
};

template <class T>
void GpuMatrix<T>::get(const shared_ptr<Matrix<T,Dynamic,1> >& A)
{
  assert(A->cols() == cols_);
  assert(A->rows() == rows_);
  checkCudaErrors(cudaMemcpy(A->data(),data_, cols_*rows_*sizeof(T),
                cudaMemcpyDeviceToHost));
};


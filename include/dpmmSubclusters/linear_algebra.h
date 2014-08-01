#ifndef _LINEAR_ALGEBRA_H_INCLUDED_
#define _LINEAR_ALGEBRA_H_INCLUDED_

#include <Eigen/Dense>
#include <Eigen/LU>

#include "dpmmSubclusters/common.h"

inline double traceAxB(arr(double)A, arr(double)B, int D)
{
   double temp = 0;
   for (int d1=0; d1<D; d1++)
   {
      for (int d2=0; d2<D; d2++)
         temp += A[d1+d2*D]*B[d2+d1*D];
   }
   return temp;
}


inline void AxB(arr(double)A, arr(double)B, arr(double) output, int D)
{
   int D2 = pow(D,2);
   for (int d1=0; d1<D2; d1++)
   {
      double temp = 0;
      int di = d1%D;
      int dj = d1/D;
      for (int d2=0; d2<D; d2++)
         temp += A[di+d2*D]*B[d2+dj*D];
      output[d1] = temp;
   }
}

inline double trace(arr(double) mtx, int D)
{
   double value = 0;
   for (int d=0; d<D; d++)
      value += mtx[d*(D+1)];
   return value;
}

inline double det(arr(double) mtx, int D)
{
   double determinant;
   switch (D)
   {
      /*case 1:
         determinant = mtx[0];
         break;
      case 2:
         determinant = mtx[0]*mtx[3] - mtx[1]*mtx[2];
         break;
      case 3:
         // 0 3 6
         // 1 4 7
         // 2 5 8

         determinant = mtx[0]*(mtx[4]*mtx[8]-mtx[5]*mtx[7])
                     - mtx[3]*(mtx[1]*mtx[8]-mtx[2]*mtx[7])
                     + mtx[6]*(mtx[1]*mtx[5]-mtx[2]*mtx[4]);
         break;*/
      default:
         Eigen::Map<Eigen::MatrixXd> temp(mtx, D, D);
         determinant = temp.determinant();
         break;
   }

   return determinant;
}

inline void InvMat2x2(double *mat1, double *result)
{
   double m0 = mat1[0];
   double m1 = mat1[1];
   double m2 = mat1[2];
   double m3 = mat1[3];
   double tmp_sum = (m0*m3-m1*m2);
   if(tmp_sum<=0 || fabs(m1-m2)>1E-5)
   {
      mexPrintf("%e\t%e\n%e\t%e\n", mat1[0], mat1[2], mat1[1], mat1[3]);
      mexErrMsgTxt("InvMat2x2: Input matrix should be positive definite! ");
   }

   tmp_sum = 1/tmp_sum;
   result[0] = m3 * tmp_sum;
   result[1] = -m1 * tmp_sum;
   result[2] = result[1];
   result[3] = m0 * tmp_sum;
}

inline void InvMat3x3(arr(double) x, arr(double) y) // assumes symmetric
{
   double determinant = det(x, 3);
   double a00 = x[0];
   double a01 = x[1];
   double a02 = x[2];
   double a11 = x[4];
   double a12 = x[5];
   double a22 = x[8];
   y[0] = (a22*a11 - a12*a12)/determinant;
   y[1] = (a02*a12 - a22*a01)/determinant;
   y[2] = (a12*a01 - a02*a11)/determinant;
   y[4] = (a00*a22 - a02*a02)/determinant;
   y[5] = (a02*a01 - a12*a00)/determinant;
   y[8] = (a11*a00 - a01*a01)/determinant;
   y[3] = y[1];
   y[6] = y[2];
   y[7] = y[5];
}

inline void InvMat(arr(double)A, int D)
{
   switch (D)
   {
      /*case 1:
         A[0] = 1.0/A[0];
         break;
      case 2:
         InvMat2x2(A, A);
         break;
      case 3:
         InvMat3x3(A, A);
         break;*/
      default:
         Eigen::Map<Eigen::MatrixXd> temp(A, D, D);
         temp = temp.inverse();
         break;
   }
}

// computes x'Ax, where x is Dx1, and A is DxD symmetric matrix
inline double xt_A_x(arr(double) x, arr(double)A, int D)
{
   double val;
   //x0, x1, x2, A00, A01, A02, A11, A12, A22;
   switch (D)
   {
      /*case 1:
         val = pow(x[0],2)*A[0];
         break;
      case 2:
         x0 = x[0];
         x1 = x[1];
         A00 = A[0];
         A01 = A[1];
         A11 = A[3];
         val = x0*(x0*A00 + x1*A01) + x1*(x0*A01 + x1*A11);
         break;
      case 3:
         x0 = x[0];
         x1 = x[1];
         x2 = x[2];
         A00 = A[0];
         A01 = A[1];
         A02 = A[2];
         A11 = A[4];
         A12 = A[5];
         A22 = A[8];
         val = x0*(x0*A00 + x1*A01 + x2*A02) + x1*(x0*A01 + x1*A11 + x2*A12) + x2*(x0*A02 + x1*A12 + x2*A22);
         break;*/
      default:
         val = 0;
         for (int d1=0; d1<D; d1++)
         {
            double data1 = x[d1];
            for (int d2=0; d2<D; d2++)
               val += data1*x[d2]*A[d1+d2*D];
         }
         break;
   }
   return val;
}

// computes (x-mu)'A(x-mu), where x is Dx1, and A is DxD symmetric matrix
inline double xmut_A_xmu(arr(double) x, arr(double) mu, arr(double)A, int D)
{
   double val;
   //x0, x1, x2, A00, A01, A02, A11, A12, A22;
   switch (D)
   {
      /*case 1:
         val = pow(x[0]-mu[0],2)*A[0];
         break;
      case 2:
         x0 = x[0]-mu[0];
         x1 = x[1]-mu[1];
         A00 = A[0];
         A01 = A[1];
         A11 = A[3];
         val = x0*(x0*A00 + x1*A01) + x1*(x0*A01 + x1*A11);
         break;
      case 3:
         x0 = x[0]-mu[0];
         x1 = x[1]-mu[1];
         x2 = x[2]-mu[2];
         A00 = A[0];
         A01 = A[1];
         A02 = A[2];
         A11 = A[4];
         A12 = A[5];
         A22 = A[8];
         val = x0*(x0*A00 + x1*A01 + x2*A02) + x1*(x0*A01 + x1*A11 + x2*A12) + x2*(x0*A02 + x1*A12 + x2*A22);
         break;*/
      default:
         val = 0;
         for (int d1=0; d1<D; d1++)
         {
            double data1 = x[d1]-mu[d1];
            for (int d2=0; d2<D; d2++)
               val += data1*(x[d2]-mu[d2])*A[d1+d2*D];
         }
         break;
   }
   return val;
}

inline double xt_Ainv_x(arr(double) x, arr(double)A, int D, arr(double) tempSpace)
{
   //double determinant = det(A, D);
   double val = -1.0; //, x0, x1, x2, A00, A01, A02, A11, A12, A22, a00, a01, a02, a11, a12, a22;
   switch (D)
   {
      /*case 1:
         val = pow(x[0],2)/A[0];
         break;
      case 2:
         x0 = x[0];
         x1 = x[1];
         A00 = A[3]/determinant;
         A01 = -A[1]/determinant;
         A11 = A[0]/determinant;
         val = x0*(x0*A00 + x1*A01) + x1*(x0*A01 + x1*A11);
         break;
      case 3:
         x0 = x[0];
         x1 = x[1];
         x2 = x[2];
         a00 = A[0];
         a01 = A[1];
         a02 = A[2];
         a11 = A[4];
         a12 = A[5];
         a22 = A[8];

         A00 = (a22*a11 - a12*a12)/determinant;
         A01 = (a02*a01 - a22*a01)/determinant;
         A02 = (a12*a01 - a02*a11)/determinant;
         A11 = (a00*a22 - a02*a02)/determinant;
         A12 = (a02*a01 - a12*a00)/determinant;
         A22 = (a11*a00 - a01*a01)/determinant;

         val = x0*(x0*A00 + x1*A01 + x2*A02) + x1*(x0*A01 + x1*A11 + x2*A12) + x2*(x0*A02 + x1*A12 + x2*A22);
         break;*/
      default:
         mexErrMsgTxt("xt_A_x not supported!");
         break;
   }
   return val;
}

inline void xxt(arr(double) x, int D, arr(double) output)
{
   // output should be DxD already
   for (int d1=0; d1<D; d1++) for (int d2=0; d2<D; d2++)
      output[d1+d2*D] = x[d1]*x[d2];
}

inline void xxt_add(arr(double) x, int D, arr(double) output)
{
   // output should be DxD already
   for (int d1=0; d1<D; d1++) for (int d2=0; d2<D; d2++)
      output[d1+d2*D] += x[d1]*x[d2];
}

inline void xxt_sub(arr(double) x, int D, arr(double) output)
{
   // output should be DxD already
   for (int d1=0; d1<D; d1++) for (int d2=0; d2<D; d2++)
      output[d1+d2*D] -= x[d1]*x[d2];
}


#endif

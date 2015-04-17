/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#include <mmf/optimizationSO3_approx.hpp>

//template<typename T>
//double OptSO3ApproxCpu<T>::conjugateGradient(Matrix<T,3,3>& R, uint32_t
//    maxIter)
//{
//  jsc::Timer t0;
//  uint32_t N = 0;
//  if( fabs(R.determinant()-1.0) > 1e-6)
//  {
//#ifndef NDEBUG
//    cout<<" == renormalizing rotation to get back to det(R) = 1"<<endl;
//#endif
//    R.col(0) = R.col(0)/R.col(0).norm();
//    R.col(1) = R.col(1)/R.col(1).norm();
//    R.col(2) = R.col(0).cross(R.col(1));
//  }
//  T res0 = conjugateGradientPreparation_impl(R,N);
//  dtPrep_ = t0.toctic("--- association ");
//  T resEnd = conjugateGradient_impl(R, res0, N, maxIter);
//  conjugateGradientPostparation_impl(R);
//  dtCG_ = t0.toctic("--- conjugateGradient");
//  t_++; // keep track of timesteps 
//  R_ = R; // keep track of previous rotation estimate for rot. velocity computation
//  return resEnd;
//};

template<typename T>
double OptSO3ApproxCpu<T>::conjugateGradient(Matrix<T,3,3>& R, 
      const Matrix<T,Dynamic,Dynamic>& qKarch, 
      const Matrix<T,Dynamic,1>& Ns, uint32_t maxIter)
{
//  jsc::Timer t0;
  if( fabs(R.determinant()-1.0) > 1e-6)
  {
#ifndef NDEBUG
    cout<<" == renormalizing rotation to get back to det(R) = 1"<<endl;
#endif
    R.col(0) = R.col(0)/R.col(0).norm();
    R.col(1) = R.col(1)/R.col(1).norm();
    R.col(2) = R.col(0).cross(R.col(1));
  }
  qKarch_ = qKarch;
  Ns_ = Ns;
  T res0 = this->evalCostFunction(R);
//  T res0 = conjugateGradientPreparation_impl(R,N);
//  dtPrep_ = t0.toctic("--- association ");
  T resEnd = conjugateGradient_impl(R, res0, Ns.sum(), maxIter);
  conjugateGradientPostparation_impl(R);
//  dtCG_ = t0.toctic("--- conjugateGradient");
  t_++; // keep track of timesteps 
  R_ = R; // keep track of previous rotation estimate for rot. velocity computation
  return resEnd;
};

//template<typename T>
//T OptSO3ApproxCpu<T>::conjugateGradientPreparation_impl( 
//    Matrix<T,3,3>& R, uint32_t& N)
//{
////  Timer t0;
//  N =0;
//  //TODO allow setting of indicators
//  T res0 = this->computeAssignment(R,N)/T(N);
//
//  if(this->t_ == 0){
//   // init karcher means with columns of the rotation matrix (takes longer!)
//  qKarch_ << R.col(0),-R.col(0),R.col(1),-R.col(1),R.col(2),-R.col(2);
////    // init karcher means with rotated version of previous karcher means
////    qKarch_ =  (this->R_*R.transpose())*qKarch_; 
//  }
//  qKarch_ = karcherMeans(qKarch_, 5.e-5, 10);
////  t0.toctic("----- karcher mean");
//  // compute a rotation matrix from the karcher means (not necessary)
//  Matrix<T,3,3> Rkarch;
//  Rkarch.col(0) =  qKarch_.col(0);
//  Rkarch.col(1) =  qKarch_.col(2) - qKarch_.col(2).dot(qKarch_.col(0))
//    *qKarch_.col(0);
//  Rkarch.col(1) /= Rkarch.col(1).norm();
//  Rkarch.col(2) = Rkarch.col(0).cross(Rkarch.col(1));
//#ifndef NDEBUG
//  cout<<"R: "<<endl<<R<<endl;
//  cout<<"Rkarch: "<<endl<<Rkarch<<endl<<"det(Rkarch)="<<Rkarch.determinant()<<endl;
//#endif
////  t0.tic();
//  computeSuffcientStatistics();
////  t0.toctic("----- sufficient statistics");
//  return res0; // cost fct value
//}

template<typename T>
T OptSO3ApproxCpu<T>::conjugateGradient_impl(Matrix<T,3,3>& R, T
    res0, uint32_t maxIter)
{
//  jsc::Timer t0;
  Matrix<T,3,3> G_prev, G, H, M_t_min, J;
//  Matrix<T,3,3> R = R0;
  vector<T> res(1,res0);

#ifndef NDEBUG
  cout<<"R0="<<endl<<R<<endl;
  cout<<"residual 0 = "<<res[0]<<endl;
#endif

  //T ts[10] = {0.0,0.1,0.2,0.3,0.4,0.5,0.7,1.0,1.5,2.0};
  //T ts[10] = {0.0,0.01,0.02,0.03,0.04,0.05,0.07,.1,.2,.3};
//  jsc::Timer t1;
  for(uint32_t i =0; i<maxIter; ++i)
  {
    computeJacobian(J,R,N);
#ifndef NDEBUG
  cout<<"J="<<endl<<J<<endl;
#endif
//    t0.toctic("--- jacobian ");
    updateGandH(G,G_prev,H,R,J,M_t_min,i%3 == 0); // want first iteration to reset H
#ifndef NDEBUG
  cout<<(i%3 == 2)<<endl;
  cout<<"R="<<endl<<R<<endl;
  cout<<"G="<<endl<<G<<endl;
  cout<<"H="<<endl<<H<<endl;
#endif
//    t0.toctic("--- update G and H ");
    T f_t_min = linesearch(R,M_t_min,H,t_max_,dt_);
//    t0.toctic("--- linesearch ");
    if(f_t_min == 999999.0f) break;

    res.push_back(f_t_min);
    T dresidual = res[res.size()-2] - res[res.size()-1];
    if( abs(dresidual) < 1e-7 )
    {
#ifndef NDEBUG
      cout<<"converged after "<<res.size()<<" delta residual "
        << dresidual <<" residual="<<res[res.size()-1]<<endl;
#endif
      break;
    }else{
#ifndef NDEBUG
      cout<<"delta residual " << dresidual
        <<" residual="<<res[res.size()-1]<<endl;
#endif
    }
  }
//  dtCG_ = t1.toc();
//  R0 = R; // update R
  return  res.back();
}

template<typename T>
void
OptSO3ApproxCpu<T>::conjugateGradientPostparation_impl(
    Matrix<T,3,3>& R)
{};

/* evaluate cost function for a given assignment of npormals to axes */
template<typename T>
T OptSO3ApproxCpu<T>::evalCostFunction(Matrix<T,3,3>& R)
{
  T c = 0.0f;
  for (uint32_t j=0; j<6; ++j)
  { 
    const T dot = max(-1.0f,min(1.0f,(qKarch_.col(j).transpose() * R.col(j/2))(0)));
    if(j%2 ==0){
      c += Ns_(j) * acos(dot)* acos(dot);
    }else{
      c += Ns_(j) * acos(-dot)* acos(-dot);
    }
  }
  return c/Ns_.sum();
}


/* compute Jacobian */
template<typename T>
void OptSO3ApproxCpu<T>::computeJacobian(Matrix<T,3,3>&J,
    Matrix<T,3,3>& R)
{
  J = Matrix<T,3,3>::Zero();
#ifndef NDEBUG
  cout<<"qKarch"<<endl<<qKarch_<<endl;
  cout<<"xSums_"<<endl<<xSums_<<endl;
  cout<<"Ns_"<<endl<<Ns_<<endl;
#endif
  for (uint32_t j=0; j<6; ++j)
  {
    // this is not using the robust cost function!
    T dot = max(-1.0f,min(1.0f,(qKarch_.col(j).transpose() * R.col(j/2))(0)));
    T eps = acos(dot);
    // if dot -> 1 the factor in front of qKarch -> 0
    if(j%2 ==0){
      eps = acos(dot);
      if(-0.99< dot && dot < 0.99)
        J.col(j/2) -= ((2.*Ns_(j)*eps)/(sqrt(1.f-dot*dot))) * qKarch_.col(j);
      else if(dot >= 0.99)
      { // taylor series around 0.99 according to Mathematica
       J.col(j/2) -= (2.*Ns_(j)*(1.0033467240646519 - 0.33601724502488395
             *(-0.99 + dot) + 0.13506297338381046* (-0.99 + dot)*(-0.99 + dot))) *
         qKarch_.col(j);
      }else if (dot <= -0.99)
      {
       J.col(j/2) -= (2.*Ns_(j)*(21.266813135156017 - 1108.2484926534892*(0.99
               + dot) + 83235.29739487475*(0.99 + dot)*(0.99 + dot))) *
         qKarch_.col(j);
      }
    }else{
      dot *= -1.;
      eps = acos(dot);
      if(-0.99< dot && dot < 0.99)
        J.col(j/2) += ((2.*Ns_(j)*eps)/(sqrt(1.f-dot*dot))) * qKarch_.col(j);
      else if(dot >= 0.99)
      { // taylor series around 0.99 according to Mathematica
       J.col(j/2) += (2.*Ns_(j)*(1.0033467240646519 - 0.33601724502488395
             *(-0.99 + dot) + 0.13506297338381046* (-0.99 + dot)*(-0.99 + dot))) *
         qKarch_.col(j);
      }else if (dot <= -0.99)
      {
       J.col(j/2) += (2.*Ns_(j)*(21.266813135156017 - 1108.2484926534892*(0.99
               + dot) + 83235.29739487475*(0.99 + dot)*(0.99 + dot))) *
         qKarch_.col(j);
      }
    }
//    cout<<" dot="<<dot<<" eps="<<eps<<" sqrt()="<<sqrt(1.f-dot*dot)
//      <<" N="<<Ns_(j)<<" J="<<endl<<J<<endl;
  }
}

//Matrix<T,3,6> OptSO3ApproxCpu<T>::meanInTpS2_GPU(Matrix<T,3,6>& p)
//{
//  Matrix<T,4,6> mu_karch = Matrix<T,4,6>::Zero();
//  T *h_mu_karch = mu_karch.data();
//  T *h_p = p.data();
//  meanInTpS2GPU(h_p, d_p_, h_mu_karch, d_mu_karch_, this->cld_.d_x(),
//      this->cld_.d_z(), d_weights_, this->cld_.N());
//  Matrix<T,3,6> mu = mu_karch.topRows(3);
//  for(uint32_t i=0; i<6; ++i)
//    if(mu_karch(3,i) >0)
//      mu.col(i) /= mu_karch(3,i);
//  return mu;
//}

//Matrix<T,3,6> OptSO3ApproxCpu<T>::karcherMeans(
//    const Matrix<T,3,6>& p0, T thresh, uint32_t maxIter)
//{
//  Matrix<T,3,6> p = p0;
//  Matrix<T,6,1> residuals;
//  for(uint32_t i=0; i< maxIter; ++i)
//  {
////    Timer t0;
//    Matrix<T,3,6> mu_karch = meanInTpS2_GPU(p);
////    t0.toctic("meanInTpS2_GPU");
//#ifndef NDEBUG
//    cout<<"mu_karch"<<endl<<mu_karch<<endl;
//#endif
//    residuals.fill(0.0f);
//    for (uint32_t j=0; j<6; ++j)
//    {
//      p.col(j) = S2_.Exp_p(p.col(j), mu_karch.col(j));
//      residuals(j) = mu_karch.col(j).norm();
//    }
//#ifndef NDEBUG
////    cout<<"p"<<endl<<p<<endl;
//    cout<<"karcherMeans "<<i<<" residuals="<<residuals.transpose()<<endl;
//#endif
//    if( (residuals.array() < thresh).all() )
//    {
//#ifndef NDEBUG
//      cout<<"converged after "<<i<<" residuals="
//        <<residuals.transpose()<<endl;
//#endif
//      break;
//    }
//  }
//  return p;
//}

//template<typename T>
//void OptSO3ApproxCpu<T>::computeSuffcientStatistics()
//{
//  // compute rotations to north pole
//  Matrix<T,2*6,3,RowMajor> Rnorths(2*6,3);
//  for (uint32_t j=0; j<6; ++j)
//  {
//    Rnorths.middleRows<2>(j*2) =
//      S2_.north_R_TpS2(qKarch_.col(j)).topRows<2>();
//  }
//
//  Matrix<T,7,6,ColMajor> SSs;
//  sufficientStatisticsOnTpS2GPU(qKarch_.data(), d_mu_karch_, 
//    Rnorths.data(), d_Rnorths_, this->cld_.d_x(), this->cld_.d_z() ,
//    this->cld_.N(), SSs.data(), d_SSs_);
//  
//  for (uint32_t j=0; j<6; ++j)
//  {
//    xSums_.col(j) = SSs.block<2,1>(0,j);
//    Ss_[j](0,0) =  SSs(2,j);
//    Ss_[j](0,1) =  SSs(3,j);
//    Ss_[j](1,0) =  SSs(4,j);
//    Ss_[j](1,1) =  SSs(5,j);
//    Ns_(j) = SSs(6,j);
//  }
//}

template<typename T>
void OptSO3ApproxCpu<T>::updateGandH(Matrix<T,3,3>& G, Matrix<T,3,3>&
    G_prev, Matrix<T,3,3>& H, const Matrix<T,3,3>& R, const
    Matrix<T,3,3>& J, const Matrix<T,3,3>&
    M_t_min, bool resetH)
{
  G_prev = G;
  G = J - R * J.transpose() * R;
  G = R*enforceSkewSymmetry(R.transpose()*G);

  if(resetH)
  {
    H= -G;
  }else{
    Matrix<T,3,3> tauH = H * M_t_min; //- R_prev * (RR.transpose() * N_t_min);
    T gamma = ((G-G_prev)*G).trace()/(G_prev*G_prev).trace();
    H = -G + gamma * tauH;
    H = R*enforceSkewSymmetry(R.transpose()*H);
  }
}

template<typename T>
T OptSO3ApproxCpu<T>::linesearch(Matrix<T,3,3>& R, Matrix<T,3,3>& M_t_min,
    const Matrix<T,3,3>& H, T t_max, T dt)
{
  Matrix<T,3,3> A = R.transpose() * H;

  EigenSolver<MatrixXf> eig(A);
  MatrixXcf U = eig.eigenvectors();
  MatrixXcf invU = U.inverse();
  VectorXcf d = eig.eigenvalues();
#ifndef NDEBUG
  cout<<"A"<<endl<<A<<endl;
  cout<<"U"<<endl<<U<<endl;
  cout<<"d"<<endl<<d<<endl;
#endif

  Matrix<T,3,3> R_t_min=R;
  T f_t_min = 999999.0f;
  T t_min = 0.0f;
  //for(int i_t =0; i_t<10; i_t++)
  for(T t =0.0f; t<t_max; t+=dt)
  {
    //T t= ts[i_t];
    VectorXcf expD = ((d*t).array().exp());
    MatrixXf MN = (U*expD.asDiagonal()*invU).real();
    Matrix<T,3,3> R_t = R*MN.topLeftCorner(3,3);

    T detR = R_t.determinant();
    T maxDeviationFromI = ((R_t*R_t.transpose() 
          - Matrix<T,3,3>::Identity()).cwiseAbs()).maxCoeff();
    if ((R_t(0,0)==R_t(0,0)) 
        && (abs(detR-1.0f)< 1e-2) 
        && (maxDeviationFromI <1e-1))
    {
      T f_t = evalCostFunction(R_t);
#ifndef NDEBUG
      cout<< " f_t = "<<f_t<<endl;
#endif
      if (f_t_min > f_t && f_t != 0.0f)
      {
        R_t_min = R_t;
        M_t_min = MN.topLeftCorner(3,3);
        f_t_min = f_t;
        t_min = t;
      }
    }else{
      cout<<"R_t is corrupted detR="<<detR
        <<"; max deviation from I="<<maxDeviationFromI 
        <<"; nans? "<<R_t(0,0)<<" f_t_min="<<f_t_min<<endl;
    }
  }
  if(f_t_min == 999999.0f) return f_t_min;
  // case where the MN is nans
  R = R_t_min;
#ifndef NDEBUG
  cout<<"R: det(R) = "<<R.determinant()<<endl<<R<<endl;
  cout<< "t_min="<<t_min<<" f_t_min="<<f_t_min<<endl;
#endif
  return f_t_min; 
}

template<typename T>
void OptSO3ApproxCpu<T>::Rot2M(Matrix<T,3,3>& R, T *mu)
{
  for(uint32_t k=0; k<6; ++k){
    int j = k/2; // which of the rotation columns does this belong to
    T sign = (- T(k%2) +0.5f)*2.0f; // sign of the axis
    mu[k] = sign*R(0,j);
    mu[k+6] = sign*R(1,j);
    mu[k+12] = sign*R(2,j);
  }
};

template<typename T>
Matrix<T,Dynamic,Dynamic> OptSO3ApproxCpu<T>::M()
{
  Matrix<T,Dynamic,Dynamic> M(3,6);
  for(uint32_t k=0; k<6; ++k){
    int j = k/2; // which of the rotation columns does this belong to
    T sign = (- T(k%2) +0.5f)*2.0f; // sign of the axis
    M.col(k) = sign*R_.col(j);
  }
  return M;
};

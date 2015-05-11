// Copyright (c) 2015 Qihoo 360 Technology Co. Ltd
// Author: Mu Yixiang (muyixiang@360.cn)

#include "stdio.h"
#include "math.h"
#include <cblas.h>
#include <iostream>
double matrix[9][4]={
    {1,47,76,24}, //include x0=1
    {1,46,77,23},
    {1,48,74,22},
    {1,34,76,21},
    {1,35,75,24},
    {1,34,77,25},
    {1,55,76,21},
    {1,56,74,22},
    {1,55,72,22},
};

double matrix1[36] = {
    1,47,76,24, //include x0=1
    1,46,77,23,
    1,48,74,22,
    1,34,76,21,
    1,35,75,24,
    1,34,77,25,
    1,55,76,21,
    1,56,74,22,
    1,55,72,22,

};

double result[]={1,
    1,
    1,
    2,
    2,
    2,
    3,
    3,
    3,};
double result1[]={
  1,0,0,
  1,0,0,
  1,0,0,
  0,1,0,
  0,1,0,
  0,1,0,
  0,0,1,
  0,0,1,
  0,0,1,

};

double theta[3][4]={
    {0.3,0.3,0.01,0.01},
    {0.5,0.5,0.01,0.01},
    {0.1,0.2,0.3,0.4}}; // include theta0

double theta1[12]={
    0.3,0.3,0.01,0.01,
    0.5,0.5,0.01,0.01,
    0.1,0.2,0.3,0.4}; // include theta0

double function_g(double x)
{
    double ex = pow(2.718281828,x);
    return ex/(1+ex);
}

double function_e(double x)
{
    return pow(2.718281828,x);
}

int main(void)
{
    enum CBLAS_ORDER Order=CblasRowMajor;
    enum CBLAS_TRANSPOSE TransX=CblasNoTrans;
    enum CBLAS_TRANSPOSE TransW=CblasTrans;
    int M=9;//X的行数，O的行数
    int N=3;//W的列数，O的列数
    int K=4;//X的列数，W的行数
    double alpha=1;
    double beta=0;
    int lda=K;
    int ldb=K;
    int ldc=N;
    double* tmp = new double[N*M];
    cblas_dgemm(Order, TransX, TransW,
                M, N, K,
                alpha, matrix1, lda,
                theta1, ldb,
                beta, tmp, ldc
                );
    for(int i=0;i<9;i++) {
      for(int j=0;j<3;j++) {
        tmp[i*3+j] = function_e(tmp[i*3+j]);
      }
    }
    double* p = new double[N*M];
    double* sum = new double[M];
    double ones[3]={1,1,1};

    Order = CblasRowMajor;
    TransX = CblasNoTrans;
    TransW = CblasTrans;
    M=9;
    N=1;
    K=3;
    alpha=1;
    beta = 0;
    lda=K;
    ldb=K;
    ldc=N;
    cblas_dgemm(Order, TransX, TransW,
                M, N, K,
                alpha, tmp, lda,
                ones, ldb,
                beta, sum, ldc
               );
    for(int i=0;i<9;i++) {
    }
    for(int i=0;i<9;i++) {
      for(int j=0;j<3;j++) {
        p[i*3+j]=tmp[i*3+j]/sum[i];
        std::cout<<p[i*3+j]<<" ";
      }
      double likelihood = log(p[i*3 + (int)result[i]-1]);
      std::cout<<" likelihood:"<<likelihood;
      std::cout<<std::endl;
    }

    double* A = new double[1000000];
    double* B = new double[1000000];
    for(int i=0;i<1000000;i++) {
      A[i]=i;
      B[i]=1;
    }
    double* C = new double[10000];
    while(true)
    {
 /*        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                  100,100,100,
                  1,A,100,
                  B,100,
                  0,C,100);*/
/*          cblas_dgemv(CblasRowMajor, CblasTrans,
                  5,5,
                  1,A,5,
                  B,1,
                  0,C,1);

        for(int i=0;i<100;i++){
        //  std::cout<<C[i]<<std::endl;
        }
        //std::cout<<std::endl;
*/
        cblas_dscal(1000000, 10, A, 1);
      //  for(int i=0;i<100;i++) {
      //    std::cout<<A[i]<<std::endl;
      //  }
        cblas_dscal(1000000, 0.1, A, 1);
      //  for(int i=0;i<100;i++) {
      //    std::cout<<A[i]<<std::endl;
      //  }
      //  std::cout<<std::endl;

  //    cblas_daxpy(10000, 1, A, 1, B, 1);
  //    cblas_daxpy(10000, -1, A, 1, B, 1);

      /*
        double* delta = new double[4*3];
        enum CBLAS_ORDER Order=CblasRowMajor;
        enum CBLAS_TRANSPOSE TransX=CblasNoTrans;
        enum CBLAS_TRANSPOSE TransW=CblasTrans;
        int M=9;//X的行数，O的行数
        int N=3;//W的列数，O的列数
        int K=4;//X的列数，W的行数
        double alpha=1;
        double beta=0;
        int lda=K;
        int ldb=K;
        int ldc=N;
        double* tmp = new double[N*M];
        cblas_dgemm(Order, TransX, TransW,
                    M, N, K,
                    alpha, matrix1, lda,
                    theta1, ldb,
                    beta, tmp, ldc
                    );
        for(int i=0;i<9;i++) {
          for(int j=0;j<3;j++) {
            tmp[i*3+j] = function_e(tmp[i*3+j]);
          }
        }
        double* p = new double[N*M];
        double* sum = new double[M];
        double ones[3]={1,1,1};

        Order = CblasRowMajor;
        TransX = CblasNoTrans;
        TransW = CblasTrans;
        M=9;
        N=1;
        K=3;
        alpha=1;
        beta = 0;
        lda=K;
        ldb=K;
        ldc=N;
        cblas_dgemm(Order, TransX, TransW,
                    M, N, K,
                    alpha, tmp, lda,
                    ones, ldb,
                    beta, sum, ldc
                   );
//        std::cout<<"sum:"<<std::endl;
        for(int i=0;i<9;i++) {
          for(int j=0;j<3;j++) {
            p[i*3+j]=tmp[i*3+j]/sum[i];
          }
          double likelihood = log(p[i*3 + (int)result[i]-1]);
        //  std::cout<<" my likelihood:"<<likelihood;
 //         std::cout<<sum[i]<<std::endl;
        }
        double* Y = new double[9*3];
        cblas_dcopy(9*3, result1, 1, Y, 1);
        alpha = -1;
        cblas_daxpy(9*3, alpha, p, 1, Y, 1);

        Order = CblasRowMajor;
        TransW = CblasTrans;
        TransX = CblasNoTrans;
        M = 3;
        N = 4;
        K = 9;
        lda = M;
        ldb = N;
        ldc = N;
        alpha = 1.0;
        beta = 0;
        delta = new double[M*N];
        cblas_dgemm(CblasRowMajor, TransW, TransX,
                    M, N, K,
                    alpha, Y, lda,
                    matrix1, ldb,
                    beta, delta, ldc);

        alpha=0.001;
        //cblas_daxpy(4*3, alpha, delta, 1, theta1, 1);

        Order = CblasRowMajor;
        TransX = CblasNoTrans;
        TransW = CblasTrans;
        M = 9;
        N = 3;
        K = 4;
        lda = K;
        ldb = K;
        ldc = N;
        alpha = 1;
        beta = 0;
        tmp = new double[9*3];
        cblas_dgemm(Order, TransX, TransW,
                    M, N, K,
                    alpha, matrix1, lda,
                    theta1, ldb,
                    beta, tmp, ldc
                    );
        for(int i=0;i<9;i++) {
          for(int j=0;j<3;j++) {
            tmp[i*3+j] = function_e(tmp[i*3+j]);
          }
        }

        sum = new double[M];
        //dones[3]={1,1,1};

        Order = CblasRowMajor;
        TransX = CblasNoTrans;
        TransW = CblasTrans;
        M=9;
        N=1;
        K=3;
        alpha=1;
        beta = 0;
        lda=K;
        ldb=K;
        ldc=N;
        cblas_dgemm(Order, TransX, TransW,
                    M, N, K,
                    alpha, tmp, lda,
                    ones, ldb,
                    beta, sum, ldc
                   );
   /*      for(int i=0;i<9;i++) {
          std::cout<<"sample:"<<i<<"\t";
          for(int j=0;j<3;j++) {
            p[i*3+j]=tmp[i*3+j]/sum[i];
            std::cout<<p[i*3+j]<<" ";
          }
          double likelihood = log(p[i*3 + (int)result[i]-1]);
          std::cout<<" likelihood:"<<likelihood;
          std::cout<<std::endl;
          }*/

    }
    return 0;
}



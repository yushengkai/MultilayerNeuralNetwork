// Copyright (c) 2015 Qihoo 360 Technology Co. Ltd
// Author: Yu Shengkai (yushengkai@360.cn)

#include "solver/sgd.h"
#include <iostream>
#include <cmath>
#include <cblas.h>
#include <glog/logging.h>
bool SGD::Init(NN* n, DataSet* trdata, DataSet* tedata, double lr, int m) {
  nn = n;
  trainData = trdata;
  testData = tedata;
  learning_rate = lr;
  minibatchsize = m;
  return true;
}

bool SGD::DoSolve() {
  int epoch=0;
  std::vector<double*> weight_matrixs = nn->weight_matrixs;
  std::vector<double*> delta_matrixs = nn->delta_matrixs;
  std::vector<double*> error_matrixs = nn->error_matrixs;
  std::vector<double*> layer_values = nn->layer_values;
  std::vector<int> layersizes = nn->layersizes;
  while(!isFinish()) {
    std::cout<<"epoch:"<<epoch++<<std::endl;
    int batch_num = 0;
    for(int i=0;i<trainData->length;i+=minibatchsize) {
      std::cout<<i<<"/"<<trainData->length<<std::endl;
      int batchsize = (i+minibatchsize>trainData->length)?
          trainData->length-i:minibatchsize;
      double* feature_start_ptr = trainData->feature + i*trainData->width;
      double* target_start_ptr = trainData->target + i;
      std::cout<<"nn forward"<<std::endl;
      nn->Forward(feature_start_ptr, batchsize);
      //update network output and intermediate variable
      std::cout<<"compute gradient"<<std::endl;
      ComputeGradient(weight_matrixs,
                      delta_matrixs,
                      error_matrixs,
                      layer_values,
                      layersizes,
                      target_start_ptr,
                      batchsize);
      std::cout<<"update weight"<<std::endl;
      UpdateWeight(weight_matrixs,
                   delta_matrixs,
                   layersizes);
    }
    double trainloss, testloss;
    LogLoss(trainData, trainloss);
    LogLoss(testData, testloss);
    std::cout<<"train loss:"<<trainloss<<std::endl;
    std::cout<<"test loss:"<<testloss<<std::endl;
  }
  return true;
}

bool SGD::ComputeGradient(std::vector<double*> weight_matrixs,
                          std::vector<double*> delta_matrixs,
                          std::vector<double*> error_matrixs,
                          std::vector<double*> layer_values,
                          std::vector<int> layersizes,
                          double* target,
                          int batchsize
                          ) {
  for(int layer=layersizes.size()-1;layer>=1;layer--){
    std::cout<<"layer:"<<layer<<std::endl;
    int layersize = layersizes[layer];
    double* factor = error_matrixs[layer-1];
    double* delta = delta_matrixs[layer-1];
    if(layer == layersizes.size()-1) {
      for(int i=0;i<batchsize;i++) {
        int t = (int)target[i];
        for(int j=0;j<layersize;j++) {
          factor[layersize*i + j] = 0;
        }
        factor[layersize*i + t]=1;
      }

      double* output = layer_values.back();//minibatchsize *2
      int N = batchsize * layersize;
      double alpha = -1;
      cblas_daxpy(N, alpha, output, 1, factor, 1);

      double* X = layer_values[layer-1];
      int hidelayer_size = layersizes[layer-1];
      int weight_idx = layer-1;
      const int M = layersize;
      N = hidelayer_size;
      const int K = batchsize;
      const int lda = M;
      const int ldb = N;
      const int ldc = N;
      cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                M, N, K,
                1.0, factor, lda,
                X, ldb,
                0.0, delta, ldc
                );
      std::cout<<"finish if..."<<std::endl;
    } else {
      std::cout<<"get into else..."<<std::endl;
      int weight_idx = layer-1;
      int downstream_layersize = layersizes[layer+1];
      double* downstream_factor = error_matrixs[weight_idx+1];//下游的
      double* downstream_weight = weight_matrixs[weight_idx+1];
      int M = batchsize;
      int N = layersize;
      int K = downstream_layersize;
      int lda = K;
      int ldb = K;
      int ldc = N;
      std::cout<<"before cblas degemm 1 ..."<<std::endl;
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                  M, N, K,
                  -1.0, downstream_factor, lda,
                  downstream_weight, ldb,
                  0.0, factor, ldc
                 );//把下游的误差，通过权值累加到这一层
      double* O = layer_values[layer];
      std::cout<<"after cblas dgemm 2 ..."<<std::endl;

      for(int i=0;i<batchsize;i++) {
        for(int j=0;j<layersize;j++) {
          int idx=i*layersize+j;
          factor[idx]=factor[idx]*O[idx]*(1-O[idx]);
        }
      }
      double* X = layer_values[layer-1];
      int hidelayer_size = layersizes[layer-1];
      M = layersize;
      N = hidelayer_size;
      K = batchsize;
      lda = M;
      ldb = N;
      ldc = N;
      std::cout<<"before cblas dgemm 2 ..."<<std::endl;
      std::cout<<"X:"<<std::endl;
      for(int i=0;i<batchsize;i++) {
        for(int j=0;j<hidelayer_size;j++) {
          std::cout<<X[i*hidelayer_size+j]<<" ";
        }
        std::cout<<std::endl;
      }
      std::cout<<"factor:"<<std::endl;

      cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                  M, N, K,
                  -1.0, factor, lda,
                  X, ldb,
                  0.0, delta, ldc
                 );
      std::cout<<"after cblas dgemm 2 ..."<<std::endl;
    }

//    int hidelayer_size=layersizes[layer-1];
//    int weight_idx = layer - 1;
//    cblas_daxpy(layersize*hidelayer_size, learning_rate, delta, 1,
//                weight_matrixs[weight_idx], 1
//               );
//  在这里不减梯度，计算完所有的梯度之后，在另外一个函数里面更新权重
  }
  return true;
}

bool SGD::UpdateWeight(std::vector<double*> weight_matrixs,
                       std::vector<double*> delta_matrixs,
                       std::vector<int> layersizes) {
  for(int layer=layersizes.size()-1;layer>=1;layer--){
    int layersize = layersizes[layer];
    int hidelayer_size=layersizes[layer-1];
    double* delta = delta_matrixs[layer-1];
    double* weight = weight_matrixs[layer-1];
    cblas_daxpy(layersize*hidelayer_size, learning_rate, delta, 1,
            weight, 1
          );

  }
  return true;
}

bool SGD::isFinish() {
  return false;
}

bool SGD::LogLoss(DataSet* dataset, double &logloss) {
  //call LogLoss after calling Forward
  double* feature = dataset->feature;
  double* target = dataset->target;
  int instancenum = dataset->length;
  int outputsize = nn->GetOutputSize();
  logloss = 0;
  double* logloss_tmp = new double[1];
  for(int i=0;i<instancenum;i+=minibatchsize) {
    int batchsize =
        i+minibatchsize<instancenum ? minibatchsize : instancenum - i;
    double* feature_start_ptr = feature + i*nn->GetInputSize();
    double* target_start_ptr = target + i;

    nn->Forward(feature_start_ptr, batchsize);
    double* y  = nn->layer_values.back();
    for(int b=0;b<batchsize;b++) {
//      std::cout<<std::endl;
//      std::cout<<target_start_ptr[b]<<std::endl;
      for(int j=0;j<outputsize;j++) {
//        std::cout<<y[b*outputsize+j]<<" ";
        y[b*outputsize+j] = log(y[b*outputsize+j]);
      }
//      std::cout<<std::endl;
    }
    double* factor = nn->error_matrixs.back();//借用一下这个空间
    for(int i=0;i<batchsize;i++) {
        int t = (int)target_start_ptr[i];
        for(int j=0;j<outputsize;j++) {
          factor[outputsize*i + j] = 0;
        }
        factor[outputsize*i + t]=1;
      }


    const enum CBLAS_ORDER Order = CblasRowMajor;
    const enum CBLAS_TRANSPOSE TransT = CblasNoTrans;
    const enum CBLAS_TRANSPOSE TransO = CblasTrans;
    const int M = 1;
    const int N = 1;
    const int K = batchsize * outputsize;
    const double alpha = 1;
    const double beta = 0;
    const int lda = K;
    const int ldb = K;
    const int ldc = N;
    cblas_dgemm(Order, TransT, TransO,
                M, N, K,
                alpha, factor, lda,
                y, ldb,
                beta, logloss_tmp, ldc
               );
      logloss+=logloss_tmp[0];
  }
  logloss/=instancenum;
  logloss = -logloss;
  delete [] logloss_tmp;
  return true;
}



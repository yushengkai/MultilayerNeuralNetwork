// Copyright (c) 2015 Qihoo 360 Technology Co. Ltd
// Author: Yu Shengkai (yushengkai@360.cn)


#ifndef NN_H_
#define NN_H_

#include <vector>
#include <string>
#include "main/LookupTable.h"

class NN {
 private:
  int inputsize;
  int outputsize;
  int minibatchsize;
  std::vector<int> layersizes;
  std::vector<double*> weight_matrixs;
  std::vector<double*> bias_vectors;
  std::vector<double*> layer_values;
  std::vector<double*> delta_matrixs;
  std::vector<double*> error_matrixs;
//  std::vector<double*> layer_sigmas;
  double* nn_output;
  double* softmax_sum;
  bool with_bias;
  double learning_rate;
 public:
  NN(){}
  bool Init(LookupTable *lookup_table, std::string param,
            int m, std::string init_type, bool wb);
  bool Forward(double* input, int batchsize);
  bool LogLoss(double* target, double& logloss);
  bool Derivative(double* target, int batchsize);
  bool Train(double* feature, double* target, int instancenum);
  void InitWeight(std::string init_type);
  int GetMiniBatchSize(){return minibatchsize;}
  int GetOutputSize(){return outputsize;}
  double* GetOutput(){return nn_output;}
//  void CompareWithTorch();
};



#endif  // NN_H_


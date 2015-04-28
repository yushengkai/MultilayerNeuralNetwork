// Copyright (c) 2015 Qihoo 360 Technology Co. Ltd
// Author: Yu Shengkai (yushengkai@360.cn)


#ifndef NN_H_
#define NN_H_

#include <vector>
#include <string>
#include "net/LookupTable.h"
#include "tool/util.h"
class NN {
 private:
  int inputsize;
  int outputsize;
  int minibatchsize;
  double* softmax_sum;
  bool with_bias;
  double learning_rate;
 public:
  std::vector<int> layersizes;
  std::vector<double*> weight_matrixs;
  std::vector<double*> bias_vectors;
  std::vector<double*> layer_values;
  std::vector<double*> delta_matrixs;
  std::vector<double*> error_matrixs;
  double* nn_output;
  NN(){}
  bool Init(LookupTable *lookup_table, std::string param,
            int m, std::string init_type, bool wb, double l);
  bool Forward(double* input, int batchsize);
  bool LogLoss(double* feature, double* target, double &logloss, int instancenum);
  bool Derivative(double* target, int batchsize);
  bool Train(DataSet* trainData);
  void InitWeight(std::string init_type);
  int GetMinibatchSize(){return minibatchsize;}
  int GetOutputSize(){return outputsize;}
  int GetInputSize(){return inputsize;}
  double* GetOutput(){return nn_output;}
  void CompareWithTorch();
};



#endif  // NN_H_


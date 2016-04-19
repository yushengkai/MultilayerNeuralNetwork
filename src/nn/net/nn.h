// Copyright (c) 2015 Qihoo 360 Technology Co. Ltd
// Author: Yu Shengkai (yushengkai@360.cn)


#ifndef NN_H_
#define NN_H_

#include <vector>
#include <string>
#include <map>
#include "net/lookup_table.h"
#include "net/bias_layer.h"
#include "tool/util.h"

class NN {
 private:
  int inputsize;
  int outputsize;
  int minibatchsize;
  double* softmax_sum;
  bool with_bias;
  double learning_rate;
  int table_width;
  int total_length;
  std::vector<int> group_sizes;
  std::vector<double*> group_ptrs;
  double* tmp_ones;
 public:

  LookupTable* lookup_table;
  Bias_Layer* bias_layer;
  std::vector<int> layersizes;
  std::vector<int> bias_feature;
  std::vector<double*> weight_matrixs;
  std::vector<double*> bias_vectors;
  std::vector<double*> layer_values;
  std::vector<double*> delta_matrixs;
  std::vector<double*> error_matrixs;
  std::map<int, double*> embedding_delta;
  double* delta_x;
  double* nn_output;
  double* nn_input;
  ~NN();
  bool Init(LookupTable *lookup_table, std::string layer_param,
            std::string bias_param, int m, std::string init_type,
            bool wb, double l);
  bool LookupFromTable(SparseDataSet* dataset);
  bool Forward(double* input, int batchsize);
  bool SparseForward(SparseDataSet* dataset);
  bool AUCLogLoss(SparseDataSet* dataset, double& auc, double &logloss);
  bool AUC(SparseDataSet* trainData, double &auc, int instancenum);

  bool Derivative(SparseDataSet* dataset);
  bool Train(SparseDataSet* trainData, SparseDataSet* testData);
  void InitWeight(std::string init_type);
  int GetMinibatchSize(){return minibatchsize;}
  int GetOutputSize(){return outputsize;}
  int GetInputSize(){return inputsize;}
  double* GetOutput(){return nn_output;}
  void CompareWithTorch();
  std::string PositionBucket(int* featureid);
};



#endif  // NN_H_


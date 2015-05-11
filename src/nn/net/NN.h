// Copyright (c) 2015 Qihoo 360 Technology Co. Ltd
// Author: Yu Shengkai (yushengkai@360.cn)


#ifndef NN_H_
#define NN_H_

#include <vector>
#include <string>
#include <map>
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
  int table_width;
  int total_length;
  std::vector<int> group_sizes;
  std::vector<double*> group_ptrs;
  double* tmp_ones;
 public:

  LookupTable* lookup_table;
  std::vector<int> layersizes;
  std::vector<double*> weight_matrixs;
  std::vector<double*> bias_vectors;
  std::vector<double*> layer_values;
  std::vector<double*> delta_matrixs;
  std::vector<double*> error_matrixs;
  std::map<int, double*> embedding_delta;
  double* delta_x;
  double* nn_output;
  double* nn_input;
  ~NN(){
    for(int layer=0;layer<layersizes.size();layer++) {
      if(layer>=1) {
        delete [] weight_matrixs[layer-1];
        delete [] bias_vectors[layer-1];
        delete [] delta_matrixs[layer-1];
        delete [] error_matrixs[layer-1];
      }
      delete [] layer_values[layer];

    }
    for(std::map<int, double*>::iterator iter=embedding_delta.begin();
        iter!=embedding_delta.end();iter++) {
      double* ptr = iter->second;
      delete [] ptr;
    }
    delete [] delta_x;
  }
  bool Init(LookupTable *lookup_table, std::string param,
            int m, std::string init_type, bool wb, double l);
  bool LookupFromTable(SparseDataSet* sparse_feature);
  bool Forward(double* input, int batchsize);
  bool SparseForward(SparseDataSet* sparse_feature);
  bool LogLoss(SparseDataSet* trainData, double &logloss, int instancenum);
  bool Derivative(SparseDataSet* trainData);
  bool Train(SparseDataSet* trainData);
  void InitWeight(std::string init_type);
  int GetMinibatchSize(){return minibatchsize;}
  int GetOutputSize(){return outputsize;}
  int GetInputSize(){return inputsize;}
  double* GetOutput(){return nn_output;}
  void CompareWithTorch();
};



#endif  // NN_H_


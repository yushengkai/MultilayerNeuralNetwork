// Copyright (c) 2015 Qihoo 360 Technology Co. Ltd
// Author: Yu Shengkai (yushengkai@360.cn)

#ifndef SGD_H_
#define SGD_H_

#include <vector>

#include "net/nn.h"
#include "tool/util.h"
class SGD {
 private:
  NN* nn;
  DataSet* trainData;
  DataSet* testData;
  int minibatchsize;
  double learning_rate;
 public:
  bool Init(NN* n, DataSet* trdata, DataSet* teData, double lr, int m);
  bool DoSolve();
  bool ComputeGradient(std::vector<double*> weight_matrixs,
                std::vector<double*> delta_matrixs,
                std::vector<double*> error_matrixs,
                std::vector<double*> layer_values,
                std::vector<int> layersizes,
                double* target,
                int batchsize);//static
  bool UpdateWeight(std::vector<double*> weight_matrixs,
                    std::vector<double*> delta_matrixs,
                    std::vector<int> layersizes);
  bool LogLoss(DataSet* dataset, double &logloss);
  bool isFinish();
};

#endif  // SGD_H_


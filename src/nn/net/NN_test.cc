// Copyright (c) 2015 Qihoo 360 Technology Co. Ltd
// Author: Mu Yixiang (muyixiang@360.cn)


#include "gtest/gtest.h"
#include "gflags/gflags.h"
#include "net/NN.h"
#include <iostream>
#include <fstream>
#include <vector>
#include "gflag/flag.h"
#include <string>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

TEST(DeltaTest, TestCase) {
  LookupTable* lookup_table = new LookupTable();
  if(lookup_table->Init(FLAGS_lookup_table_param, FLAGS_lookup_table_width)) {

  }

  NN* nn = new NN();
  nn->Init(lookup_table, FLAGS_nn_param, FLAGS_minibatchsize,
           FLAGS_init_type, FLAGS_with_bias, FLAGS_learning_rate);
  nn->CompareWithTorch();
  std::vector<double*> delta_matrixs = nn->delta_matrixs;
  std::ifstream fin("data/unittest.grad");
  std::string line="";
  for(int layer=1;layer<nn->layersizes.size();layer++) {
    int node_num = nn->layersizes[layer];
    int input_num = nn->layersizes[layer-1];
    double* delta = delta_matrixs[layer-1];
    double* delta_bias = delta + node_num*input_num;
    for(int i=0;i<node_num*input_num;i++) {
      getline(fin, line);
      boost::trim(line);
      double torch_grad = boost::lexical_cast<double>(line);
      double my_grad = delta[i];
      EXPECT_NEAR(torch_grad, my_grad, 1e-6);
    }
    for(int i=0;i<node_num;i++) {
      getline(fin, line);
      boost::trim(line);
      double torch_grad = boost::lexical_cast<double>(line);
      double my_grad = delta_bias[i];
      EXPECT_NEAR(torch_grad, my_grad, 1e-6);
    }
  }
  fin.close();
}

TEST(ForwardTest, TestCase) {
  LookupTable* lookup_table = new LookupTable();
  if(lookup_table->Init(FLAGS_lookup_table_param, FLAGS_lookup_table_width)) {

  }

  NN* nn = new NN();
  nn->Init(lookup_table, FLAGS_nn_param, FLAGS_minibatchsize,
           FLAGS_init_type, FLAGS_with_bias, FLAGS_learning_rate);
  double* output = NULL;
  nn->CompareWithTorch();
  std::vector<double*> delta_matrixs = nn->delta_matrixs;
  std::ifstream fin("data/unittest.output");
  std::string line="";
  output = nn->nn_output;

  for(int i=0;i<nn->layersizes.back()*FLAGS_test_batchsize;i++) {
    double my_output = output[i];
    getline(fin, line);
    boost::trim(line);
    double torch_output = boost::lexical_cast<double>(line);
    EXPECT_NEAR(torch_output, my_output , 1e-6);
  }
  fin.close();
}

TEST(UpdateTest, TestCase) {
  LookupTable* lookup_table = new LookupTable();
  if(lookup_table->Init(FLAGS_lookup_table_param, FLAGS_lookup_table_width)) {

  }

  NN* nn = new NN();
  nn->Init(lookup_table, FLAGS_nn_param, FLAGS_minibatchsize,
           FLAGS_init_type, FLAGS_with_bias, FLAGS_learning_rate);
  double* output = NULL;
  nn->CompareWithTorch();
  std::vector<double*> delta_matrixs = nn->delta_matrixs;
  std::ifstream fin("data/weight.update");
  std::string line="";
  output = nn->nn_output;
 int idx=0;
  for(int layer=1;layer<nn->layersizes.size();layer++) {
    int node_num = nn->layersizes[layer];
    int input_num = nn->layersizes[layer-1];
    double* weight = nn->weight_matrixs[layer-1];
    double* weight_bias = nn->bias_vectors[layer-1];
    for(int i=0;i<node_num*input_num;i++) {
      double my_weight = weight[i];
      getline(fin, line);
      boost::trim(line);
      double torch_weight = boost::lexical_cast<double>(line);
      EXPECT_NEAR(torch_weight, my_weight , 1e-6);
    }
    for(int i=0;i<node_num;i++) {
      double my_weight = weight_bias[i];
      getline(fin, line);
      boost::trim(line);
      double torch_weight = boost::lexical_cast<double>(line);
      EXPECT_NEAR(torch_weight, my_weight, 1e-6);
    }
  }
  fin.close();
}



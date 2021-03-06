// Copyright (c) 2015 Qihoo 360 Technology Co. Ltd
// Author: Mu Yixiang (muyixiang@360.cn)


#include "gtest/gtest.h"
#include "gflags/gflags.h"
#include "net/nn.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include "gflag/flag.h"
#include <string>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
/*
class NN_test: public  ::testing::Test {
 protected:
  NN* nn;
 public:
  virtual void SetUp(){
    std::cout<<"NN_test SetUp"<<std::endl;
     LookupTable* lookup_table = new LookupTable();
    if(lookup_table->Init(FLAGS_lookup_table_param, FLAGS_lookup_table_width)) {

    }
    std::cout<<"init lookup table"<<std::endl;
     nn->Init(lookup_table, FLAGS_nn_param, FLAGS_minibatchsize,
              FLAGS_init_type, FLAGS_with_bias, FLAGS_learning_rate);
     std::cout<<"init nn"<<std::endl;
     nn->CompareWithTorch();
  }
  virtual void TearDown() {}
};

TEST_F(NN_test, TestCase) {
  std::vector<double*> delta_matrixs = nn->delta_matrixs;
}
*/
TEST(NNDelta_Test, TestCase) {
  LookupTable* lookup_table = new LookupTable();
  std::map<int, int> term_feature_map;
  if(lookup_table->Init(FLAGS_lookup_table_param, term_feature_map, FLAGS_lookup_table_width)) {

  }
  NN* nn = new NN();
  nn->Init(lookup_table, FLAGS_nn_layer_param, FLAGS_bias_param, FLAGS_minibatchsize,
           FLAGS_init_type, FLAGS_with_bias, FLAGS_learning_rate);

  nn->CompareWithTorch();
  std::vector<double*> delta_matrixs = nn->delta_matrixs;

  std::ifstream fin("data/sparse_unittest.delta");
  
  EXPECT_TRUE(fin!=NULL);
  std::string line="";
  for(int i=0;i<15000000;i++) {
      getline(fin, line);
  }
  std::cout<<"here !"<<std::endl;
  std::ofstream fout("data/gtest_delta");
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
      fout<<torch_grad<<"\t"<<my_grad<<std::endl;
    }
  }
  fin.close();
  fout.close();
}

TEST(Embedding_test, TestCase) {
  LookupTable* lookup_table = new LookupTable();
  std::map<int, int> term_feature_map;
  if(lookup_table->Init(FLAGS_lookup_table_param, term_feature_map, FLAGS_lookup_table_width)) {

  }
  NN* nn = new NN();
  nn->Init(lookup_table, FLAGS_nn_layer_param, FLAGS_bias_param, FLAGS_minibatchsize,
           FLAGS_init_type, FLAGS_with_bias, FLAGS_learning_rate);

  nn->CompareWithTorch();
  int table_width = lookup_table->table_width;
  std::string line;
  std::ofstream fout("data/gtest_delta");
  double* delta;
  std::ifstream fin("data/sparse_unittest.delta");

  std::map<int, double*> embedding_delta_torch;
  for(int i=0;i<lookup_table->total_length;i++) {
    int real_feature_id = i/table_width;
    getline(fin, line);
    boost::trim(line);
    double torch_value = boost::lexical_cast<double>(line);
    if(i%table_width==0) {
      delta = new double[table_width];
      embedding_delta_torch[real_feature_id] = delta;
    }
    delta[i%table_width] = torch_value;
  }
  std::map<int, double*> embedding_delta = nn->embedding_delta;
  EXPECT_TRUE(fin!=NULL);
  std::vector<std::string> parts;
  for(std::map<int, double*>::iterator iter=embedding_delta.begin();
      iter!=embedding_delta.end();iter++) {
    //getline(fin, line);
    int real_featureid = iter->first;
    double* delta = iter->second;
    //boost::trim(line);
    //boost::split(parts, line, boost::is_any_of(" "));
    for(int i=0;i<table_width;i++) {
//      getline(fin, line);
//      boost::trim(line);
     
     // boost::trim(parts[i]);
//      double torch_value = boost::lexical_cast<double>(line);
      double torch_value = embedding_delta_torch[real_featureid][i];
      double my_value = delta[i];
      EXPECT_NEAR(torch_value, my_value, 1e-6);
      //std::cout<<torch_value<<"\t"<<my_value<<"\t"<<torch_value/my_value<<std::endl;
      fout<<torch_value<<"\t"<<my_value<<std::endl;

    }
  }  
  fin.close();
  fout.close();
}

TEST(ForwardTest, TestCase) {
  LookupTable* lookup_table = new LookupTable();
  std::map<int, int> term_feature_map;
  if(lookup_table->Init(FLAGS_lookup_table_param, term_feature_map, FLAGS_lookup_table_width)) {

  }

  NN* nn = new NN();
  nn->Init(lookup_table, FLAGS_nn_layer_param, FLAGS_bias_param, FLAGS_minibatchsize,
           FLAGS_init_type, FLAGS_with_bias, FLAGS_learning_rate);
  double* output = NULL;
  nn->CompareWithTorch();
  std::vector<double*> delta_matrixs = nn->delta_matrixs;
  std::ifstream fin("data/sparse_unittest.output");
  EXPECT_TRUE(fin!=NULL);
  std::string line="";
  output = nn->nn_output;
  std::vector<std::string> parts;
  std::ofstream fout("data/gtest_delta");

  for(int i=0;i<FLAGS_test_batchsize;i++) {
    getline(fin, line);
    boost::trim(line);
    boost::split(parts, line, boost::is_any_of(" "));

    double my_output = output[i*2];
    my_output = log(my_output);
    double torch_output = boost::lexical_cast<double>(parts[0]);
    EXPECT_NEAR(torch_output, my_output , 1e-6);
    fout<<torch_output<<"\t"<<my_output<<std::endl;
    torch_output = boost::lexical_cast<double>(parts[1]);
    my_output = output[i*2+1];
    my_output = log(my_output);
    EXPECT_NEAR(torch_output, my_output , 1e-6);
    fout<<torch_output<<"\t"<<my_output<<std::endl;

  }
  fin.close();
  fout.close();
}

TEST(UpdateTest, TestCase) {
  LookupTable* lookup_table = new LookupTable();
  std::map<int, int> term_feature_map;
  if(lookup_table->Init(FLAGS_lookup_table_param, term_feature_map, FLAGS_lookup_table_width)) {

  }

  NN* nn = new NN();
  nn->Init(lookup_table, FLAGS_nn_layer_param, FLAGS_bias_param, FLAGS_minibatchsize,
           FLAGS_init_type, FLAGS_with_bias, FLAGS_learning_rate);
  nn->CompareWithTorch();
  //std::vector<double*> delta_matrixs = nn->delta_matrixs;
  std::ifstream fin("data/sparse_unittest.update");
  EXPECT_TRUE(fin!=NULL);
  std::string line="";
  int table_width = lookup_table->table_width;
  for(int i=0;i<lookup_table->group_sizes.size();i++){
    for(int j=0;j<lookup_table->group_sizes[i];j++) {
      for(int k=0;k<table_width;k++) {
        getline(fin, line);
        boost::trim(line);
        double torch_value = boost::lexical_cast<double>(line);
        double my_value = lookup_table->group_ptrs[i][j*table_width+k];
        EXPECT_NEAR(torch_value, my_value, 1e-6);
       // std::cout<<"\t"<<torch_value<<"\t"<<my_value<<"\t"<<torch_value/my_value<<std::endl;
      }
    }
  }
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


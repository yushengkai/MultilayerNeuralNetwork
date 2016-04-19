// Copyright () 2015 Qihoo 360 Technology Co. Ltd
// Author: Yu Shengkai (yushengkai@360.cn)
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <iostream>
#include "net/nn.h"
#include "net/bias_layer.h"
#include "tool/util.h"
#include "solver/sgd.h"
#include "gflag/flag.h"
/*
DECLARE_string(lookup_table_param);
DECLARE_int32(lookup_table_width);
DECLARE_int32(minibatchsize);
DECLARE_double(sigma);
DECLARE_double(mu);
DECLARE_string(init_type);
DECLARE_string(nn_param);
DECLARE_bool(with_bias);
DECLARE_string(tranfer_func);
DECLARE_string(log_dir);
DECLARE_double(learning_rate);
DECLARE_bool(logtostderr);
*/

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  FLAGS_log_dir = "./log";
  std::string trainfolder = "/data/yushengkai/sparse_dnn/test_binary";
  std::string testfolder = "/data/yushengkai/sparse_dnn/test_binary";
  //binaryname = "data/tmp.txt";
  SparseDataSet* trainData = new SparseDataSet();
  SparseDataSet* testData = new SparseDataSet();
  std::map<int, int> term_feature_map;
  if(!ReadSparseDataFromBinFolder(trainfolder, FLAGS_bias_feature,
                                  FLAGS_term_feature, term_feature_map, trainData)) {
    LOG(ERROR)<<"Read Sparse data failed...";
  }
  if(!ReadSparseDataFromBinFolder(testfolder, FLAGS_bias_feature,
                                  FLAGS_term_feature, term_feature_map, testData)) {

  }
//  RemovePositionFeature(testData);
//  RemovePositionFeature(trainData);
  LookupTable* lookup_table = new LookupTable();
  if(lookup_table->Init(trainData->table_param, term_feature_map, FLAGS_lookup_table_width)) {

  }
  std::string bias_param = trainData->bias_param;
  std::cout<<"bias_param:"<<bias_param<<std::endl;
  NN* nn = new NN();
  nn->Init(lookup_table, FLAGS_nn_layer_param, bias_param, FLAGS_minibatchsize,
           FLAGS_init_type, FLAGS_with_bias, FLAGS_learning_rate);
  std::cerr<<"init NN finish!"<<std::endl;
  nn->Train(trainData, testData);
  DeleteSparseData(trainData);

  /*
  std::string filename = "data/train.csv";
  LOG(INFO)<<"Begin to read train data...";
  std::cout<<"Begin to read train data..."<<std::endl;
  DataSet* trainData = new DataSet();
  DataSet* testData = new DataSet();
  if(!ReadMNIST(filename, trainData)) {
    LOG(ERROR)<<"Read MNIST faild...";
  }
  LOG(INFO)<<"Begin to read test data...";
  std::cout<<"Begin to read test data..."<<std::endl;
  filename = "data/test.csv";
  if(!ReadMNIST(filename, testData)) {
    LOG(ERROR)<<"Read MNIST faild...";
  }
  std::cout<<"Begin to train"<<std::endl;
  LOG(INFO)<<"Begin to train";

  SGD* sgd_solver = new SGD();

  sgd_solver->Init(nn, trainData, testData, FLAGS_learning_rate, FLAGS_minibatchsize);
  nn->Train(trainData);
  */
  //sgd_solver->DoSolve();
}

// Copyright () 2015 Qihoo 360 Technology Co. Ltd
// Author: Yu Shengkai (yushengkai@360.cn)
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <iostream>
#include "net/NN.h"
#include "tool/util.h"
#include "solver/sgd.h"
/*
DEFINE_string(lookup_table_param, "10:20:30:40", "lengths of lookup tables");
DEFINE_int32(lookup_table_width, 196, "width of all the lookup tables");
DEFINE_int32(minibatchsize, 50, "minibatchsize");
DEFINE_double(sigma, 0.1, "sigma of gaussian distribution");
DEFINE_double(mu, 0, "mu of gaussian distribution");
DEFINE_string(init_type, "normal", "initial type normal, 123, 1, 0, fromfile");
DEFINE_string(nn_param, "200:10", "layer size of nn");
DEFINE_bool(with_bias, false, "with bias");
DEFINE_string(tranfer_func, "sigmoid", "sigmoid, tanh, ReLU");
DEFINE_string(log_dir, "./log", "log dir");
DEFINE_double(learning_rate, 0.001, "learning rate");
DEFINE_bool(logtostderr, true, "log into stderr");
*/


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


int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  FLAGS_log_dir = "./log";

  LookupTable* lookup_table = new LookupTable();
  if(lookup_table->Init(FLAGS_lookup_table_param, FLAGS_lookup_table_width)) {

  }

  NN* nn = new NN();
  nn->Init(lookup_table, FLAGS_nn_param, FLAGS_minibatchsize,
           FLAGS_init_type, FLAGS_with_bias, FLAGS_learning_rate);
  int featuresize;
  int instancenum;
  double* feature = NULL;
  double* target = NULL;
  nn->CompareWithTorch();
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

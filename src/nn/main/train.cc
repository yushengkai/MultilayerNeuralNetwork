// Copyright () 2015 Qihoo 360 Technology Co. Ltd
// Author: Yu Shengkai (yushengkai@360.cn)
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <iostream>
#include "main/NN.h"
#include "main/util.h"


DEFINE_string(lookup_table_param, "10:20:30:40", "lengths of lookup tables");
DEFINE_int32(lookup_table_width, 196, "width of all the lookup tables");
DEFINE_int32(minibatchsize, 50, "minibatchsize");
DEFINE_double(sigma, 0.1, "sigma of gaussian distribution");
DEFINE_double(mu, 0, "mu of gaussian distribution");
DEFINE_string(init_type, "normal", "initial type normal, 123, 1, 0");
DEFINE_string(nn_param, "200:10", "layer size of nn");
DEFINE_bool(with_bias, false, "with bias");
DEFINE_string(tranfer_func, "sigmoid", "sigmoid, tanh, ReLU");
DEFINE_string(log_dir, "./log", "log dir");



int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  FLAGS_log_dir = "./log";

  LookupTable* lookup_table = new LookupTable();
  if(lookup_table->Init(FLAGS_lookup_table_param, FLAGS_lookup_table_width)) {

  }
  lookup_table->DebugInit();


  NN* nn = new NN();
  nn->Init(lookup_table, FLAGS_nn_param, FLAGS_minibatchsize, FLAGS_init_type, FLAGS_with_bias);
  double* input = new double[lookup_table->GetOutputWidth()*FLAGS_minibatchsize];
  for(int i=0;i<lookup_table->GetOutputWidth()*FLAGS_minibatchsize;i++) {
    input[i] = (double)(i+1);
  }
  nn->Forward(input, 9);
  double* output = nn->GetOutput();
  for(int i=0;i<nn->GetMiniBatchSize();i++) {
    for(int j=0;j<nn->GetOutputSize();j++) {
      int idx = i*nn->GetOutputSize() + j;
//     std::cout<<output[idx]<<" ";
    }
//    std::cout<<std::endl;
  }
  int featuresize;
  int instancenum;
  double* feature = NULL;
  double* target = NULL;
  std::string filename = "data/train.csv";
  LOG(INFO)<<"Begin to read train data...";
  if(!ReadMNIST(filename, feature, target, featuresize, instancenum)) {
    LOG(ERROR)<<"Read MNIST faild...";
  }
  LOG(INFO)<<"Begin to train";

  nn->Train(feature, target, instancenum);

}

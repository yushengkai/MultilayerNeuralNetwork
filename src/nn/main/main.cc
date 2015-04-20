// Copyright (c) 2015 Qihoo 360 Technology Co. Ltd
// Author: Yu ShengKai (yushengkai@360.cn)

#include <cblas.h>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/random.hpp>
#include <glog/logging.h>
#include <gflags/gflags.h>

DEFINE_string(lookup_table_param, "10:20:30", "lengths of lookup tables");
DEFINE_int32(lookup_table_width, 50, "width of all the lookup tables");
DEFINE_int32(minibatchsize, 10, "minibatchsize");
DEFINE_double(sigma, 0.1, "sigma of gaussian distribution");
DEFINE_double(mu, 0, "mu of gaussian distribution");
DEFINE_string(init_type, "normal", "initial type normal, 123, 1, 0");
DEFINE_string(nn_param, "100:50:2", "layer size of nn");
DEFINE_bool(with_bias, false, "with bias");
DEFINE_string(tranfer_func, "sigmoid", "sigmoid, tanh, ReLU");
using namespace boost;

static mt19937 rng(static_cast<unsigned>(std::time(0)));
boost::normal_distribution<double> norm_dist(FLAGS_mu, FLAGS_sigma);
variate_generator<mt19937&, normal_distribution<double> >
normal_sample(rng, norm_dist);

double sigmoid(double x) {
  return 1/(1+exp(-x));
}

double tanha(double x) {
  return tanh(x);
//  return (exp(x) - exp(-x))/(exp(x) + exp(-x));
}

double ReLU(double x) {
  return log(1+exp(x));
}

class LookupTable {
private:
  int table_width;
  int total_length;
  double* central_array;
  std::vector<int> group_sizes;
  std::vector<double*> group_ptrs;
public:
  LookupTable(){}
  bool Init(std::string param, int k);
  double* QueryVector(int groupid, int featureid);
  int GetTableWidth();
  int GetOutputWidth();
  void print_argv();
  void DebugInit();
};

int LookupTable::GetTableWidth() {
  return table_width;
}

int LookupTable::GetOutputWidth() {
  return table_width * group_sizes.size();
}

bool LookupTable::Init(std::string param, int k) {
  std::vector<std::string> parts;
  total_length = 0;
  table_width = k;
  boost::trim(param);
  boost::split(parts, param, boost::is_any_of(":"));
  for(unsigned int i=0;i<parts.size();i++) {
    int value = boost::lexical_cast<int>(parts[i]);
    group_sizes.push_back(value);
    total_length += table_width * value;
  }
  central_array = new double[total_length];

  for(int i=0; i<total_length; i++) {
    central_array[i] = normal_sample();
  }
  int offset=0;
  group_ptrs.push_back(central_array);
  for(unsigned int i=0; i < group_sizes.size()-1; i++) {
    offset += group_sizes[i] * table_width;
    group_ptrs.push_back(central_array + offset);
  }
  return true;
}

void LookupTable::DebugInit() {
  double value = 0;
  int idx=0;
  for(unsigned int i=0;i<group_sizes.size();i++) {
    for(int j=0;j<group_sizes[i] * table_width;j++) {
      central_array[idx] = value;
      idx++;
      value++;
    }
  }
}

double* LookupTable::QueryVector(int groupid, int featureid) {
  if(groupid<0 || groupid >=group_sizes.size()) {
    LOG(ERROR)<<"groupid error: groupid = "<<groupid<<" not in "<<"[0, "
        <<group_sizes.size()<<")";
    return NULL;
  }
  if(featureid<0|| featureid >=group_sizes[groupid]) {
    LOG(ERROR)<<"featureid error: featureid = "<<featureid<<" not in "<<"[0, "
        <<group_sizes[groupid]<<")";
    return NULL;
  }

  return group_ptrs[groupid] + featureid*table_width;
}

void LookupTable::print_argv() {

}

class NN {
 private:
  int inputsize;
  int outputsize;
  int minibatchsize;
  std::vector<int> layersizes;
  std::vector<double*> weight_matrixs;
  std::vector<double*> bias_vectors;
  std::vector<double*> layer_values;
  std::vector<double*> layer_sigmas;
  double* nn_output;
  bool with_bias;
 public:
  NN(){}
  bool Init(LookupTable *lookup_table, std::string param,
            int m, std::string init_type, bool wb);
  bool Forward(double* input);
  void InitWeight(std::string init_type);
  void PrintWeight();
  int GetMiniBatchSize(){return minibatchsize;}
  int GetOutputSize(){return outputsize;}
  double* GetOutput(){return nn_output;}
  void CompareWithTorch();
};

bool NN::Init(LookupTable * lookup_table, std::string param,
              int m, std::string init_type, bool wb) {
  /*
   * m :minibatchsize
   * init_type: normal, 123 ,1, 0
   *  */
  with_bias=wb;
  minibatchsize = m;
  inputsize = lookup_table->GetOutputWidth();
  layersizes.push_back(inputsize);
  boost::trim(param);
  std::vector<std::string> parts;
  boost::split(parts, param, boost::is_any_of(":"));
  std::cout<<"nn layers:"<<inputsize;
  for(unsigned int i=0;i<parts.size();i++) {
    int value = boost::lexical_cast<int>(parts[i]);
    if(value<=0) {
      return false;
    }
    std::cout<<"->"<<value;
    layersizes.push_back(value);
  }
  outputsize = layersizes.back();
  std::cout<<std::endl;
  for(unsigned int i=0;i<layersizes.size();i++) {
    if(i==0) {
      double* layer_value = NULL;
      layer_values.push_back(layer_value);
      double* layer_sigma = NULL;
      layer_sigmas.push_back(layer_sigma);
      continue;
    }

    int layer_value_size = minibatchsize * layersizes[i];
    double* layer_value = new double[layer_value_size];
    layer_values.push_back(layer_value);
    double* layer_sigma = new double[layer_value_size];
    layer_sigmas.push_back(layer_sigma);

    int node_num = layersizes[i];
    int weight_num_of_node = layersizes[i-1];
    int weight_num = weight_num_of_node * node_num;
    double* weight_matrix = new double[weight_num];
    double* bias_vector = new double[node_num];
    weight_matrixs.push_back(weight_matrix);
    bias_vectors.push_back(bias_vector);
  }
  nn_output = layer_values.back();
  InitWeight(init_type);
  return true;
}

void NN::InitWeight(std::string init_type) {
  for(unsigned int i=1;i<layersizes.size();i++) {
    int node_num = layersizes[i];
    int weight_num_of_node = layersizes[i-1];
    int weight_num = node_num * weight_num_of_node;
    double* weight_matrix = weight_matrixs[i-1];
    double* bias_vector = bias_vectors[i-1];
    for(int j=0;j<weight_num;j++) {
      if(init_type=="normal")
      {
        weight_matrix[j] = normal_sample();
      } else if(init_type == "123") {
        weight_matrix[j] = (double)(j+1);
      } else if(init_type == "1") {
        weight_matrix[j] = 1;
      } else if (init_type == "0") {
        weight_matrix[j] = 0;
      } else {
        weight_matrix[j] = normal_sample();
      }
    }
    for(int j=0;j<node_num;j++) {
      if(init_type=="normal")
      {
        bias_vector[j] = normal_sample();
      } else if(init_type == "123") {
        bias_vector[j] = (double)(j+1);
      } else if(init_type == "1") {
        bias_vector[j] = 1;
      } else if (init_type == "0") {
        bias_vector[j] = 0;
      } else {
        bias_vector[j] = normal_sample();
      }
    }
  }
  if(!with_bias) {
    for(unsigned int i=1;i<layersizes.size();i++) {
      int node_num = layersizes[i];
      double* bias_vector = bias_vectors[i-1];
      for(int j=0;j<node_num;j++) {
        bias_vector[j] = 0;
      }
    }
  }
}

void NN::PrintWeight() {
  std::cout<<"weight:"<<std::endl;
  for(unsigned int i=1;i<layersizes.size();i++) {
    int node_num = layersizes[i];
    int weight_num_of_node = layersizes[i-1];
    int weight_num = node_num * weight_num_of_node;
    double* weight_matrix = weight_matrixs[i-1];
    double* bias_vector = bias_vectors[i-1];
    for(int j=0;j<weight_num;j++) {
      std::cout<<weight_matrix[j]<<" ";
    }
    std::cout<<std::endl;
    //  for(int j=0;j<node_num;j++) {
    //    bias_vector[j] = normal_sample();
    //  }
  }
  std::cout<<"bias"<<std::endl;
  for(unsigned int i=1;i<layersizes.size();i++) {
    int node_num = layersizes[i];
    double* bias_vector = bias_vectors[i-1];
    for(int j=0;j<node_num ;j++) {
      std::cout<<bias_vector[j]<<" ";
    }
    std::cout<<std::endl;
  }
}

bool NN::Forward(double* input) {
  layer_values[0] = input;
  /*
   * minibatch
   * minibatchsize * inputsize  = real input size
   *  */

  double* normal_devider = new double[minibatchsize];
  for(int layer=1;layer<layersizes.size();layer++) {

    const enum CBLAS_ORDER Order=CblasRowMajor;
    const enum CBLAS_TRANSPOSE TransX=CblasNoTrans;
    const enum CBLAS_TRANSPOSE TransW=CblasTrans;
    const int M=minibatchsize;//X的行数，O的行数
    const int N=layersizes[layer];//W的列数，O的列数
    const int K=layersizes[layer-1];//X的列数，W的行数
    const double alpha=1;
    const double beta=0;
    const int lda=K;
    const int ldb=K;
    const int ldc=N;
    double* X = layer_values[layer-1];
    double* W= weight_matrixs[layer-1];
    double* sigma = layer_sigmas[layer];
    double* output = layer_values[layer];
    cblas_dgemm(Order, TransX, TransW, M, N, K,
                alpha, X, lda,
                W, ldb,
                beta, sigma, ldc);
    for(int i=0;i<minibatchsize;i++) {
      normal_devider[i]=0;
      for(int j=0;j<layersizes[layer];j++) {
        int idx = i * layersizes[layer] + j;
        if(layer == layersizes.size()-1) {
          output[idx] = exp(sigma[idx]);
          normal_devider[i] += output[idx];
        }else {
          output[idx] = sigmoid(sigma[idx]);
        }
    //    output[idx] = sigma[idx];
      }
    }
  }

  double* final_output = layer_values.back();
  for(int i=0;i<minibatchsize;i++) {
    double sum = 0;
    for(int j=0;j<layersizes.back();j++) {
      int idx = i * layersizes.back() + j;
      final_output[idx] /= normal_devider[i];

    }
  }
}

void NN::CompareWithTorch() {
  std::ifstream weight_fin("data/weight.txt");
  std::ifstream input_fin("data/input.txt");
  std::string line;
  std::vector<std::string> parts;
  //std::cout<<"minibatchsize:"<<minibatchsize<<std::endl;
  double* input = new double[minibatchsize*inputsize];
  for(int i=0;i<minibatchsize;i++) {
    getline(input_fin, line);
    boost::trim(line);
    boost::split(parts, line, boost::is_any_of(" "));
    if(parts.size()!=inputsize) {
      LOG(ERROR)<<"input file error";
    }
    for(int j=0;j<inputsize;j++) {
      int idx = i * inputsize + j;
      input[idx] = boost::lexical_cast<double>(parts[j]);
    }
  }
  for(int layer=1;layer<layersizes.size();layer++) {
 //   boost::trim(line);
//    boost::split(parts, line, boost::is_any_of(" "));
    double* weight = weight_matrixs[layer-1];
    int weight_length = layersizes[layer-1] * layersizes[layer];

//    if(parts.size()!=weight_length) {
 //     LOG(ERROR)<<"weight file error. parts.size() = "<<parts.size()
  //        <<" weight_length="<<weight_length;
   //   return;
   // }
 //   std::cout<<parts.size()<<std::endl;
    for(int j=0;j<weight_length;j++) {
      getline(weight_fin, line);
      weight[j] = boost::lexical_cast<double>(line);
    }
  }
  weight_fin.close();
  input_fin.close();

  Forward(input);
  double* output = GetOutput();
  std::cout<<"Compare With Torch"<<std::endl;
  for(int i=0;i<GetMiniBatchSize();i++) {
    for(int j=0;j<GetOutputSize();j++) {
      int idx = i*GetOutputSize() + j;
      std::cout<<output[idx]<<" ";
    }
    std::cout<<std::endl;
  }

}


int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  FLAGS_log_dir = "./log";
  //LOG(INFO)<<"Hello Glog";
  //LOG(ERROR)<<"glog error";
  int i=0;
  double A[6] = {1.0,2.0,3.0,4.0,5.0,6.0};
  double B[6] = {1.0,2.0,3.0,4.0,5.0,6.0};
  double C[9] = {.5,.5,.5,.5,.5,.5,.5,.5,.5};
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,3,3,2,1,A, 2, B, 2,0,C,3);
  std::cout.precision(15);
  for(i=0; i<3; i++)
  {
    for(int j=0;j<3;j++)
    {
      std::cout<<C[i*3+j]<<" ";
    }
      std::cout<<("\n");
  }

  LookupTable* lookup_table = new LookupTable();
  if(lookup_table->Init(FLAGS_lookup_table_param, FLAGS_lookup_table_width)) {

  }
  lookup_table->DebugInit();
  double* feature;
  if ((feature = lookup_table->QueryVector(2, 9)) == NULL) {
    LOG(ERROR)<<"query vector error ...";
  } else {
 //   std::cout<<"feature queried:"<<std::endl;
    for(int i=0; i < lookup_table->GetTableWidth(); i++) {
 //     std::cout<<feature[i]<<" ";
    }
 //   std::cout<<std::endl;
  }
  std::cout<<"lookup table output size:"<<lookup_table->GetOutputWidth()<<std::endl;
  NN* nn = new NN();
  nn->Init(lookup_table, FLAGS_nn_param, FLAGS_minibatchsize, FLAGS_init_type, FLAGS_with_bias);
  double* input = new double[lookup_table->GetOutputWidth()*FLAGS_minibatchsize];
  for(int i=0;i<lookup_table->GetOutputWidth()*FLAGS_minibatchsize;i++) {
    input[i] = (double)(i+1);
  }
  nn->Forward(input);
  double* output = nn->GetOutput();
  for(int i=0;i<nn->GetMiniBatchSize();i++) {
    for(int j=0;j<nn->GetOutputSize();j++) {
      int idx = i*nn->GetOutputSize() + j;
      std::cout<<output[idx]<<" ";
    }
    std::cout<<std::endl;
  }
  nn->CompareWithTorch();
  //  nn->PrintWeight();
/*
  static mt19937 rng(static_cast<unsigned>(std::time(0)));
  boost::normal_distribution<double> norm_dist(FLAGS_mu, FLAGS_sigma);
  variate_generator<mt19937&, normal_distribution<double> >
        normal_sample(rng, norm_dist);

  for(int i=0;i<100;i++)
  {
    std::cout<<normal_sample()<<std::endl;
  }
*/
}


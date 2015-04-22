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



DEFINE_string(lookup_table_param, "10:20:30:40", "lengths of lookup tables");
DEFINE_int32(lookup_table_width, 1, "width of all the lookup tables");
DEFINE_int32(minibatchsize, 9, "minibatchsize");
DEFINE_double(sigma, 0.1, "sigma of gaussian distribution");
DEFINE_double(mu, 0, "mu of gaussian distribution");
DEFINE_string(init_type, "normal", "initial type normal, 123, 1, 0");
DEFINE_string(nn_param, "5:3", "layer size of nn");
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
  std::vector<double*> delta_matrixs;
  std::vector<double*> error_matrixs;
//  std::vector<double*> layer_sigmas;
  double* nn_output;
  double* logloss;
  double* softmax_sum;
  bool with_bias;
  double learning_rate;
 public:
  NN(){}
  bool Init(LookupTable *lookup_table, std::string param,
            int m, std::string init_type, bool wb);
  bool Forward(double* input);
  bool LogLoss(double* target);
  bool Derivative(double* target);
  void InitWeight(std::string init_type);
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
  learning_rate = 0.001;
  with_bias=wb;
  minibatchsize = m;
  softmax_sum = new double[minibatchsize];
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
      continue;
    }

    int layer_value_size = minibatchsize * layersizes[i];
    double* layer_value = new double[layer_value_size];
    layer_values.push_back(layer_value);

    int node_num = layersizes[i];
    int weight_num_of_node = layersizes[i-1];
    int weight_num = weight_num_of_node * node_num;
    double* weight_matrix = new double[weight_num];
    double* bias_vector = new double[node_num];
    weight_matrixs.push_back(weight_matrix);
    bias_vectors.push_back(bias_vector);

    double* delta = new double[weight_num];
    double* error = new double[layer_value_size];
    delta_matrixs.push_back(delta);
    error_matrixs.push_back(error);
  }
  nn_output = layer_values.back();
  logloss = new double[1];
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


bool NN::Forward(double* input) {
  layer_values[0] = input;
  /*
   * minibatch
   * minibatchsize * inputsize  = real input size
   *  */

  for(int layer=1;layer<layersizes.size();layer++) {

    enum CBLAS_ORDER Order=CblasRowMajor;
    enum CBLAS_TRANSPOSE TransX=CblasNoTrans;
    enum CBLAS_TRANSPOSE TransW=CblasTrans;
    int M=minibatchsize;//X的行数，O的行数
    int N=layersizes[layer];//W的列数，O的列数
    int K=layersizes[layer-1];//X的列数，W的行数
    double alpha=1;
    double beta=0;
    int lda=K;
    int ldb=K;
    int ldc=N;
    double* X = layer_values[layer-1];
    double* W= weight_matrixs[layer-1];
    double* output = layer_values[layer];
    cblas_dgemm(Order, TransX, TransW, M, N, K,
                alpha, X, lda,
                W, ldb,
                beta, output, ldc);
    for(int i=0;i<minibatchsize;i++) {
      for(int j=0;j<layersizes[layer];j++) {
        int idx = i * layersizes[layer] + j;
        if(layer == layersizes.size()-1) {
          output[idx] = exp(output[idx]);
        }else {
          output[idx] = sigmoid(output[idx]);
        }
      }
    }

    if(layer == layersizes.size()-1){
      double ones[3]={1,1,1};
      Order = CblasRowMajor;
      TransX = CblasNoTrans;
      TransW = CblasTrans;
      M=minibatchsize;
      N=1;
      K=layersizes[layer];
      alpha=1;
      beta = 0;
      lda=K;
      ldb=K;
      ldc=N;

      cblas_dgemm(Order, TransX, TransW,
                  M, N, K,
                  alpha, output, lda,
                  ones, ldb,
                  beta, softmax_sum, ldc
                 );

      std::cout<<std::endl;
      for(int i=0;i<minibatchsize;i++) {
        std::cout<<"sample "<<i<<": ";
        for(int j=0;j<layersizes[layer];j++) {
          int idx = i*layersizes[layer]+j;
          output[idx] = output[idx]/softmax_sum[i];
          std::cout<<output[idx]<<" ";
        }
        std::cout<<std::endl;
      }
    }
  }
}

bool NN::Derivative(double* target) {

  for(int layer=layersizes.size()-1;layer>=1;layer--){

    int layersize = layersizes[layer];
    double* factor = error_matrixs[layer-1];
    double* delta = delta_matrixs[layer-1];
    if(layer == layersizes.size()-1) {
      for(int i=0;i<minibatchsize;i++) {
        int t = (int)target[i];
        factor[layersize*i]=0;
        factor[layersize*i+1]=0;
        factor[layersize*i+2]=0;
        factor[layersize*i + t]=1;
      }

      double* output = layer_values.back();//minibatchsize *2
      int N = minibatchsize * layersize;
      double alpha = -1;
      cblas_daxpy(N, alpha, output, 1, factor, 1);

      double* X = layer_values[layer-1];
      int hidelayer_size = layersizes[layer-1];
      int weight_idx = layer-1;
      const int M = layersize;
      N = hidelayer_size;
      const int K = minibatchsize;
      const int lda = M;
      const int ldb = N;
      const int ldc = N;
      cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                M, N, K,
                1.0, factor, lda,
                X, ldb,
                0.0, delta, ldc
                );
//      cblas_daxpy(layersize*hidelayer_size, learning_rate, delta, 1,
//               weight_matrixs[weight_idx], 1
//              );
    } else {

      int weight_idx = layer-1;
      int downstream_layersize = layersizes[layer+1];
      double* downstream_factor = error_matrixs[weight_idx+1];//下游的
      double* downstream_weight = weight_matrixs[weight_idx+1];
      int M = minibatchsize;
      int N = layersize;
      int K = downstream_layersize;
      int lda = K;
      int ldb = K;
      int ldc = N;

      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                  M, N, K,
                  1.0, downstream_factor, lda,
                  downstream_weight, ldb,
                  0.0, factor, ldc
                 );//把下游的误差，通过权值累加到这一层
      double* O = layer_values[layer];
      //std::cout<<"layersize:"<<layersize<<std::endl;
      for(int i=0;i<minibatchsize;i++) {
        for(int j=0;j<layersize;j++) {
          int idx=i*layersize+j;
          factor[idx]=factor[idx]*O[idx]*(1-O[idx]);
        //  std::cout<<factor[idx]<<" ";
        }
      }
      for(int i=0;i<minibatchsize;i++) {
        for(int j=0;j<layersize;j++) {
        //  std::cout<<O[i*layersize+j]<<" ";
        }
        //std::cout<<std::endl;
      }
      double* X = layer_values[layer-1];
      for(int i=0;i<minibatchsize;i++) {
        for(int j=0;j<layersizes[layer-1];j++) {
//          std::cout<<X[i*layersizes[layer-1]+j]<<" ";
        }
//        std::cout<<std::endl;
      }
      int hidelayer_size = layersizes[layer-1];
      M = layersize;
      N = hidelayer_size;
      K = minibatchsize;
      lda = M;
      ldb = M;
      ldc = N;
      cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                  M, N, K,
                  -1.0, factor, lda,
                  X, ldb,
                  0.0, delta, ldc
                  );
    }

    int hidelayer_size=layersizes[layer-1];
    int weight_idx = layer - 1;
    cblas_daxpy(layersize*hidelayer_size, learning_rate, delta, 1,
                weight_matrixs[weight_idx], 1
               );
  }
  return true;
}

void NN::CompareWithTorch() {
  std::ifstream weight_fin("data/weight.txt");
  std::ifstream input_fin("data/input.txt");
  std::ifstream target_fin("data/target.txt");
  std::string line;
  std::vector<std::string> parts;
  //std::cout<<"minibatchsize:"<<minibatchsize<<std::endl;
  double* input = new double[minibatchsize*inputsize];
  double* target = new double[minibatchsize];
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
   for(int i=0;i<minibatchsize;i++) {
    getline(target_fin, line);
    boost::trim(line);
    target[i] = boost::lexical_cast<double>(line);
  }

  std::cout<<"load weight"<<std::endl;
  for(int layer=1;layer<layersizes.size();layer++) {
    double* weight = weight_matrixs[layer-1];
    int weight_length = layersizes[layer-1] * layersizes[layer];
    for(int j=0;j<weight_length;j++) {
      getline(weight_fin, line);
      weight[j] = boost::lexical_cast<double>(line);
    }
  }
  weight_fin.close();
  input_fin.close();
  target_fin.close();
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
//  LogLoss(target);
  for(int i=0;i<10000;i++) {
    Forward(input);
    Derivative(target);
  }
}

bool NN::LogLoss(double* target) {
  //call LogLoss after calling Forward
  int k = 3;
  double* y = new double[k*minibatchsize];
  for(int i=0;i<minibatchsize;i++) {
    y[2*i+(int)target[i]]=1;
    y[2*i+(1-(int)target[i])]=0;
  }
  double* output = layer_values.back();

  const enum CBLAS_ORDER Order = CblasRowMajor;
  const enum CBLAS_TRANSPOSE TransT = CblasNoTrans;
  const enum CBLAS_TRANSPOSE TransO = CblasTrans;
  const int M = 1;
  const int N = 1;
  const int K = minibatchsize * k;
  const double alpha = 1;
  const double beta = 0;
  const int lda = K;
  const int ldb = K;
  const int ldc = N;
  cblas_dgemm(Order, TransT, TransO,
              M, N, K,
              alpha, y, lda,
              output, ldb,
              beta, logloss, ldc
             );
  logloss[0] /=minibatchsize;

  double* sum=new double[minibatchsize];
  double ones_mat[2] = {1,1};
  for(int i=0;i<minibatchsize;i++) {

  }


  return true;
}


int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  FLAGS_log_dir = "./log";
  //LOG(INFO)<<"Hello Glog";
  //LOG(ERROR)<<"glog error";

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
//  std::cout<<"lookup table output size:"<<lookup_table->GetOutputWidth()<<std::endl;
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
//     std::cout<<output[idx]<<" ";
    }
//    std::cout<<std::endl;
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


// Copyright (c) 2015 Qihoo 360 Technology Co. Ltd
// Author: Yu Shengkai (yushengkai@360.cn)

#include <cblas.h>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <time.h>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/random.hpp>
#include <glog/logging.h>

#include "tool/util.h"
#include "gflag/flag.h"
#include "net/nn.h"

static boost::mt19937 rng(static_cast<unsigned>(std::time(0)));
boost::normal_distribution<double> norm_dist1(0, 0.1);
boost::variate_generator<boost::mt19937&, boost::normal_distribution<double> >
normal_sample1(rng, norm_dist1);

bool NN::Init(LookupTable* lt, std::string layer_param, std::string bias_param,
              int m, std::string init_type, bool wb, double l) {
  bias_layer = new Bias_Layer();
  bias_layer->Init(bias_param);
  bias_feature = bias_layer->GetBiasFeature();
  std::cout<<"bias feature:"<<std::endl;
  for(std::vector<int>::iterator iter=bias_feature.begin();iter!=bias_feature.end();iter++) {
    std::cout<<*iter<<std::endl;
  }
  int bias_outputsize = bias_layer->GetOutputSize();
  std::cout<<"bias outputsize:"<<bias_outputsize<<std::endl;
  lookup_table = lt;
  int table_outputsize = lookup_table->GetOutputWidth();
  learning_rate = l;
  with_bias=wb;
  minibatchsize = m;
  tmp_ones = new double[minibatchsize];
  for(int i=0;i<minibatchsize;i++) {
    tmp_ones[i] = 1;
  }
  softmax_sum = new double[minibatchsize];

  inputsize = table_outputsize + bias_outputsize;
  std::cout<<"table outputsize:"<<table_outputsize<<std::endl;
  std::cout<<"bias_outputsize:"<<bias_outputsize<<std::endl;
  layersizes.push_back(inputsize);
  boost::trim(layer_param);
  std::vector<std::string> parts;
  boost::split(parts, layer_param, boost::is_any_of(":"));
  for(unsigned int i=0;i<parts.size();i++) {
    int value = boost::lexical_cast<int>(parts[i]);
    if(value<=0) {
      return false;
    }
    layersizes.push_back(value);
  }
  std::cout<<"NN architecture:"<<std::endl;
  for(unsigned int i=0;i<layersizes.size();i++) {
    std::cout<<layersizes[i]<<"->";
  }
  std::cout<<std::endl;
  outputsize = layersizes.back();
  for(unsigned int i=0;i<layersizes.size();i++) {
    int layer_value_size = minibatchsize * layersizes[i];
    double* layer_value = new double[layer_value_size];
    for(int j=0;j<layer_value_size;j++) {
      layer_value[j] = 0;
    }
    layer_values.push_back(layer_value);

    if(i==0) {
      continue;
    }

    int node_num = layersizes[i];
    int weight_num_of_node = layersizes[i-1] + 1;
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
  delta_x = new double[inputsize*minibatchsize];
  nn_input = layer_values[0];
  double* onehot_startptr = nn_input + table_outputsize;
  std::cout<<"table width:"<<table_outputsize<<std::endl;
  if(bias_layer->SetArray(onehot_startptr)) {
    std::cout<<"init array successful"<<std::endl;
  } else {
    std::cout<<"init array failed"<<std::endl;
  }
  nn_output = layer_values.back();
  InitWeight(init_type);
  std::cout<<"InitWeight "<<init_type<<std::endl;
  return true;
}

void NN::InitWeight(std::string init_type) {

  std::ifstream fin;
  if(init_type == "fromfile") {
    fin.open("data/embedding.weight");
//    std::cout<<"init weight from data/weight.txt"<<std::endl;
  }

  std::string line;
  for(int i=0;i<lookup_table->total_length;i++) {
    if(init_type == "fromfile") {
      getline(fin, line);
      boost::trim(line);
      double value = boost::lexical_cast<double>(line);
      lookup_table->central_array[i] = value;
      //////////////
    }
  }
  for(unsigned int i=1;i<layersizes.size();i++) {
    int node_num = layersizes[i];
    int weight_num_of_node = layersizes[i-1];
    int weight_num = node_num * weight_num_of_node;
    double* weight_matrix = weight_matrixs[i-1];
    double* bias_vector = bias_vectors[i-1];
   for(int j=0;j<weight_num;j++) {
      if(init_type=="normal")
      {
        weight_matrix[j] = normal_sample1();
      } else if(init_type == "123") {
        weight_matrix[j] = (double)(j+1);
      } else if(init_type == "1") {
        weight_matrix[j] = 1;
      } else if (init_type == "0") {
        weight_matrix[j] = 0;
      } else if(init_type == "fromfile") {
        getline(fin, line);
        boost::trim(line);
        weight_matrix[j] = boost::lexical_cast<double>(line);
      } else {
        weight_matrix[j] = normal_sample1();
      }
    }
    for(int j=0;j<node_num;j++) {
      if(init_type=="normal")
      {
        bias_vector[j] = normal_sample1();
      } else if(init_type == "123") {
        bias_vector[j] = (double)(j+1);
      } else if(init_type == "1") {
        bias_vector[j] = 1;
      } else if (init_type == "0") {
        bias_vector[j] = 0;
      } else if(init_type == "fromfile") {
        getline(fin, line);
        boost::trim(line);
        bias_vector[j] = boost::lexical_cast<double>(line);
      } else {
        bias_vector[j] = normal_sample1();
      }
    }
  }
  if(init_type == "fromfile") {
    fin.close();
  }

}
bool NN::LookupFromTable(SparseDataSet* dataset) {
  if(dataset->length>minibatchsize) {
    return false;
  }
  for(int i=0;i<inputsize*minibatchsize;i++) {
    nn_input[i] = 0;
  }
  int** feature = dataset->feature;
  int* width = dataset->width;
  int length = dataset->length;

  int N = lookup_table->GetTableWidth();
  std::map<int, int> group_offset = lookup_table->group_offset;
  for(unsigned int ins=0;ins<length;ins++) {
    for(unsigned int i=0;i<width[ins];i++) {
      int featureid = feature[ins][i];
      int groupid=dataset->groupid[ins][i];

      if(std::find(bias_feature.begin(), bias_feature.end(), groupid) == bias_feature.end()) {
        double* feature = lookup_table->QueryVector(groupid, featureid);
        if(feature == NULL) {
           continue;//如果找不到这个特征，就不找了，这个特征无效
        }
        //int real_featureid = (feature - lookup_table->central_array)/lookup_table->table_width;
        double* group_input = nn_input + inputsize*ins + group_offset[groupid];
        //std::cout<<"group offset:"<<group_offset[groupid]<<"\tgroupid:"<<groupid<<std::endl;
        cblas_daxpy(N, 1,feature,1,group_input,1);
      } else {
        //std::cout<<"groupid:"<<groupid<<std::endl;
        bias_layer->FillOneHot(groupid, featureid);
      //  bias_layer->Print();
      }
    }
  }
  return true;
}

bool NN::SparseForward(SparseDataSet* dataset) {
  if(!LookupFromTable(dataset)) {
    return false;
  }
  Forward(nn_input, dataset->length);
  return true;
}

bool NN::Forward(double* input, int batchsize) {
  layer_values[0] = input;
  /*
   * minibatch
   * minibatchsize * inputsize  = real input size
   *  */
  for(int layer=1;layer<layersizes.size();layer++) {

    enum CBLAS_ORDER Order=CblasRowMajor;
    enum CBLAS_TRANSPOSE TransX=CblasNoTrans;
    enum CBLAS_TRANSPOSE TransW=CblasTrans;
    int M=batchsize;//X的行数，O的行数
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
    for(int i=0;i<batchsize;i++) {
      for(int j=0;j<layersizes[layer];j++) {
        int idx = i * layersizes[layer] + j;
        output[idx] += bias_vectors[layer-1][j];//add bias
        if(layer == layersizes.size()-1) {
          output[idx] = exp(output[idx]);
        }else {
          output[idx] = sigmoid(output[idx]);
        }
      }
    }

    if(layer == layersizes.size()-1){
      Order = CblasRowMajor;
      TransX = CblasNoTrans;
      TransW = CblasTrans;
      M=batchsize;
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
                  tmp_ones, ldb,
                  beta, softmax_sum, ldc
                 );

      for(int i=0;i<batchsize;i++) {
        for(int j=0;j<layersizes[layer];j++) {
          int idx = i*layersizes[layer]+j;
          output[idx] = output[idx]/softmax_sum[i];
        }
      }
    }
  }
}

bool NN::Derivative(SparseDataSet* dataset) {

  double* target = dataset->target;
  int batchsize = dataset->length;
  for(int layer=layersizes.size()-1;layer>=1;layer--){
    int layersize = layersizes[layer];
    double* factor = error_matrixs[layer-1];
    double* delta = delta_matrixs[layer-1];
    double* X = layer_values[layer-1];
    int hidelayer_size = layersizes[layer-1];
    double* delta_bias = delta + layersize*hidelayer_size;
    if(layer == layersizes.size()-1) {
      for(int i=0;i<batchsize;i++) {
        int t = (int)target[i];
        for(int j=0;j<layersize;j++) {
          factor[layersize*i + j] = 0;
        }
        factor[layersize*i + t]=1;
      }

      double* output = layer_values.back();//minibatchsize *2
      int N = batchsize * layersize;
      double alpha = -1;
      cblas_daxpy(N, alpha, output, 1, factor, 1);
      int weight_idx = layer-1;
      int M = layersize;
      N = hidelayer_size;
      int K = batchsize;
      int lda = M;
      int ldb = N;
      int ldc = N;
      cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                M, N, K,
                -1.0/batchsize, factor, lda,
                X, ldb,
                0.0, delta, ldc
                );
      M = layersize;
      N = 1;
      K = batchsize;
      lda = M;
      ldb = N;
      ldc = N;
      cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                  M, N, K,
                  -1.0/batchsize, factor, lda,
                  tmp_ones, ldb,
                  0.0, delta_bias,ldc
                 );
    } else if(layer>=1) {

      int weight_idx = layer-1;
      int downstream_layersize = layersizes[layer+1];
      double* downstream_factor = error_matrixs[weight_idx+1];//下游的
      double* downstream_weight = weight_matrixs[weight_idx+1];
      int M = batchsize;
      int N = layersize;
      int K = downstream_layersize;
      int lda = K;
      int ldb = N;//之前这里是N
      int ldc = N;

      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,//之前这里是trans， notrans
                  M, N, K,
                  1.0, downstream_factor, lda,
                  downstream_weight, ldb,
                  0.0, factor, ldc
                 );//把下游的误差，通过权值累加到这一层

      double* O = layer_values[layer];
      for(int i=0;i<batchsize;i++) {
        for(int j=0;j<layersize;j++) {
          int idx=i*layersize+j;
          factor[idx]=factor[idx]*O[idx]*(1-O[idx]);
          //计算sigmoid层的梯度
        }
      }

      M = layersize;
      N = hidelayer_size;
      K = batchsize;
      lda = M;
      ldb = N;
      ldc = N;

      cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                  M, N, K,
                  -1.0/batchsize, factor, lda,
                  X, ldb,
                  0.0, delta, ldc
                 );

      M = layersize;
      N = 1;
      K = batchsize;
      lda = M;
      ldb = N;
      ldc = N;
      cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                  M, N, K,
                  -1.0/batchsize, factor, lda,
                  tmp_ones, ldb,
                  0.0, delta_bias, ldc
                 );
      if(layer == 1) {
        M = batchsize;
        N = hidelayer_size;
        K = layersize;
        lda = K;
        ldb = N;
        ldc = N;
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K,
                    -1.0/batchsize, factor, lda,
                    weight_matrixs[0], ldb,
                    0.0, delta_x, ldc
                   );
      }
    }
  }


  for(int layer=layersizes.size()-1;layer>=1;layer--){
    int layersize = layersizes[layer];
    int hidelayer_size = layersizes[layer-1];
    double* delta = delta_matrixs[layer-1];
    double* delta_bias = delta + layersize*hidelayer_size;
    int weight_idx = layer - 1;
    cblas_daxpy(layersize*hidelayer_size, -learning_rate, delta, 1,
               weight_matrixs[weight_idx], 1
               );
    cblas_daxpy(layersize, -learning_rate, delta_bias, 1,
                bias_vectors[weight_idx], 1
                );
  }

  int* width = dataset->width;
  int** instances = dataset->feature;
  int table_width = lookup_table->table_width;
  for(std::map<int,double*>::iterator iter=embedding_delta.begin();
      iter!=embedding_delta.end();iter++) {
    delete [] iter->second;
  }
  embedding_delta.clear();
  double* delta_sums = delta_x;
  for(int i=0;i<batchsize;i++) {
    double* delta_sum = delta_sums + i*inputsize;
    for(int j=0;j<width[i];j++) {
      int groupid = dataset->groupid[i][j];
      if(std::find(bias_feature.begin(), bias_feature.end(), groupid) != bias_feature.end()) {
        continue;
      }

      double* delta_group = delta_sum + lookup_table->group_offset[groupid];
      //std::cout<<"groupid:"<<groupid<<"\tgroup_offset:"
      //    <<lookup_table->group_offset[groupid]<<std::endl;
      int featureid = instances[i][j];
      double* feature = lookup_table->QueryVector(groupid, featureid);
      int real_featureid = (feature - lookup_table->central_array)/table_width;
      if(feature == NULL) {
        continue;
      }
      std::map<int,double*>::iterator iter = embedding_delta.find(real_featureid);
      if(iter != embedding_delta.end()) {
        double* feature_delta = embedding_delta[real_featureid];
        cblas_daxpy(table_width, 1.0, delta_group, 1, feature_delta, 1);
      } else {
        double* feature_delta = new double[table_width];
        cblas_dcopy(table_width, delta_group, 1, feature_delta, 1);
        embedding_delta[real_featureid] = feature_delta;
      }
    }
    //std::cout<<std::endl;
  }
  for(std::map<int, double*>::iterator iter=embedding_delta.begin();
      iter!=embedding_delta.end();iter++) {
    double* delta = iter->second;
    int real_featureid = iter->first;
    double* feature_ptr = lookup_table->central_array + real_featureid*table_width;

    cblas_daxpy(table_width, -learning_rate, delta, 1, feature_ptr, 1);
  }
  return true;

}

bool NN::Train(SparseDataSet* trainData, SparseDataSet* testData) {
  int epoch = 0;
  int** feature = trainData->feature;
  int** groupid = trainData->groupid;
  double* target = trainData->target;
  int instancenum = trainData->length;
  while(true) {
    clock_t start, finish;
    double total_time;
    start = clock();
    std::cout<<"epoch:"<<epoch++<<std::endl;
    for(int i=0;i<instancenum;i+=minibatchsize) {
      if(i%1000==0) {
        std::cout<<"\r"<<i<<"/"<<instancenum<<"      batchsize:"<<minibatchsize
            <<"\ttable_width:"<<lookup_table->table_width;
        std::cout.flush();
      }

      int batchsize =
          i+minibatchsize<instancenum ? minibatchsize : instancenum - i;
      double* target_start_ptr = target + i;
      SparseDataSet* dataset = new SparseDataSet();
      dataset->length = batchsize;
      dataset->feature = feature + i;
      dataset->groupid = groupid + i;
      dataset->target = target + i;
      dataset->width = trainData->width + i;
      std::string position_bucket="";
//      for(int i=0;i<dataset->length;i++) {
//        position_bucket = PositionBucket(dataset->feature[i]);
//        std::cout<<"position bucket:"<<position_bucket<<std::endl;
//      }
      SparseForward(dataset);
      Derivative(dataset);
      delete dataset;
    }
    finish = clock();
    double logloss, auc;
    std::cerr<<"\nbegin to compute Auc and loss"<<std::endl;
    AUCLogLoss(testData, auc, logloss);

    total_time = (double)(finish - start)/CLOCKS_PER_SEC;
    std::cout<<"LogLoss:"<<logloss<<"\nAUC:"<<auc<<std::endl;
    std::cout<<"total time:"<<total_time<<std::endl<<std::endl;
  }
  delete []  lookup_table->central_array;
  return true;
}

void NN::CompareWithTorch() {
  int test_size = 20;
  std::string line;
  std::vector<std::string> parts;
  //std::cout<<"minibatchsize:"<<minibatchsize<<std::endl;
  std::vector<std::vector<int> > inputs;
  std::vector<int> targets;
  std::string filename = "data/sparse_unittest.dat.tmp";
  std::string binaryname = "data/tmp.txt";
  SparseDataSet* unittest_dataset = new SparseDataSet();
  ReadSparseData(filename, binaryname, unittest_dataset);
  unittest_dataset->length = 11;
  
  InitWeight("fromfile");
  SparseForward(unittest_dataset);
  Derivative(unittest_dataset);
}

std::string NN::PositionBucket(int* featureid) {
  std::string bucket;
  std::string adtype = std::to_string(featureid[0]);
  std::string has_img = std::to_string(featureid[1]);
  std::string has_link = std::to_string(featureid[2]);
  std::string adpos = std::to_string(featureid[3]);
  bucket = adtype+"_"+has_img+"_"+has_link+"_"+adpos;
  return bucket;
}



bool NN::AUCLogLoss(SparseDataSet* dataset,double& auc, double &logloss) {
  int instancenum = dataset->length;
  logloss = 0;
  double* logloss_tmp = new double[1];
  int kAucBucketRound = 10000;
  double* target = dataset->target;
  double* outputs = new double[instancenum];
  std::map<std::string, double*> pro_bucket_map;
  std::map<std::string, int> pro_bucket_count;
//  double* pro_bucket = new double[2*kAucBucketRound];
//  for(int i=0;i<2*kAucBucketRound;i++) {
//    pro_bucket[i] = 0;
//  }
  int** confuse_matrix = new int*[2];
  for(int i=0;i<2;i++) {
    confuse_matrix[i] = new int[2];
    for(int j=0;j<2;j++) {
      confuse_matrix[i][j]=0;
    }
  }
  int target_idx=0;
  std::ofstream fout("data/output.txt");
  for(int i=0;i<instancenum;i+=minibatchsize) {
    int batchsize =
        i+minibatchsize<instancenum ? minibatchsize : instancenum - i;
    if(i%1000==0) {
      std::cout<<"\r"<<i<<"/"<<instancenum<<"      batchsize:"<<minibatchsize;
      std::cout.flush();
    }
    SparseDataSet* batchdataset = new SparseDataSet();
    batchdataset->length = batchsize;
    batchdataset->feature = dataset->feature + i;
    batchdataset->groupid = dataset->groupid + i;
    batchdataset->width = dataset->width + i;
    double* target_start_ptr = dataset->target + i;
    SparseForward(batchdataset);
    double* y  = layer_values.back();
    cblas_dcopy(batchsize, y+1, 2, outputs+i, 1);

    for(int j=0;j<batchsize;j++) {
      for(int k=0;k<outputsize;k++) {
        fout<<y[j*outputsize+k]<<" ";
      }
      int pred=0;
      if(y[j*outputsize+1]>0.5){
        pred = 1;
      }
      int t=target[target_idx++];
      fout<<t<<std::endl;
      confuse_matrix[t][pred]++;
    }
    fout<<std::endl;

    for(int b=0;b<batchsize;b++) {
      for(int j=0;j<outputsize;j++) {
        y[b*outputsize+j] = log(y[b*outputsize+j]);
      }
    }
    double* factor = error_matrixs.back();//借用一下这个空间
    for(int i=0;i<batchsize;i++) {
      int t = (int)target_start_ptr[i];
      for(int j=0;j<outputsize;j++) {
        factor[outputsize*i + j] = 0;
      }
      factor[outputsize*i + t]=1;
    }


    const enum CBLAS_ORDER Order = CblasRowMajor;
    const enum CBLAS_TRANSPOSE TransT = CblasNoTrans;
    const enum CBLAS_TRANSPOSE TransO = CblasTrans;
    const int M = 1;
    const int N = 1;
    const int K = batchsize * outputsize;
    const double alpha = 1;
    const double beta = 0;
    const int lda = K;
    const int ldb = K;
    const int ldc = N;
    cblas_dgemm(Order, TransT, TransO,
                M, N, K,
                alpha, factor, lda,
                y, ldb,
                beta, logloss_tmp, ldc
               );
    logloss+=logloss_tmp[0];
  }
  double maxvalue = -999;
  double minvalue = 999;
  for(int i=0;i<instancenum;i++) {
    if(maxvalue<outputs[i]) {
      maxvalue = outputs[i];
    }
    if(minvalue>outputs[i]) {
      minvalue = outputs[i];
    }
  }
  fout.close();
  std::cout<<"\nconfuse matrix:"<<std::endl;
  for(int i=0;i<2;i++) {
    for(int j=0;j<2;j++) {
      std::cout<<confuse_matrix[i][j]<<"\t";
    }
    std::cout<<std::endl;
  }

  double tmp = maxvalue-minvalue;
  tmp=kAucBucketRound/tmp;
  std::string position_bucket;
  double* global_pro_bucket = new double[2*kAucBucketRound];
  for(int i=0;i<2*kAucBucketRound;i++) {
    global_pro_bucket[i] = 0;
  }
  for(int i=0;i<instancenum;i++) {
    position_bucket = PositionBucket(dataset->feature[i]);
 //   std::cout<<"position_bucket:"<<position_bucket<<std::endl;
    double* pro_bucket;
    if(pro_bucket_map.find(position_bucket) == pro_bucket_map.end()) {
      pro_bucket = new double[2*kAucBucketRound];
      for(int k=0;k<2*kAucBucketRound;k++) {
        pro_bucket[k] = 0;
      }
      pro_bucket_map[position_bucket] = pro_bucket;
      pro_bucket_count[position_bucket] = 0;
      std::cout<<position_bucket<<" prosition bucket not in pro_bucket_map ..."<<std::endl;
    } else {
      pro_bucket = pro_bucket_map[position_bucket];
    }
    outputs[i]-=minvalue;
    outputs[i]*=tmp;
    outputs[i] = ceil(outputs[i]);
    int idx = (int)outputs[i];
    pro_bucket[idx]++;
    global_pro_bucket[idx]++;
    pro_bucket_count[position_bucket]++;
    if(target[i]==1) {
      pro_bucket[idx+kAucBucketRound]++;
      global_pro_bucket[idx + kAucBucketRound]++;
    }
  }
  pro_bucket_map["global"] = global_pro_bucket;

  for(std::map<std::string, double*>::iterator iter=pro_bucket_map.begin();
      iter!=pro_bucket_map.end();iter++) {
    double* pro_bucket = iter->second;
    std::string position_bucket = iter->first;
    double auc_temp=0;
    double no_click=0;
    double click_sum=0;
    double no_click_sum=0;
    double old_click_sum=0;

    for(int i=2*kAucBucketRound-1;i>=kAucBucketRound;i--) {
      if(pro_bucket[i]!=0) {
        double num_impression = pro_bucket[i - kAucBucketRound];
        double num_click = pro_bucket[i];
        auc_temp += (click_sum+old_click_sum)*no_click/2.0;
        old_click_sum = click_sum;
        no_click = 0;
        no_click += num_impression-num_click;
        no_click_sum = no_click_sum + num_impression - num_click;
        click_sum=click_sum + num_click;
      }
    }
    auc_temp = auc_temp+(click_sum + old_click_sum)*no_click/2.0;
    auc = auc_temp/(click_sum*no_click_sum);
    std::cout<<"instance num:"<<pro_bucket_count[position_bucket]<<"\t"
        <<position_bucket<<" AUC:"<<auc<<std::endl;
    delete [] pro_bucket;
  }
  delete [] outputs;

  std::cout<<"instancenum:"<<instancenum<<std::endl;
  logloss/=instancenum;
  logloss = -logloss;
  delete [] logloss_tmp;
  return true;
}

NN::~NN() {
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


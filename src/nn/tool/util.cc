// Copyright (c) 2015 Qihoo 360 Technology Co. Ltd
// Author: Yu Shengkai (yushengkai@360.cn)

#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include "tool/util.h"

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

bool ReadMNIST(std::string filename, DataSet* dataset) {
  double* feature;
  double* target;
  int featuresize;
  int instancenum;
  int count = 0;
  std::ifstream fin(filename.c_str());
  if (!fin){
    return false;
  }
  featuresize = 784;
  std::string line;
  getline(fin, line);
  while(getline(fin ,line)) {
    count++;
  }
  instancenum = count;
  feature = new double[count* 784];
  target = new double[count];
  fin.clear();
  fin.seekg(0);
  getline(fin, line);
  std::vector<std::string> parts;
  int idx=0;
  while(getline(fin, line)) {
    boost::trim(line);
    boost::split(parts, line, boost::is_any_of(","));
    target[idx] = boost::lexical_cast<double>(parts[0]);
    for(int i=0;i<784;i++) {
      feature[idx*784+i] = (double)boost::lexical_cast<double>(parts[i]);
    }
    idx++;

  }
  fin.close();
  dataset->feature = feature;
  dataset->target = target;
  dataset->length = instancenum;
  dataset->width = featuresize;
  return true;
}

bool ReadSparseData(std::string filename, SparseDataSet* dataset) {
  double** feature;
  double* target;
  int* width;
  int length;
  int max_feature_id = 0;
  int min_feature_id = 999999999;
  std::ifstream fin(filename.c_str());
  std::string line;
  std::vector<std::string> parts;
  int count = 0;
  std::cout<<"begin to count dataset"<<std::endl;
  while(getline(fin, line)) {
    count++;
  }
  fin.clear();
  fin.seekg(0);
  feature = new double*[count];
  target = new double[count];
  width = new int[count];
  int idx = 0;
  std::cout<<"datasize:"<<count<<std::endl;
  while(getline(fin, line)) {
    if(idx%1000==0) {
      std::cout<<idx<<"/"<<count<<"\r";
      std::cout.flush();
    }
    boost::trim(line);
    boost::split(parts, line, boost::is_any_of(" "));
    target[idx] = boost::lexical_cast<double>(parts[0]);
    width[idx] = parts.size() - 1;
    feature[idx] = new double[width[idx]];
    for(unsigned int i=1;i<parts.size();i++) {
      boost::trim(parts[i]);
      feature[idx][i-1] =
          boost::lexical_cast<double>(parts[i].substr(0, parts[i].size()-2));
      if(feature[idx][i] > max_feature_id) {
        max_feature_id = feature[idx][i];
      }
      if(feature[idx][i-1] < min_feature_id) {
        min_feature_id = feature[idx][i];
      }
    }
    idx++;
  }
  fin.close();
  dataset->feature = feature;
  dataset->target = target;
  dataset->width = width;
  dataset->length = count;
  dataset->max_featureid = max_feature_id;
}




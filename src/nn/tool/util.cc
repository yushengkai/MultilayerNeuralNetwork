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
    //std::cout<<line<<std::endl;
    boost::trim(line);
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




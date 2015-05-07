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

bool ReadSparseData(std::string filename, std::string binaryname, SparseDataSet* dataset) {
  int** feature;
  int** groupid;
  double* target;
  int* width;
  int length;
  int max_feature_id = 0;
  int min_feature_id = 999999999;
  std::ifstream fin(filename.c_str());
  std::string line;
  std::vector<std::string> parts;
  int count = 0;
  while(getline(fin, line)) {
    count++;
  }
  fin.clear();
  fin.seekg(0);
  feature = new int*[count];
  groupid = new int*[count];
  target = new double[count];
  width = new int[count];
  int idx = 0;
  std::ofstream fout(binaryname.c_str(), std::ios::binary);
  fout.write((char*)&count, sizeof(int));
  fout.write((char*)&max_feature_id, sizeof(int));
  while(getline(fin, line)) {
    if(idx%1000==0) {
      std::cout<<idx<<"/"<<count<<"\r";
      std::cout.flush();
    }
    boost::trim(line);
    boost::split(parts, line, boost::is_any_of(" "));
    target[idx] = boost::lexical_cast<double>(parts[0]);
    width[idx] = parts.size() - 1;
    feature[idx] = new int[width[idx]];
    groupid[idx] = new int[width[idx]];
    fout.write((char*)&target[idx],sizeof(double));
    int size=parts.size()-1;
    fout.write((char*)&size, sizeof(int));
    for(unsigned int i=1;i<parts.size();i++) {
      boost::trim(parts[i]);
      feature[idx][i-1] =
          boost::lexical_cast<double>(parts[i].substr(0, parts[i].size()-2));
      groupid[idx][i-1] = i-1<20 ? 0 : 1;
      if(feature[idx][i-1] > max_feature_id) {
        max_feature_id = feature[idx][i-1];
      }
      if(feature[idx][i-1] < min_feature_id) {
        min_feature_id = feature[idx][i-1];
      }
    }
    fout.write((char*)feature[idx], sizeof(int)*(parts.size()-1));
    fout.write((char*)groupid[idx], sizeof(int)*(parts.size()-1));
    idx++;
  }
  fout.seekp(sizeof(int));
  fout.write((char*)&max_feature_id, sizeof(int));
  fin.close();
  fout.close();
  dataset->feature = feature;
  dataset->target = target;
  dataset->groupid = groupid;
  dataset->width = width;
  dataset->length = count;
  dataset->max_featureid = max_feature_id;
  return true;
}

bool ReadSparseDataFromBin(std::string binaryname, SparseDataSet* dataset) {
  int** feature;
  int** groupid;
  double* target;
  int* width;
  int length;
  int max_feature_id;
  std::ifstream fin(binaryname.c_str(), std::ios::binary);
  fin.read((char*)&length, sizeof(int));
  std::cout<<"Instance num:"<<length<<std::endl;
  fin.read((char*)&max_feature_id, sizeof(int));
  std::cout<<"Max feature id:"<<max_feature_id<<std::endl;
  feature = new int*[length];
  groupid = new int*[length];
  target = new double[length];
  width = new int[length];
  int tmp_max=0;
  int tmp_min=999999;
  for(int i=0;i<length;i++) {
    fin.read((char*)(target+i), sizeof(double));
    //std::cout<<"target "<<i<<" "<<target[i]<<std::endl;
    fin.read((char*)(width+i),sizeof(int));
    feature[i] = new int[width[i]];
    groupid[i] = new int[width[i]];
    fin.read((char*)feature[i], sizeof(int)*width[i]);
    fin.read((char*)groupid[i], sizeof(int)*width[i]);

    for(int j=0;j<width[i];j++) {
      if(feature[i][j]>tmp_max) {
        tmp_max = feature[i][j];
      }
      if(feature[i][j]<tmp_min) {
        tmp_min = feature[i][j];
      }
    }
  }
  fin.close();
  std::cout<<"max feature id:"<<tmp_max<<std::endl;
  std::cout<<"min feature id:"<<tmp_min<<std::endl;
  dataset->feature = feature;
  dataset->target = target;
  dataset->groupid = groupid;
  dataset->width = width;
  dataset->max_featureid = max_feature_id;
  dataset->length = length;
  return true;
}

bool DeleteSparseData(SparseDataSet* dataset) {
  for(int i=0;i<dataset->length;i++) {
    delete [] dataset->groupid[i];
    delete [] dataset->feature[i];
  }
  delete [] dataset->width;
  delete [] dataset->feature;
  delete [] dataset->target;
  delete dataset;
}


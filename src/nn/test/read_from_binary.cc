// Copyright (c) 2015 Qihoo 360 Technology Co. Ltd
// Author: Yu Shengkai (yushengkai@360.cn)

#include "from_binary/dataset.h"
#include <string>
#include <fstream>
#include <iostream>
int main() {
  Problem problem;
  std::string filename = "/data/yushengkai/sparse_dnn/binary_file";
  std::ifstream fin(filename.c_str());
  problem.read_from_binary(fin);
  std::cout<<"total length:"<<problem.l<<std::endl;
  fin.close();
  filename = "/data/yushengkai/sparse_dnn/binary1";
  fin.open(filename.c_str());
  problem.read_from_binary(fin);
  std::cout<<"total length:"<<problem.l<<std::endl;
  fin.close();

}


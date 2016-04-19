// Copyright (c) 2015 Qihoo 360 Technology Co. Ltd
// Author: Yu Shengkai (yushengkai@360.cn)


#ifndef LOOKUPTABLE_H_
#define LOOKUPTABLE_H_

#include <string>
#include <vector>
#include <map>
#include <fstream>

class LookupTable {
 public:
  int table_width;
  std::map<int, int> group_sizes;
  std::map<int, double*> group_ptrs;
  std::map<int, int> group_offset;

  int total_length;
  int feature_length;
  double* central_array;
  double* delta_lookuptable;
  LookupTable(){}
  ~LookupTable(){
  };
  bool Init(std::string param, std::map<int, int> term_feature, int k);
  double* QueryVector(int groupid, int featureid);
  int GetTableWidth();
  int GetOutputWidth();
  int GroupId(int real_featureid);
  void print_argv();
  void DebugInit();
  void InitFromStream(std::ifstream fin);
};



#endif  // LOOKUPTABLE_H_


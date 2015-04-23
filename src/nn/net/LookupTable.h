// Copyright (c) 2015 Qihoo 360 Technology Co. Ltd
// Author: Yu Shengkai (yushengkai@360.cn)


#ifndef LOOKUPTABLE_H_
#define LOOKUPTABLE_H_

#include <string>
#include <vector>

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



#endif  // LOOKUPTABLE_H_


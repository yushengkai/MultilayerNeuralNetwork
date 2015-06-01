// Copyright (c) 2015 Qihoo 360 Technology Co. Ltd
// Author: Yu Shengkai (yushengkai@360.cn)


#ifndef BIAS_LAYER_H_
#define BIAS_LAYER_H_

#include <iostream>
#include <vector>

class Bias_Layer {
 private:
  double* onehot_array;
  int total_width;
  std::vector<int> groupid_vec;
  std::vector<int> group_width_vec;
 public:
  bool Init(std::string bias_param);
  bool FillOneHot(int groupid, int featureid);
  bool ZeroArray();
  int GetOutputSize();
  bool SetArray(double* array);
  std::vector<int> GetBiasFeature();
  void Print();
};

#endif  // BIAS_LAYER_H_


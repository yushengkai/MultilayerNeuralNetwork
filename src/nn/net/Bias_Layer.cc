// Copyright (c) 2015 Qihoo 360 Technology Co. Ltd
// Author: Yu Shengkai (yushengkai@360.cn)

#include "net/Bias_Layer.h"
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <vector>

bool Bias_Layer::Init(std::string bias_param) {
  boost::trim(bias_param);
  std::vector<std::string> parts, pieces;
  this->total_width = 0;
  boost::split(parts, bias_param, boost::is_any_of(","));
  for(unsigned int i=0;i<parts.size();i++) {
    boost::trim(parts[i]);
    if(parts[i]=="") {
      continue;
    }
    boost::split(pieces, parts[i], boost::is_any_of(":"));
    int groupid = boost::lexical_cast<int>(pieces[0]);
    int group_width = boost::lexical_cast<int>(pieces[1])+1;
    groupid_vec.push_back(groupid);
    group_width_vec.push_back(group_width);
    total_width += group_width;
  }
  return true;
}

bool Bias_Layer::FillOneHot(int groupid, int featureid) {
  int start_idx = 0;
  int end_idx = group_width_vec[0];
  for(unsigned int i=0;i<groupid_vec.size();i++) {
    if(groupid == groupid_vec[i]) {
      for(int j=start_idx;j<end_idx;j++) {
        onehot_array[j] = 0;
      }
      int idx = start_idx + featureid;
      onehot_array[idx] = 1;
      break;
    }
    start_idx = end_idx;
    if(i < groupid_vec.size()-1){
      end_idx = end_idx + group_width_vec[i+1];
    }
  }

  return true;
}

bool Bias_Layer::ZeroArray() {
  for(int i=0;i<total_width;i++) {
    onehot_array[i] = 0;
  }
}

int Bias_Layer::GetOutputSize() {
  return total_width;
}


bool Bias_Layer::SetArray(double* array) {
  onehot_array = array;
  if(onehot_array + total_width) {
    return true;
  } else {
    return false;
  }
}

std::vector<int> Bias_Layer::GetBiasFeature() {
  return groupid_vec;
}

void Bias_Layer::Print() {
  for(int i=0;i<total_width;i++) {
    std::cout<<onehot_array[i]<<" ";
  }
  std::cout<<std::endl;
}


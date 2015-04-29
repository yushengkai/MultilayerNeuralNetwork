// Copyright (c) 2015 Qihoo 360 Technology Co. Ltd
// Author: Yu Shengkai (yushengkai@360.cn)a

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/random.hpp>
//#include <glog/logging.h>

#include "tool/util.h"
#include "net/LookupTable.h"


static boost::mt19937 rng(static_cast<unsigned>(std::time(0)));
boost::normal_distribution<double> norm_dist(0, 0.01);
boost::variate_generator<boost::mt19937&, boost::normal_distribution<double> >
normal_sample(rng, norm_dist);

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
 //   LOG(ERROR)<<"groupid error: groupid = "<<groupid<<" not in "<<"[0, "
 //       <<group_sizes.size()<<")";
    return NULL;
  }
  if(featureid<0|| featureid >=group_sizes[groupid]) {
  //  LOG(ERROR)<<"featureid error: featureid = "<<featureid<<" not in "<<"[0, "
 //       <<group_sizes[groupid]<<")";
    return NULL;
  }

  return group_ptrs[groupid] + featureid*table_width;
}

void LookupTable::print_argv() {

}



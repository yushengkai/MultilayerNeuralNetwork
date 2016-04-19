// Copyright (c) 2015 Qihoo 360 Technology Co. Ltd
// Author: Yu Shengkai (yushengkai@360.cn)a

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/random.hpp>
//#include <glog/logging.h>

#include "tool/util.h"
#include "net/lookup_table.h"


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

bool LookupTable::Init(std::string param, std::map<int, int> term_feature_map, int k) {
  std::vector<std::string> parts, pieces;
  total_length = 0;
  table_width = k;
  boost::trim(param);
  boost::split(parts, param, boost::is_any_of(","));
  int offset = 0;
  for(unsigned int i=0;i<parts.size();i++) {
    boost::trim(parts[i]);
    if(parts[i] == "") {
      continue;
    }
    boost::split(pieces, parts[i], boost::is_any_of(":"));
    int groupid = boost::lexical_cast<int>(pieces[0]);
    int value = boost::lexical_cast<int>(pieces[1]);
    value = value+1;
    group_sizes[groupid] = value;
    total_length += table_width * value;
    group_offset[groupid] = offset;
    offset+=k;
  }
  feature_length = total_length/table_width;
  central_array = new double[total_length];
  delta_lookuptable = new double[total_length];
  for(int i=0; i<total_length; i++) {
    central_array[i] = normal_sample();
  }
  std::map<int, int>::iterator iter = group_sizes.begin();
  int groupid = iter->first;
  int tmp = iter->second;
  offset=0;
  for(std::map<int,int>::iterator iter=group_sizes.begin();iter!=group_sizes.end();iter++) {
    int groupid = iter->first;
    group_ptrs[groupid] = central_array + offset;
    offset += tmp*table_width;
    tmp = iter->second;
  }
  for(std::map<int, int>::iterator iter=term_feature_map.begin();
      iter!=term_feature_map.end();iter++) {
    int groupId = iter->first;
    int groupId_for_convert = iter->second;
    group_ptrs[groupId] = group_ptrs[groupId_for_convert];
    std::cout<<"Copy group ptrs "<<groupId<<" to "<<groupId_for_convert<<std::endl;
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

void LookupTable::InitFromStream(std::ifstream fin) {
  std::string line;
  for(int i=0;i<total_length;i++) {
    getline(fin, line);
    boost::trim(line);
  }
}

double* LookupTable::QueryVector(int groupid, int featureid) {
  if(groupid<0 || groupid >=group_sizes.size()) {
//    std::cerr<<"groupid = "<<groupid<<" >= group_sizes.size() = "<<group_sizes.size()<<std::endl;
    return NULL;
  }
  if(featureid<0|| featureid >=group_sizes[groupid]) {
//    std::cerr<<"featureid = "<<featureid<<" >= group_sizes["<<groupid<<"] = "
//        <<group_sizes[groupid]<<std::endl;
    return NULL;
  }

  return group_ptrs[groupid] + featureid*table_width;
}

void LookupTable::print_argv() {

}

int LookupTable::GroupId(int real_featureid) {
  int sum=0;
  for(unsigned int i=0;i<group_sizes.size();i++) {
    if(sum<real_featureid && real_featureid < sum+group_sizes[i]) {
      return i;
    }
    sum+=group_sizes[i];
  }
  return -1;
}

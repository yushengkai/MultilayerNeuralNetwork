// Copyright (c) 2015 Qihoo 360 Technology Co. Ltd
// Author: Yu Shengkai (yushengkai@360.cn)

#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <dirent.h>
#include <stdlib.h>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include "tool/util.h"
#include "gflag/flag.h"
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
  std::vector<std::string> parts, pieces;
  int count = 0;
  while(getline(fin, line)) {
    count++;
  }
  fin.clear();
  fin.seekg(0);
  std::cout<<"count:"<<count<<std::endl;
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
    target[idx] = boost::lexical_cast<double>(parts[1]);
    width[idx] = parts.size() - 2;
    feature[idx] = new int[width[idx]];
    groupid[idx] = new int[width[idx]];
    fout.write((char*)&target[idx],sizeof(double));
    int size=parts.size()-2;
    fout.write((char*)&size, sizeof(int));

    for(unsigned int i=2;i<parts.size();i++) {
      boost::trim(parts[i]);
      boost::split(pieces, parts[i], boost::is_any_of(":"));
      feature[idx][i-2] =
          boost::lexical_cast<double>(pieces[1]);
      //std::cout<<"feature id:"<<feature[idx]
      groupid[idx][i-2] = boost::lexical_cast<int>(pieces[0]);
      if(feature[idx][i-2] > max_feature_id) {
        max_feature_id = feature[idx][i-2];
      }
      if(feature[idx][i-2] < min_feature_id) {
        min_feature_id = feature[idx][i-2];
      }
    }
    fout.write((char*)feature[idx], sizeof(int)*(parts.size()-2));
    fout.write((char*)groupid[idx], sizeof(int)*(parts.size()-2));
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

int max_groupid = 28;
bool ReadSparseDataFromBinFolder(std::string binaryfolder,
                                 std::string bias_feature,
                                 std::string term_feature,
                                 std::map<int, int>& term_feature_map,
                                 SparseDataSet* dataset) {
  DIR* dir = opendir(binaryfolder.c_str());
  struct dirent *dirp;
  char* filename;
  int total_instance_num = 0;
  std::vector<std::string> parts, pieces;
  std::vector<int> bias_feature_vec;
  boost::split(parts, bias_feature, boost::is_any_of(","));
  for(unsigned int i=0;i<parts.size();i++) {
    boost::trim(parts[i]);
    int value = boost::lexical_cast<int>(parts[i]);
    bias_feature_vec.push_back(value);
  }
  //不进行embedding的特征
  boost::split(parts, term_feature, boost::is_any_of(";"));
  for(unsigned int i=0;i<parts.size();i++) {
    boost::trim(parts[i]);
    boost::split(pieces, parts[i], boost::is_any_of(","));
    int value = boost::lexical_cast<int>(pieces[0]);
    for(unsigned int j=1;j<pieces.size();j++) {
      int key = boost::lexical_cast<int>(pieces[j]);
      term_feature_map[key] = value;
    }
  }
  if(dir!=NULL) {
    std::map<int, int> groupMaxIndex;
    for(int i=1;i<=max_groupid;i++) {
      groupMaxIndex[i] = 0;
    }
    while((dirp = readdir(dir)) != NULL) {
      filename = dirp->d_name;
      std::string filename_str(filename);
      if(filename_str.substr(0,4) != "2015") {
        continue;
      } else {
      }
      filename_str = binaryfolder + "/" + filename_str;
      std::ifstream fin(filename_str.c_str(), std::ios::binary);
      int count_of_file = 0;
      fin.read((char*)(&count_of_file), sizeof(int));
      total_instance_num += count_of_file;
      fin.close();
    }
    closedir(dir);
    int** feature = new int*[total_instance_num];
    int** groupid = new int*[total_instance_num];
    double* target = new double[total_instance_num];
    int* width = new int[total_instance_num];
    dataset->length = total_instance_num;
    int idx=0;
    dir = opendir(binaryfolder.c_str());
    while((dirp = readdir(dir)) != NULL) {
      filename = dirp->d_name;
      std::string filename_str(filename);
      if(filename_str.substr(0,4) != "2015") {
        continue;
      }
      filename_str = binaryfolder + "/" + filename_str;
      std::ifstream fin(filename_str.c_str(), std::ios::binary);
      if(fin) {
 //       std::cerr<<"open "<<filename_str<<" successfully!"<<std::endl;
      } else {
//        std::cerr<<"open "<<filename_str<<" failded!"<<std::endl;
      }
      int count_of_file = 0;
      int idx_of_file = 0;
      fin.read((char*)(&count_of_file), sizeof(int));
      for(int i=0;i<count_of_file;i++) {
        double label=-1;
        int w;
        fin.read((char*)(&label), sizeof(double));
        if(!fin) {
          std::cerr<<"file eof.[label] idx = "<<idx<<" label = "<<label<<std::endl;
          break;
        }
        target[idx] = label;
        fin.read((char*)(&w), sizeof(int));
        if(!fin) {
          std::cerr<<"file eof.[width] idx = "<<idx<<std::endl;
          break;
        }
        width[idx] = w;
        feature[idx] = new int[w];
        groupid[idx] = new int[w];
        fin.read((char*)groupid[idx], sizeof(int)*w);
        if(!fin) {
          std::cerr<<"file eof.[groupid] idx = "<<idx<<std::endl;
          break;
        }
        fin.read((char*)feature[idx], sizeof(int)*w);
        if(!fin) {
          std::cerr<<"file eof.[featureid] idx = "<<idx<<std::endl;
          break;
        }
        for(int j=0;j<w;j++) {
          int fid = feature[idx][j];
          int gid = groupid[idx][j];
          int groupId_for_convert;
          if(term_feature_map.find(gid) != term_feature_map.end()) {
            groupId_for_convert = term_feature_map[gid];
            gid = groupId_for_convert;
          }
          if(gid > max_groupid ||  gid <=0) {
            continue;
          }
          groupMaxIndex[gid] = groupMaxIndex[gid]>fid?groupMaxIndex[gid]:fid;
        }

        idx++;
        idx_of_file++;
      }
      std::cout<<"\r                             ";
      std::cout<<"\rload data:"<<(idx+0.0)*100/total_instance_num<<" %";
      std::cout.flush();
      std::string result;
      if(idx_of_file == count_of_file) {
        result = " equal";
      } else {
        result = " unequal";
      }
      fin.close();
    }
    dataset->feature = feature;
    dataset->width = width;
    dataset->groupid = groupid;
    dataset->length = idx;
    dataset->target = target;
    dataset->groupMaxIndex = groupMaxIndex;
    int maxgid = 0;
    for(std::map<int,int>::iterator iter=groupMaxIndex.begin();
        iter!=groupMaxIndex.end();iter++) {
      int gid = iter->first;
      maxgid = maxgid < gid ? gid : maxgid;
      int maxindex = iter->second;
    }
    std::string param = "";
    std::cout<<std::endl;
    for(std::map<int,int>::iterator it=groupMaxIndex.begin();it!=groupMaxIndex.end();it++) {
      std::string gid = std::to_string(it->first);
      std::string tmp = std::to_string(it->second);
      if(find(bias_feature_vec.begin(), bias_feature_vec.end(), it->first) != bias_feature_vec.end()) {
        continue;
      }
      if(it->second == 0) {
        continue;
      }
      param+=gid+":"+tmp+",";
    }
    dataset->table_param = param;
    param = "";
    for(unsigned int i=0;i<bias_feature_vec.size();i++) {
      int gid = bias_feature_vec[i];
      int fid = groupMaxIndex[gid];
      param+=std::to_string(gid)+":"+std::to_string(fid)+",";
    }
    dataset->bias_param = param;
    //std::cout<<"max groupid:"<<maxgid<<std::endl;
    //std::cout<<"groupid num:"<<groupMaxIndex.size()<<std::endl;
    std::cout<<"\nlookup table param:\n"<<dataset->table_param<<std::endl;
    std::cout<<"bias feature param:\n"<<dataset->bias_param<<std::endl;
    //std::cerr<<"read "<<idx<<" instances. total instance num = "<<total_instance_num<<std::endl;
    closedir(dir);
  } else {
    std::cerr<<"open folder faild ..."<<std::endl;
  }
  std::cerr<<std::endl;
  //std::cerr<<"close the folder"<<std::endl;
  return true;
}

bool DropFeature(int groupid) {
  if(1<=groupid && groupid<=4) {
    return true;
  }
  if(23<=groupid && groupid<=24) {
    return true;
  }
  if(27<=groupid && groupid<=28) {
    return true;
  }
  return false;
}

bool RemovePositionFeature(SparseDataSet* dataset) {
  int length = dataset->length;
  int* width = dataset->width;
  int** feature = dataset->feature;
  int** groupid = dataset->groupid;
  for(int i=0;i<length;i++) {
    for(int j=0;j<width[i];j++) {
      if(DropFeature(groupid[i][j])) {
        feature[i][j]=-1;
      }
    }
  }
  return true;
}

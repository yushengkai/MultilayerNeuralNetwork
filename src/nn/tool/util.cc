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


bool ReadSparseDataFromBinFolder(std::string binaryfolder,
                                 SparseDataSet* dataset) {
  DIR* dir = opendir(binaryfolder.c_str());
  struct dirent *dirp;
  char* filename;
  int total_instance_num = 0;
  if(dir!=NULL) {
    std::map<int, int> groupMaxIndex;
    while((dirp = readdir(dir)) != NULL) {
      filename = dirp->d_name;
      std::string filename_str(filename);
      if(filename_str.substr(0,4) != "part") {
        continue;
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
      if(filename_str.substr(0,4) != "part") {
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
        if(idx<count_of_file) {
          for(int j=0;j<w;j++) {
            int fid = feature[idx][j];
            std::map<int, int>::iterator iter = groupMaxIndex.find(groupid[idx][j]);
            int gid = groupid[idx][j];
            if(iter == groupMaxIndex.end()) {
              groupMaxIndex[gid]=0;
            }
            groupMaxIndex[gid] = groupMaxIndex[gid]>fid?groupMaxIndex[gid]:fid;
          }
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
    for(int i=0;i<=maxgid;i++) {
      std::map<int, int>::iterator iter = groupMaxIndex.find(i);
      if(iter != groupMaxIndex.end()) {
        int value = iter->second;
        int group_length = value+1;
        std::string tmp = std::to_string(group_length);
        param+=tmp;
      } else {
        std::string tmp = std::to_string(i);
        param+="1";
      }

      if(i<maxgid) {
        param+=":";
      }
    }
    dataset->table_param = param;
    //std::cout<<"max groupid:"<<maxgid<<std::endl;
    //std::cout<<"groupid num:"<<groupMaxIndex.size()<<std::endl;
    std::cout<<"\nlookup table param:\n"<<param<<std::endl;
    //std::cerr<<"read "<<idx<<" instances. total instance num = "<<total_instance_num<<std::endl;
    closedir(dir);
  } else {
    std::cerr<<"open folder faild ..."<<std::endl;
  }
  std::cerr<<std::endl;
  //std::cerr<<"close the folder"<<std::endl;
  return true;
}



// Copyright (c) 2015 Qihoo 360 Technology Co. Ltd
// Author: Yu Shengkai (yushengkai@360.cn)


#ifndef UTIL_H_
#define UTIL_H_
#include <gflags/gflags.h>
#include <string>
#include <map>
//#include <boost/random.hpp>

typedef struct DataSet {
  double* feature;
  double* target;
  int width;
  int length;
} DataSet;

typedef struct SparseDataSet {
  int** feature;
  int** groupid;
  double* target;
  int* width;
  int length;
  int max_featureid;
  std::map<int, int> groupMaxIndex;
  std::string table_param;
  std::string bias_param;
} SparseDataSet;


double sigmoid(double x);
double tanha(double );
double ReLU(double x);
bool ReadMNIST(std::string filename, DataSet* dataset);
bool ReadSparseData(std::string filename, std::string binaryname, SparseDataSet* dataset);
bool ReadSparseDataFromBin(std::string binaryname, SparseDataSet* dataset);
bool DeleteSparseData(SparseDataSet* dataset);
bool ReadSparseDataFromBinFolder(std::string binaryfolder,
                                 std::string bias_feature,
                                 std::string term_feature,
                                 std::map<int, int>& term_feature_map,
                                 SparseDataSet* dataset);
bool RemovePositionFeature(SparseDataSet* dataset);

/*

static boost::mt19937 rng(static_cast<unsigned>(std::time(0)));
boost::normal_distribution<double> norm_dist(FLAGS_mu, FLAGS_sigma);
boost::variate_generator<boost::mt19937&, boost::normal_distribution<double> >
normal_sample(rng, norm_dist);

*/
#endif  // UTIL_H_


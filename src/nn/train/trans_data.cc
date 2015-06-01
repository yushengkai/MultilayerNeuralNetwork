// Copyright (c) 2015 Qihoo 360 Technology Co. Ltd
// Author: Mu Yixiang (muyixiang@360.cn)

#include "tool/util.h"
#include <gflags/gflags.h>

DEFINE_string(filename, "/data/yushengkai/sparse_dnn/sparse_dnn_test.txt", "");
DEFINE_string(binaryname, "/data/yushengkai/sparse_dnn/sparse_dnn_test.bin", "");

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  std::string filename = FLAGS_filename;
  std::string binaryname = FLAGS_binaryname;
  SparseDataSet* trainData = new SparseDataSet();
  if(!ReadSparseData(filename, binaryname, trainData)) {
  }
}


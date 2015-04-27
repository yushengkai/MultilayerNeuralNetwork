// Copyright (c) 2015 Qihoo 360 Technology Co. Ltd
// Author: Mu Yixiang (muyixiang@360.cn)


#include "gtest/gtest.h"

int main(int argc, char **argv)
{
  testing::AddGlobalTestEnvironment(new StdNullEnvironment);
  testing::AddGlobalTestEnvironment(new RandomEnvironment);
  ::testing::InitGoogleTest(&argc, argv);
  google::ParseCommandLineFlags(&argc, &argv, true);
  return RUN_ALL_TESTS();
}



// Copyright (c) 2015 Qihoo 360 Technology Co. Ltd
// Author: Mu Yixiang (muyixiang@360.cn)


#ifndef FLAG_H_
#define FLAG_H_
#include <gflags/gflags.h>

DECLARE_string(lookup_table_param);
DECLARE_int32(lookup_table_width);
DECLARE_int32(minibatchsize);
DECLARE_double(sigma);
DECLARE_double(mu);
DECLARE_string(init_type);
DECLARE_string(nn_param);
DECLARE_bool(with_bias);
DECLARE_string(tranfer_func);
DECLARE_string(log_dir);
DECLARE_double(learning_rate);
DECLARE_bool(logtostderr);
DECLARE_int32(test_batchsize);

#endif  // FLAG_H_


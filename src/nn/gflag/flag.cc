// Copyright (c) 2015 Qihoo 360 Technology Co. Ltd
// Author: Yu Shengkai (yushengkai@360.cn)
#include <gflags/gflags.h>
#include "gflag/flag.h"

DEFINE_string(lookup_table_param, "10:20:30:40", "lengths of lookup tables");
DEFINE_int32(lookup_table_width, 196, "width of all the lookup tables");
DEFINE_int32(minibatchsize, 50, "minibatchsize");
DEFINE_double(sigma, 0.1, "sigma of gaussian distribution");
DEFINE_double(mu, 0, "mu of gaussian distribution");
DEFINE_string(init_type, "normal", "initial type normal, 123, 1, 0, fromfile");
DEFINE_string(nn_param, "200:10", "layer size of nn");
DEFINE_bool(with_bias, false, "with bias");
DEFINE_string(tranfer_func, "sigmoid", "sigmoid, tanh, ReLU");
DEFINE_string(log_dir, "./log", "log dir");
DEFINE_double(learning_rate, 0.001, "learning rate");
DEFINE_bool(logtostderr, false, "log into stderr");
DEFINE_int32(test_batchsize, 10, "test batchsize");


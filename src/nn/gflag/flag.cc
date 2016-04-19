// Copyright (c) 2015 Qihoo 360 Technology Co. Ltd
// Author: Yu Shengkai (yushengkai@360.cn)
#include <gflags/gflags.h>
#include "gflag/flag.h"

DEFINE_string(bias_feature, "", "lengths of lookup tables");
DEFINE_string(lookup_table_param,"0:149999,1:149999", "");
DEFINE_string(bias_param, "", "");
DEFINE_int32(lookup_table_width, 50, "width of all the lookup tables");
DEFINE_int32(minibatchsize, 100, "minibatchsize");
DEFINE_double(sigma, 0.1, "sigma of gaussian distribution");
DEFINE_double(mu, 0, "mu of gaussian distribution");
DEFINE_string(init_type, "normal", "initial type normal, 123, 1, 0, fromfile");
DEFINE_string(nn_layer_param, "10:2", "layer size of nn");
DEFINE_bool(with_bias, false, "with bias");
DEFINE_string(tranfer_func, "sigmoid", "sigmoid, tanh, ReLU");
DEFINE_string(log_dir, "./log", "log dir");
DEFINE_double(learning_rate, 0.001, "learning rate");
DEFINE_bool(logtostderr, false, "log into stderr");
DEFINE_int32(test_batchsize, 11, "test batchsize");
DEFINE_int32(max_groupid, 28, "max groupid");
DEFINE_string(term_feature, "", "term feature refer to the same lookup table");
//DEFINE_string(term_feature, "5,6;10,11,12,13", "term feature refer to the same lookup table");

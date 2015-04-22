// Copyright (c) 2015 Qihoo 360 Technology Co. Ltd
// Author: Yu Shengkai (yushengkai@360.cn)


#ifndef UTIL_H_
#define UTIL_H_
#include <gflags/gflags.h>
//#include <boost/random.hpp>
double sigmoid(double x);
double tanha(double );
double ReLU(double x);


/*

static boost::mt19937 rng(static_cast<unsigned>(std::time(0)));
boost::normal_distribution<double> norm_dist(FLAGS_mu, FLAGS_sigma);
boost::variate_generator<boost::mt19937&, boost::normal_distribution<double> >
normal_sample(rng, norm_dist);

*/
#endif  // UTIL_H_


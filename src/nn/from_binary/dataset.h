#ifndef _DATASET_H_
#define _DATASET_H_
#include <cmath>
#include <string>
#include <random>
#include <cstdlib>
#include <vector>
#include <stdint.h>
#include "gflags/gflags.h"

DECLARE_int32(group_bit);
struct FeatureNode{
  uint32_t feature;
  uint8_t get_group_id()const{
      return FeatureNode::get_group_id(feature);
  }
  uint32_t get_feature_id()const{
      return FeatureNode::get_feature_id(feature);
  }
  bool is_end()const{
      return FeatureNode::is_end(feature);
  }
  static uint8_t get_group_id(uint32_t feature){
      return (feature & (((1<<FLAGS_group_bit)-1)<<(32-FLAGS_group_bit))) >> (32-FLAGS_group_bit);
  }
  static uint32_t get_feature_id(uint32_t feature){
      return feature & ~(((1<<FLAGS_group_bit)-1)<<(32-FLAGS_group_bit));
  }
  static bool is_end(uint32_t feature){
      return feature == 0xFFFFFFFF;
  }
};
/*    int group;
    int index; // -1 means end of line, lowest index is 0 no matter what the low index of the data file
    float value;
};
*/

struct Problem{
//    uint32_t n,l;  // number of features & number of instances
    uint64_t l; // number of instance
    int8_t *y; // label
    float *weight;
    struct FeatureNode **x;
    uint64_t get_feature_num()const{
        return 0;
    }
    uint64_t get_instance_num()const{
        return l;
    }
	const float *get_weight()const{    return weight;   };
    void normalize_weight();
    uint64_t roulette_selection(std::mt19937&)const;
    uint64_t uniform_select(std::mt19937&)const;
    void shuffle();
    void free_memory(){
        if(y) delete[] y;
        if(weight) delete[] weight;
        if(x){
            delete[] x[0];
            delete[] x;
        }
    };
//    static Problem unittest(const std::string& filename = std::string("train.lbm.smp"));
    static void list_problem_struct(const Problem&,uint64_t i);
    bool read_from_binary(std::ifstream& inStream);
    bool write_ml_data(const std::string& filename)const;
};

//Problem read_problem(const char *filename);

#endif

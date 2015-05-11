#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <cctype>
#include <cerrno>
#include "dataset.h"

DEFINE_int32(group_bit,5,"number of group bits");
bool Problem::read_from_binary(std::ifstream& inStream){
    inStream.read((char*)(&l),sizeof(uint64_t));
  //  std::cout<<"[INF]number of instance="<<l<<std::endl;
    y = new int8_t[l];
    weight = new float[l];
    if(!inStream.read(reinterpret_cast<char*>(y),sizeof(int8_t) * l)){
        std::cerr<<"read y failed.\n";
        return false;
    }
    uint64_t count = 0;
    for(uint64_t i = 0; i < l; ++i){
        count += y[i];
    }
  //  std::cout<<"[INF]clicks="<<count<<std::endl;
    if(!inStream.read(reinterpret_cast<char*>(weight),sizeof(float) * l)){
        std::cerr<<"read weight failed.\n";
        return false;
    }
    uint8_t *num_features = new uint8_t[l];
    if(!inStream.read(reinterpret_cast<char*>(num_features),sizeof(uint8_t) * l)){
        std::cerr<<"read num_features failed.\n";
        return false;
    }
    uint64_t total_features = 0;
    for(uint64_t i = 0; i < l; ++i){
        total_features += num_features[i];
    }
   // std::cout<<"[INF]need total features="<<total_features<<std::endl<<std::endl;
    uint32_t *mem = new uint32_t[total_features];
    if(!inStream.read(reinterpret_cast<char*>(mem),sizeof(uint32_t) * total_features)){
        std::cerr<<"read mem failed. read bytes="<<inStream.gcount()<<"\n";
        return false;
    }
    x = new FeatureNode*[l];
    x[0] = (FeatureNode*)mem;
    for(uint64_t i = 1; i < l; ++i){
        x[i] = x[i - 1] + num_features[i-1];
    }
    delete[] num_features;
    return true;
}
bool Problem::write_ml_data(const std::string& filename)const{
    std::ofstream fout(filename.c_str(),std::ios_base::out|std::ios_base::binary);
    if(!fout.is_open()){
        std::cerr<<"[ERR] open file "<<filename<<" for write failed."<<std::endl;
        return false;
    }
    if(!fout.write((char*)&l,sizeof(uint64_t))){
        std::cerr<<"[ERR] write num instance failed."<<std::endl;
        return false;
    }
    std::vector<int32_t> labels(l,0);
    std::vector<uint32_t> num_features (l,0);
    uint64_t total_features = 0;
    for(uint64_t i = 0; i < l; ++i){
        labels[i] = y[i];
        const FeatureNode *p = x[i];
        while(!p->is_end()){
            ++num_features[i];
            ++p;
        }
        total_features += num_features[i];
    }
    if(!fout.write(reinterpret_cast<char*>(&labels[0]),sizeof(int32_t))){
        std::cerr<<"[ERR] write labels failed."<<std::endl;
        return false;
    }
    if(!fout.write(reinterpret_cast<char*>(&num_features[0]),sizeof(uint32_t))){
        std::cerr<<"[ERR] write num features failed."<<std::endl;
        return false;
    }
    if(!fout.write(reinterpret_cast<char*>(&total_features),sizeof(uint64_t))){
        std::cerr<<"[ERR] write total num features failed."<<std::endl;
        return false;
    }
    for(uint64_t i = 0; i < l; ++i){
        if(!fout.write(reinterpret_cast<char*>(&x[i]),sizeof(uint32_t) * num_features[i])){
            std::cerr<<"[ERR] write features failed."<<std::endl;
            return false;
        }
    }
    fout.close();
    return true;
}
void Problem::normalize_weight(){
    double sum = 0.0, s = 0.0;
    for(uint64_t i = 0; i < get_instance_num(); ++i){
        if(y[i] != 1)    weight[i] = 10;
        else             weight[i] = 1.0 ; //+ std::log10(weight[i]);
        sum += weight[i];
    }
	std::cout<<"[INF] sum weights="<<sum<<std::endl;
	return;
    for(uint64_t i = 0; i < get_instance_num(); ++i){
        weight[i] = s / sum;
        s += weight[i];
    }
}
uint64_t Problem::uniform_select(std::mt19937& rnd_generator)const{
    std::uniform_int_distribution<uint64_t> distribution(0,get_instance_num() - 1);
    return distribution(rnd_generator);
}
uint64_t Problem::roulette_selection(std::mt19937& rnd_generator)const{
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    double slot = distribution(rnd_generator);
    uint64_t beg = 0, end = get_instance_num() - 1, mid = 0;
    while(beg <= end){
        mid = (beg + end) / 2;
        if(slot < weight[mid])   end = mid - 1;
        else if( mid == (get_instance_num() - 1) || slot < weight[mid + 1])    return mid;
        else beg = mid + 1;
    }
    return beg;
}
void Problem::shuffle(){
    for(uint64_t i = 1; i < l; ++i){
        uint64_t j = rand() % (l - 1) + 1;
        std::swap(y[i],y[j]);
        std::swap(weight[i],weight[j]);
        std::swap(x[i],x[j]);
    }
}
void Problem::list_problem_struct(const Problem& prob,uint64_t i){
    std::cout<<"number of instances="<<prob.get_instance_num();
    std::cout<<"the "<<i<<"th line of data file is:"<<std::endl;
    for(uint32_t j = 0; j < i; ++j){
        std::cout<<(int32_t)prob.y[j]<<" "<<prob.weight[j];
        FeatureNode *x = prob.x[j];
        while (!x->is_end()){
            std::cout<<" "<<(uint32_t)x->get_group_id()<<":"<<x->get_feature_id();
            x += 1;
        }
        std::cout<<std::endl;
    }
    std::cout<<std::endl;
}


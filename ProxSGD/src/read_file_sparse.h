#ifndef __READ_FILE_H__
#define __READ_FILE_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include <random>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <algorithm>
using namespace std;

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/trim.hpp>

#include <Eigen/Core>
using namespace Eigen;

#include "utils.h"

typedef Matrix<float, Dynamic, Dynamic, RowMajor> RMatrixXf;

typedef struct _featureid_score{
  uint32_t featureid;
  float score;
} _featureid_score;

typedef struct _data_self_ref{
  int click;
  uint32_t feature_length;
  _featureid_score* featureid_score;
  struct _data_self_ref *next;
} data_self_ref;

typedef struct _data_array{
  int click;
  uint32_t feature_length;
  _featureid_score* featureid_score;
} data_array;

void 
split(const string&, const string, vector<string>&);

class SelfReferenceStruct
{
 public:

  // cli parms
  arg_params* cli_params;
  ~SelfReferenceStruct(void);

  SelfReferenceStruct(arg_params* _cli_params);


  /* utils and read_file */
  uint32_t get_data_length(const char*);
  uint32_t get_feature_length(const char*,
			      std::string, std::string, std::string);
  void load_data(data_self_ref**, data_self_ref**,
		 char*,
		 std::string, std::string, std::string,
		 uint32_t*);
  void show_data(data_self_ref**, data_self_ref**, uint8_t head=1);
  //void show_num_line(data_self_ref**, data_self_ref**, 
  //		     uint32_t data_size, uint8_t random=1, uint8_t head=1);
  void free_data(data_self_ref**, data_self_ref**);

 private:
  uint32_t get_max(uint32_t, uint32_t);
  uint32_t get_min(uint32_t, uint32_t);

};


class ArrayStruct
{
 public:

  // cli parms
  arg_params* cli_params;

  data_array* seq; // store train or test data

  ~ArrayStruct(void);

  ArrayStruct(arg_params* _cli_params);

  /* utils and read_file */
  uint32_t get_data_length(const char*);
  uint32_t get_feature_length(const char*,
			      std::string, std::string, std::string);

  void init_data(uint32_t data_size);
  void load_data(char*,
		 std::string, std::string, std::string);
  void show_data(uint32_t data_size, uint8_t head=1);
  void show_num_line(uint32_t data_size, uint8_t random=1, uint8_t head=1);
  void free_data(uint32_t data_size);

 private:
  uint32_t get_max(uint32_t, uint32_t);
  uint32_t get_min(uint32_t, uint32_t);
};

#endif //__READ_FILE_H__

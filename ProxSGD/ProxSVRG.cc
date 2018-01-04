#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <boost/algorithm/string/join.hpp>

#include "src/utils.h"
#include "src/read_file_sparse.h"
#include "src/ProxSVRG_sparse.h"

#include <unordered_map>
using namespace std;

int main(int argc, char **argv)
{

  // 1. option parser
  arg_params* cli_params = (arg_params*)malloc(sizeof(arg_params));
  read_args(argc, argv, cli_params);
  show_args(cli_params);


  // 2. make instance for normal data form
  //  2.1 read train data and initialize feature weight vector
  ArrayStruct train(cli_params);
  uint32_t train_size = 0;
  train_size = train.get_data_length(cli_params->train);
  uint32_t feature_dim = 0;
  feature_dim = train.get_feature_length(cli_params->train,
				      cli_params->delimiter, 
				      cli_params->delimiter_between,
				      cli_params->delimiter_within);
  Logging("(N,D)=(%u, %u)", train_size, feature_dim);
  train.init_data(train_size);
  train.load_data(cli_params->train,
		  cli_params->delimiter, cli_params->delimiter_between, cli_params->delimiter_within);
  //train.show_data(train_size);

  ArrayStruct valid(cli_params);
  uint32_t valid_size = 0;
  valid_size = valid.get_data_length(cli_params->valid);
  valid.init_data(valid_size);
  valid.load_data(cli_params->valid,
		  cli_params->delimiter, cli_params->delimiter_between, cli_params->delimiter_within);


  // 3. Train
  const gsl_rng_type *T;
  gsl_rng *r;
  T = gsl_rng_mt19937;        // random generator
  r = gsl_rng_alloc(T);       // random gererator pointer
  gsl_rng_set(r, time(NULL)); // initialize seed for random generator by sys clock

  // 3.1 Train
  std::vector<std::string> my_arr;
  my_arr.push_back(cli_params->out_path);
  my_arr.push_back(cli_params->out_fname);
  std::string joined = boost::algorithm::join(my_arr, "/");

  FILE *output;
  if ( (output = fopen(joined.c_str(), "w")) == NULL ) {
    printf("can not make output file");
    exit(1);
  }

  unordered_map<uint32_t, float> w;
  float w_level = 1.0f / (float)(feature_dim + 1);              // feature weight level
  float *w_level_ptr = &w_level;

  // ProxSVRG
  PROX_SVRG psvrg = PROX_SVRG(train_size, feature_dim,
			      valid_size,
			      cli_params);
  psvrg.init_feature_weight(w, feature_dim);
  psvrg.train(r, output,
	      train.seq, valid.seq,
	      w, w_level_ptr);

  fclose(output);


  // 4. free data
  train.free_data(train_size);
  valid.free_data(valid_size);

  return 0;
}

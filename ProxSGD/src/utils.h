#ifndef __UTIL__H__
#define __UTIL__H__

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <time.h>

#include <unordered_map>
#include <iostream>
#include <fstream>
#include <string>
using namespace std;

#define max_len 1024

#define Logging(fmt, args...) do  {						\
  time_t time_since_epoch = time(NULL);					\
  struct tm* tm_info = localtime(&time_since_epoch);			\
  fprintf(stderr, "%02d/%02d %02d:%02d:%02d " fmt "\n",			\
	  1+tm_info->tm_mon, tm_info->tm_mday, tm_info->tm_hour,	\
	  tm_info->tm_min, tm_info->tm_sec, ##args);			\
} while (0)

typedef std::unordered_map<unsigned int, float> feature_vector;

typedef struct _samples{
  int click;
  feature_vector fv;
} samples;

typedef struct {
  char* train;
  char* valid;
  char* test;
  char* loss;
  char* out_path;
  char* out_fname;
  char* out_test;
  float step_size;
  float lambda;                           // L2 regularization param
  uint16_t m;                             // max number of stochastic steps per inner loop
  float nu;                               // lower bound on lambda. nu is constant for nu-strong
  float step_size_adagrad;                // Adagrad
  float epsilon;                          // Adagrad
  float clip_threshold;                   // Adagrad
  char* clip_method;                      // Adagrad
  uint8_t max_iter_adagrad;               // Adagrad
  uint32_t table_size;
  uint8_t mini_batch_size;
  float convergence_threshold;
  uint8_t convergence_threshold_count_train;
  uint8_t max_iter;
  uint8_t sparse;
  uint8_t update_option;
  char* delimiter;
  char* delimiter_between;
  char* delimiter_within;
  uint8_t num_thread;
  float mini_batch_rate;
  uint8_t convergence_threshold_count_test;
} arg_params;

// for adagrad
typedef struct _E_adaptive{
  float E;
  float max_grad;
} E_adaptive;

char* string2char(std::string);
std::string char2string(const char* cstr);
void read_args(int, char**, arg_params*);
void show_args(arg_params*);
bool isEmpty(const char*);
bool isNULL(const char*);

#endif //__UTIL__H__

#include "utils.h"

static struct option options[] =
  {
    {"train", required_argument, NULL, 'a'},
    {"test", required_argument, NULL, 'b'},
    {"out_path", required_argument, NULL, 'c'},
    {"out_fname", required_argument, NULL, 'd'},
    {"step_size", required_argument, NULL, 'e'},
    {"lambda", required_argument, NULL, 'f'},
    {"m", required_argument, NULL, 'g'},
    {"nu", required_argument, NULL, 'n'},
    {"table_size", required_argument, NULL, 'o'},
    {"mini_batch_size", required_argument, NULL, 'h'},
    {"convergence_threshold", required_argument, NULL, 'i'},
    {"convergence_threshold_count_train", required_argument, NULL, 'm'},
    {"max_iter", required_argument, NULL, 'j'},
    {"sparse", required_argument, NULL, 'k'},
    {"update_option", required_argument, NULL, 'l'},
    {"delimiter", required_argument, NULL, 'p'},
    {"delimiter_between", required_argument, NULL, 'q'},
    {"delimiter_within", required_argument, NULL, 'r'},
    {"out_test", required_argument, NULL, 's'},
    {"num_thread", required_argument, NULL, 't'},
    {"mini_batch_rate", required_argument, NULL, 'u'},
    {"convergence_threshold_count_test", required_argument, NULL, 'w'},
    {"epsilon", required_argument, NULL, 'x'},
    {"clip_threshold", required_argument, NULL, 'y'},
    {"clip_method", required_argument, NULL, 'z'},
    {"max_iter_adagrad", required_argument, NULL, 'aa'},
    {"step_size_adagrad", required_argument, NULL, 'ab'},
    {"valid", required_argument, NULL, 'ac'},
    {"loss", required_argument, NULL, 'ad'},
    {0, 0, 0, 0}
  };

char* string2char(std::string str)
{
  char* cstr = new char[max_len + 1]; 
  strcpy(cstr, str.c_str());
  return cstr;
}

std::string char2string(const char* cstr)
{
  std::string str(cstr);
  return str;
}

void read_args(int argc_, char **argv_, arg_params *cli_param)
{
  
  // default values for input options
  std::string tmp_fname;
  tmp_fname.clear(); tmp_fname = "/home/tanakai/data/libsvm/sparse/a1a.train";
  cli_param->train = string2char(tmp_fname);
  tmp_fname.clear(); tmp_fname = "/home/tanakai/data/libsvm/sparse/a1a.valid";
  cli_param->valid = string2char(tmp_fname);
  tmp_fname.clear(); tmp_fname = "/home/tanakai/data/libsvm/sparse/a1a.t";
  cli_param->test = string2char(tmp_fname);

  tmp_fname.clear(); tmp_fname = "logistic";
  cli_param->loss = string2char(tmp_fname);

  tmp_fname.clear(); tmp_fname = "/home/tanakai/projects/git/c/regret/ProximalOperator/ProxSGD/result"; 
  cli_param->out_path = string2char(tmp_fname);
  tmp_fname.clear(); tmp_fname = "ProxSGD_L1.dat"; 
  cli_param->out_fname = string2char(tmp_fname);
  tmp_fname.clear(); tmp_fname = "ProxSGD_L1.dat.test";
  cli_param->out_test = string2char(tmp_fname);
  
  // numeric setting params
  cli_param->step_size = 1.0f;
  cli_param->lambda = 0.1f;
  cli_param->m = (uint16_t)10;                   //
  cli_param->nu = 0.0f;                          // if nu is zero, then t_j obey uniform distirbution
  cli_param->table_size = (uint32_t)10000;   // Non-Uniform Sampling Table
  cli_param->mini_batch_size = 1;
  cli_param->convergence_threshold = 0.001f;
  cli_param->convergence_threshold_count_train = 5;
  cli_param->max_iter = 100;
  cli_param->sparse = 1;
  cli_param->update_option = 1;                  // default update option is 1. 
                                                 // option 2 is to Iterate averaging for snapshot

  tmp_fname.clear(); tmp_fname = " ";
  cli_param->delimiter = string2char(tmp_fname);
  tmp_fname.clear(); tmp_fname = ":";
  cli_param->delimiter_between = string2char(tmp_fname);
  tmp_fname.clear(); tmp_fname = "_";
  cli_param->delimiter_within = string2char(tmp_fname);

  cli_param->num_thread = 1;
  cli_param->mini_batch_rate = 0.01f;
  cli_param->convergence_threshold_count_test = 3;

  // Adagrad
  cli_param->epsilon = 1.0e-8;
  cli_param->clip_threshold = 1.0f;
  tmp_fname.clear(); tmp_fname = "MaxClipping"; 
  cli_param->clip_method = string2char(tmp_fname);
  cli_param->max_iter_adagrad = 1;
  cli_param->step_size_adagrad = 1.0f;


  // command line options
  int dummy, index;
  while( (dummy = getopt_long(argc_, argv_, "abcdefghijk", options, &index)) != -1 ){
    switch(dummy){
    case 'a':
      cli_param->train = optarg;
      break;
    case 'b':
      cli_param->test = optarg;
      break;
    case 'c':
      cli_param->out_path = optarg;
      break;
    case 'd':
      cli_param->out_fname = optarg;
      break;
    case 'e':
      cli_param->step_size = (float)atof(optarg);
      break;
    case 'f':
      cli_param->lambda = (float)atof(optarg);
      break;
    case 'g':
      cli_param->m = (uint16_t)atoi(optarg);
      break;
    case 'n':
      cli_param->nu = (float)atof(optarg);
      break;
    case 'o':
      cli_param->table_size = (uint32_t)atoi(optarg);
      break;
    case 'h':
      cli_param->mini_batch_size = (uint8_t)atoi(optarg);
      break;
    case 'i':
      cli_param->convergence_threshold = (float)atof(optarg);
      break;
    case 'm':
      cli_param->convergence_threshold_count_train = (uint8_t)atoi(optarg);
      break;
    case 'j':
      cli_param->max_iter = (uint8_t)atoi(optarg);
      break;
    case 'k':
      cli_param->sparse = (uint8_t)atoi(optarg);
      break;
    case 'l':
      cli_param->update_option = (uint8_t)atoi(optarg);
      break;
    case 'p':
      cli_param->delimiter = optarg;
      break;
    case 'q':
      cli_param->delimiter_between = optarg;
      break;
    case 'r':
      cli_param->delimiter_within = optarg;
      break;
    case 's':
      cli_param->out_test = optarg;
      break;
    case 't':
      cli_param->num_thread = (uint8_t)atoi(optarg);
      break;
    case 'u':
      cli_param->mini_batch_rate = (float)atof(optarg);
      break;
    case 'w':
      cli_param->convergence_threshold_count_test = (uint8_t)atoi(optarg);
      break;
    case 'x':
      cli_param->epsilon = (float)atof(optarg);
      break;
    case 'y':
      cli_param->clip_threshold = (float)atof(optarg);
      break;
    case 'z':
      cli_param->clip_method = optarg;
      break;
    case 'aa':
      cli_param->max_iter_adagrad = (uint8_t)atoi(optarg);
      break;
    case 'ab':
      cli_param->step_size_adagrad = (float)atof(optarg);
      break;
    case 'ac':
      cli_param->valid = optarg;
      break;
    case 'ad':
      cli_param->loss = optarg;
      break;
    default:
      printf("Error: An unkown option\n");
      exit(1);
    }
  }
}

void show_args(arg_params *cli_param)
{
  //char* pt1 = cli_param->train;
  //printf("%s %p %p\n", cli_param->train, pt1, &(cli_param->train));

  Logging("[ProxSGD Setting Params]");
  Logging("\t Trainfile               : %s",   cli_param->train);
  Logging("\t Validfile               : %s",   cli_param->valid);
  Logging("\t Testfile                : %s",  cli_param->test);
  Logging("\t Loss                    : %s",  cli_param->loss);
  Logging("\t OutPath                 : %s",   cli_param->out_path);
  Logging("\t OutFname                : %s",    cli_param->out_fname);
  Logging("\t InitialStepSize         : %1.3e",    cli_param->step_size);
  Logging("\t L1-Lambda               : %1.3e",    cli_param->lambda);
  Logging("\t MaxInnerLoop            : %d",    cli_param->m);
  Logging("\t NonUniformSamplingParam : %1.3e",    cli_param->nu);
  Logging("\t SamplingTableSize       : %d",    cli_param->table_size);
  Logging("\t MiniBatchSize           : %d",    cli_param->mini_batch_size);
  Logging("\t ConvergenceRate         : %1.3e",    cli_param->convergence_threshold);
  Logging("\t ConvergenceCountTrain   : %d",    cli_param->convergence_threshold_count_train);
  Logging("\t ConvergenceCountTest    : %d",    cli_param->convergence_threshold_count_test);
  /*
  fprintf(stderr, "\t Trainfile               : %s\n",   cli_param->train);
  fprintf(stderr, "\t Testfile                : %s\n",  cli_param->test);
  fprintf(stderr, "\t OutPath                 : %s\n",   cli_param->out_path);
  fprintf(stderr, "\t OutFname                : %s\n",    cli_param->out_fname);
  fprintf(stderr, "\t InitialStepSize         : %1.3e\n",    cli_param->step_size);
  fprintf(stderr, "\t L1-Lambda               : %1.3e\n",    cli_param->lambda);
  fprintf(stderr, "\t MaxInnerLoop            : %d\n",    cli_param->m);
  fprintf(stderr, "\t NonUniformSamplingParam : %1.3e\n",    cli_param->nu);
  fprintf(stderr, "\t SamplingTableSize       : %d\n",    cli_param->table_size);
  fprintf(stderr, "\t Sparse                  : %d\n",    cli_param->sparse);
  fprintf(stderr, "\t F-WeightUpdateOption    : %d\n",    cli_param->update_option);
  fprintf(stderr, "\t MiniBatchSize           : %d\n",    cli_param->mini_batch_size);
  fprintf(stderr, "\t ConvergenceRate         : %1.3e\n",    cli_param->convergence_threshold);
  fprintf(stderr, "\t ConvergenceCountTrain   : %d\n",    cli_param->convergence_threshold_count_train);
  fprintf(stderr, "\t ConvergenceCountTest    : %d\n",    cli_param->convergence_threshold_count_test);
  fprintf(stderr, "\t MaxOuterLoop            : %d\n",    cli_param->max_iter);
  */

  if(strcmp(cli_param->delimiter, " ") == 0)
    Logging("\t Delimiter               : space"); //fprintf(stderr, "\t Delimiter               : space");
  else
    Logging("\t Delimiter               : %s",    cli_param->delimiter); //fprintf(stderr, "\t Delimiter               : %s",    cli_param->delimiter);

  Logging("\t DelimiterBetween        : %s",    cli_param->delimiter_between);
  Logging("\t DelimiterWithin         : %s",    cli_param->delimiter_within);
  Logging("\t NumThread               : %d",    cli_param->num_thread);
  Logging("\t MiniBatchRate           : %1.3e", cli_param->mini_batch_rate);
  /*
  fprintf(stderr, "\t DelimiterBetween        : %s\n",    cli_param->delimiter_between);
  fprintf(stderr, "\t DelimiterWithin         : %s\n",    cli_param->delimiter_within);
  fprintf(stderr, "\t NumThread               : %d\n",    cli_param->num_thread);
  fprintf(stderr, "\t MiniBatchRate           : %1.3e\n", cli_param->mini_batch_rate);
  fprintf(stderr, "\t AdagradStepSize         : %1.3e\n",    cli_param->step_size_adagrad);
  fprintf(stderr, "\t AdagradEpsilon          : %1.3e\n",    cli_param->epsilon);
  fprintf(stderr, "\t AdagradClipThreshold    : %1.3e\n",    cli_param->clip_threshold);
  fprintf(stderr, "\t AdagradClipMethod       : %s\n",    cli_param->clip_method);
  fprintf(stderr, "\t MaxIterInitWeight       : %u\n",    cli_param->max_iter_adagrad);
  */
}

bool isEmpty(const char *s)
{
  if (s==NULL ) return false;

  if( s!=NULL || strlen(s) == 0 ) return true;

  return false;
}

bool isNULL(const char *s)
{
  if( s==NULL ) return true;
  return false;
}

#include "read_file_sparse.h"

void 
split(const string &str, const string delimiter, vector<string> &array)
{
  std::string::size_type idx = str.find_first_of(delimiter);

  if ( idx != std::string::npos) {
    array.push_back(str.substr(0, idx));
    split(str.substr(idx+1), delimiter, array);
  } else {
    array.push_back(str);
  }
}


/* --------------------------- Self Reference Struct --------------------------- */
SelfReferenceStruct::~SelfReferenceStruct(void)
{}

SelfReferenceStruct::SelfReferenceStruct(arg_params* _cli_params):
  cli_params(_cli_params)
{} 

uint32_t SelfReferenceStruct::get_data_length(const char *filename)
{
  //read file
  std::string tmp_string; tmp_string.clear();
  tmp_string = char2string(filename);
  ifstream in(tmp_string);
  string s;

  if ( !in.is_open() ){
    printf("Can not open %s\n", filename);
    exit(1);
  }

  uint32_t count = 0;
  while (std::getline(in, s)) {
    istringstream iss(s);
    string line;
    //iss >> line;
    count += 1;
  }
  in.close();


  return count;
}


uint32_t SelfReferenceStruct::get_max(uint32_t a, uint32_t b)
{
  if (a > b) {
    return a;
  } else {
    return b;
  }
}

uint32_t SelfReferenceStruct::get_min(uint32_t a, uint32_t b)
{
  if (a > b) {
    return b;
  } else {
    return a;
  }
}


uint32_t SelfReferenceStruct::get_feature_length(const char *filename, 
						 string line_delimiter, 
						 string line_delimiter_between, 
						 string line_delimiter_within)
{

  std::string tmp_string; tmp_string.clear();
  tmp_string = char2string(filename);
  std::string line_buffer;

  std::ifstream in;
  in.open(filename, std::ios::in);

  if ( !in.is_open() ){
    printf("Can not open %s\n", filename);
    exit(1);
  }

  uint32_t max_feature_id = 0; uint32_t min_feature_id = 100000;
  uint32_t feature_id, feature_length;
  float feature_score;
  vector<string> my_arr, my_arr2;
  while ( std::getline(in, line_buffer) ) {
    // trim
    boost::trim_right(line_buffer);
    my_arr.clear();
    //split(line_buffer, line_delimiter, my_arr);
    boost::split(my_arr, line_buffer, boost::is_any_of(line_delimiter));
    feature_length = (uint32_t)(my_arr.size() - 1);// -1 is label, -1 1:0.1 2:0.4
    string key, value;
    for (uint32_t j = 1; j < feature_length + 1;  ++j) {
      my_arr2.clear();
      boost::split(my_arr2, my_arr[j], boost::is_any_of(line_delimiter_between));
      key.clear(); key = my_arr2[0]; feature_id = (uint32_t)atoi(key.c_str());
      value.clear(); value = my_arr2[1]; feature_score = atof(value.c_str());
      max_feature_id = get_max(max_feature_id, feature_id);
      min_feature_id = get_min(min_feature_id, feature_id);
    }

  }
  in.close();

  if (min_feature_id == 0)
    max_feature_id += 1;

  return max_feature_id;
}

void SelfReferenceStruct::load_data(data_self_ref **p, data_self_ref **start_p,
				    char *filename, 
				    std::string delimiter, std::string delimiter_between, std::string delimiter_within,
				    uint32_t *train_length)
{

  uint32_t j, k;
  data_self_ref *new_p;//新しく確保した領域を指すポインタ
  
  //read file
  std::string tmp_string; tmp_string.clear();
  tmp_string = char2string(filename);
  std::string line_buffer;

  std::ifstream in;
  in.open(filename, std::ios::in);


  // file except
  if ( !in.is_open() ){
    printf("Can not open %s\n", filename);
    exit(1);
  }


  // main
  uint32_t feature_id, feature_length;
  float feature_score; string tmp_key;
  uint32_t count = 0;
  vector<string> my_arr, my_arr2;
  while ( std::getline(in, line_buffer) ) {

    //データを格納する構造体を1個分確保
    if (count == 0) {

      *p = (data_self_ref*)malloc( sizeof(data_self_ref) );
      if ( *p == NULL ) {
        printf("can not malloc data_self_ref\n");
        exit(1);
      }
      *start_p = *p;
      (*p)->next = NULL;

    } else{

      new_p = (data_self_ref*)malloc( sizeof(data_self_ref) );
      if (new_p == NULL) {
	printf("can not malloc data_self_ref\n");
        exit(1);
      }
      (*p)->next = new_p;
      new_p->next = NULL;
      (*p) = new_p;

    }

    // trim
    boost::trim_right(line_buffer);
    my_arr.clear();
    //split(line_buffer, line_delimiter, my_arr);
    boost::split(my_arr, line_buffer, boost::is_any_of(delimiter));

    // pに新規データを追加, featureidとscoreの領域を確保
    //(*p)->click = atoi(my_arr[0].cstr());
    tmp_key.clear(); tmp_key = my_arr[0]; (*p)->click = (uint32_t)atoi(tmp_key.c_str());
    (*p)->feature_length = (uint32_t)(my_arr.size() - 1);
    (*p)->featureid_score = (_featureid_score* )malloc( ((*p)->feature_length)*sizeof(_featureid_score) );

    std::string key, value;
    for (uint32_t j = 1; j < my_arr.size();  ++j) {
      my_arr2.clear();
      boost::split(my_arr2, my_arr[j], boost::is_any_of(delimiter_between));
      key.clear(); key = my_arr2[0];
      value.clear(); value = my_arr2[1];

      //src[i]というのはコンパイル時に*(src+i)という足し算に展開されるので、
      //whileで評価するたびに足し算を行っているので、そこがボトルネックになる
      //ポインタ記法
      (*((*p)->featureid_score + (j-1))).featureid = (uint32_t)atoi(key.c_str());
      (*((*p)->featureid_score + (j-1))).score = (float)atof(value.c_str());
      //配列記法
      //(*p)->featureid_score[j-1].featureid = (uint32_t)atoi(key.c_str());
      //(*p)->featureid_score[j-1].score = (float)atof(value.c_str());

    }// over my_arr

    count += 1;
  }


  in.close();
  *train_length = count;

}

void SelfReferenceStruct::show_data(data_self_ref **p, data_self_ref **start,
				    uint8_t head){

  /*
  p : データを格納した構造体
  start : pの先頭アドレス
  */

  uint8_t num_head;
  if (head) num_head = 10;

  uint32_t j;
  uint32_t cnt = 0;
  *p = *start; //pに先頭アドレスをセット
  
  while (*p != NULL) {
    printf("%d\t%d\n", (*p)->click, (*p)->feature_length);
    
    for (j=0; j<(*p)->feature_length; j++) {
      printf("%d:%f ", (*p)->featureid_score[j].featureid,
             (*p)->featureid_score[j].score);
    }
    printf("\n");

    if ( (head) && (cnt > num_head) )
      break;
    
    (*p) = (*p)->next;
    cnt += 1;
  }

}

void SelfReferenceStruct::free_data(data_self_ref **p, data_self_ref **start){
  
  *p = *start;//pに先頭アドレスをセット
  data_self_ref *tmp;

  while (*p != NULL) {
    free( (*p)->featureid_score );
    tmp = (*p)->next;
    free(*p);
    *p = tmp;
  }

}



/* --------------------------- Array Struct --------------------------- */
ArrayStruct::~ArrayStruct(void)
{}


ArrayStruct::ArrayStruct(arg_params* _cli_params):
  cli_params(_cli_params)
{} 

uint32_t ArrayStruct::get_max(uint32_t a, uint32_t b)
{
  if (a > b) {
    return a;
  } else {
    return b;
  }
}

uint32_t ArrayStruct::get_min(uint32_t a, uint32_t b)
{
  if (a > b) {
    return b;
  } else {
    return a;
  }
}

uint32_t ArrayStruct::get_data_length(const char *filename)
{
  //read file
  std::string tmp_string; tmp_string.clear();
  tmp_string = char2string(filename);
  ifstream in(tmp_string);
  string s;

  if ( !in.is_open() ){
    printf("Can not open %s\n", filename);
    exit(1);
  }

  uint32_t count = 0;
  while (std::getline(in, s)) {
    istringstream iss(s);
    string line;
    //iss >> line;
    count += 1;
  }
  in.close();


  return count;
}

uint32_t ArrayStruct::get_feature_length(const char *filename, 
					 string line_delimiter, 
					 string line_delimiter_between, 
					 string line_delimiter_within)
{

  std::string tmp_string; tmp_string.clear();
  tmp_string = char2string(filename);
  std::string line_buffer;

  std::ifstream in;
  in.open(filename, std::ios::in);

  if ( !in.is_open() ){
    printf("Can not open %s\n", filename);
    exit(1);
  }

  uint32_t max_feature_id = 0; uint32_t min_feature_id = 100000;
  uint32_t feature_id, feature_length;
  float feature_score;
  vector<string> my_arr, my_arr2;
  while ( std::getline(in, line_buffer) ) {
    // trim
    boost::trim_right(line_buffer);
    my_arr.clear();
    //split(line_buffer, line_delimiter, my_arr);
    boost::split(my_arr, line_buffer, boost::is_any_of(line_delimiter));
    feature_length = (uint32_t)(my_arr.size() - 1);// -1 is label, -1 1:0.1 2:0.4
    string key, value;
    for (uint32_t j = 1; j < feature_length + 1;  ++j) {
      my_arr2.clear();
      boost::split(my_arr2, my_arr[j], boost::is_any_of(line_delimiter_between));
      key.clear(); key = my_arr2[0]; feature_id = (uint32_t)atoi(key.c_str());
      value.clear(); value = my_arr2[1]; feature_score = atof(value.c_str());
      max_feature_id = get_max(max_feature_id, feature_id);
      min_feature_id = get_min(min_feature_id, feature_id);
    }

  }
  in.close();

  if (min_feature_id == 0)
    max_feature_id += 1;

  return max_feature_id;
}

void ArrayStruct::init_data(uint32_t data_size)
{
  seq = (data_array*)malloc( sizeof(data_array) * data_size );
}

void ArrayStruct::load_data(char *filename, 
			    std::string delimiter, 
			    std::string delimiter_between, 
			    std::string delimiter_within)
{ 
  //read file
  std::string tmp_string; tmp_string.clear();
  tmp_string = char2string(filename);
  std::string line_buffer;

  std::ifstream in;
  in.open(filename, std::ios::in);


  // file except
  if ( !in.is_open() ){
    printf("Can not open %s\n", filename);
    exit(1);
  }


  // main
  uint32_t j, k;
  uint32_t feature_id, feature_length;
  float feature_score; string tmp_key;
  uint32_t cnt=0;
  vector<string> my_arr, my_arr2;
  while ( std::getline(in, line_buffer) ) {

    // trim
    boost::trim_right(line_buffer);
    my_arr.clear();
    //split(line_buffer, line_delimiter, my_arr);
    boost::split(my_arr, line_buffer, boost::is_any_of(delimiter));

    // data_arrayに新規データを追加, featureidとscoreの領域を確保
    tmp_key.clear(); tmp_key = my_arr[0]; 
    seq[cnt].click = atoi(tmp_key.c_str());
    seq[cnt].feature_length = (uint32_t)(my_arr.size() - 1);
    seq[cnt].featureid_score = (_featureid_score*)malloc(sizeof(_featureid_score) * seq[cnt].feature_length);

    std::string key, value;
    for (uint32_t j = 1; j < my_arr.size();  ++j) {
      my_arr2.clear();
      boost::split(my_arr2, my_arr[j], boost::is_any_of(delimiter_between));
      key.clear(); key = my_arr2[0];
      value.clear(); value = my_arr2[1];

      //配列記法
      seq[cnt].featureid_score[j-1].featureid = (uint32_t)atoi(key.c_str());
      seq[cnt].featureid_score[j-1].score = (float)atof(value.c_str());
    }// over my_arr

    cnt += 1;

  }


  in.close();

}

void ArrayStruct::show_data(uint32_t data_size, uint8_t head){

  /*
  p : データを格納した構造体
  start : pの先頭アドレス
  */

  uint8_t num_head;
  if (head) num_head = 10;
  uint32_t cnt=0;

  for (uint32_t j = 0; j < data_size; ++j) {

    for (uint32_t k = 0; k < seq[j].feature_length; ++k)
      fprintf(stdout, "%d:%f ", seq[j].featureid_score[k].featureid, seq[j].featureid_score[k].score);
    fprintf(stdout, "\n");

    if ( (head) && (cnt > num_head) )
      break;
    cnt += 1;

  } 

}

void ArrayStruct::show_num_line(uint32_t data_size, uint8_t random, uint8_t head)
{

  uint8_t num_head;
  if (head) num_head = 10;
  uint32_t cnt=0;

  if (random) {

    // set random generator
    const gsl_rng_type *T;
    gsl_rng *rng;
    unsigned long int seed;
    T = gsl_rng_mt19937;
    rng = gsl_rng_alloc(T);
    seed = 1;
    gsl_rng_set(rng, seed);


    // show
    for (uint32_t j = 0; j < data_size; ++j) {
      uint32_t cnt_up = gsl_rng_uniform_int(rng, data_size);
      fprintf(stdout, "%u  ", cnt_up);
      for (uint32_t k = 0; k < seq[cnt_up].feature_length; ++k)
	fprintf(stdout, "%d:%f ", seq[cnt_up].featureid_score[k].featureid, seq[cnt_up].featureid_score[k].score);
      fprintf(stdout, "\n");
      
      if ( (head) && (cnt > num_head) )
	break;
      cnt += 1;
    }

  }

}

void ArrayStruct::free_data(uint32_t data_size)
{

  for (uint32_t j = 0; j < data_size; ++j)
    free( seq[j].featureid_score );
  free(seq);

}

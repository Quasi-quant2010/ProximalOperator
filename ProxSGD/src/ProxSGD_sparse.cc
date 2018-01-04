#include "ProxSGD_sparse.h"


PROX_SGD::PROX_SGD(uint32_t _N, uint32_t _D,
		   uint32_t _V,
		   arg_params* _option_args):
  /*
     N : Num of train
     D : Num of feature
     V : Num of validation
   */
  N(_N), 
  D(_D),
  V(_V),
  option_args(_option_args)
/*
  E(RowVectorXf::Zero(_D+1))   // adaptive learning rate initialize,
                               // diagonal matrix. Therefore, VectorXd
                               //  feature-weight-dim + weight-level
                               //          _D         +      1
*/
{}


void PROX_SGD::train(gsl_rng *r, FILE *fname, 
		     data_array *seq, data_array *valid,
		     unordered_map<uint32_t, float> &w, float *w_level)
{
  // PROX_SGD
  Logging("[Proximal Stochastic Gradient Descent with L1]");
  //Logging("CurIter\tBoolCount\tLoss\tNonZeros");
  fprintf(stdout, "CurIter\tBoolCount\tTrainLoss\tValidLoss\tNonZeros\n");
  opt_sparse(r, fname,
	     seq, valid,
	     w, w_level);
}

void PROX_SGD::init_feature_weight(unordered_map<uint32_t, float>& _w, uint32_t len, uint8_t bool_zero)
{

  float tmp = 1.0f;
  if (bool_zero)
    tmp = 0.0f;

  for (uint32_t j=0; j < len; ++j) {
    unordered_map<uint32_t, float>::iterator iter = _w.find(j);
    if (iter == _w.end())
      _w[j] = tmp / (float)(len + 1);; //keyが存在しない
  }
}

uint32_t PROX_SGD::CntNonZero(unordered_map<uint32_t, float>& _w)
{
  uint32_t cnt = 0;
  unordered_map<uint32_t, float>::iterator iter = _w.begin();
  for (; iter != _w.end(); ++iter)
    if (iter->second > 0.0f) 
      cnt += 1;
  return cnt;
}

void PROX_SGD::show_feature_weight(unordered_map<uint32_t, float>& _w)
{
  unordered_map<uint32_t, float>::iterator iter = _w.begin();
  for (; iter != _w.end(); ++iter) 
    fprintf(stdout, "%u:%f\n", iter->first , iter->second);
}

void PROX_SGD::init_vector(float* a, uint32_t len)
{
  for (uint32_t i=0; i < len; i++)
    a[i] = 0.0;
}



/* ---------------- Main ------------------------- */
void PROX_SGD::opt_sparse(gsl_rng* _r, FILE *_fname,
			  data_array*_seq, data_array*_valid,
			  unordered_map<uint32_t, float>& _w, float *_w_level)
{
  float before_loss, after_loss, cur_loss_rate, train_loss;
  uint16_t cur_iter;
  float inner_product_, pred;
  uint8_t bool_count = 0;

  gsl_rng_type *T = (gsl_rng_type *)gsl_rng_mt19937; // random generator
  gsl_rng *r = gsl_rng_alloc(T);                     // random gererator pointer
  gsl_rng_set (r, time(NULL));                       // initialize seed for random generator by sys clock

  uint32_t mini_batch = (uint32_t)(this->option_args->mini_batch_rate * this->N);
  if (mini_batch < 1)
    mini_batch = 1;

  // select loss func
  if (strcmp(this->option_args->loss, "logistic") == 0) {
    LogLikelihood  = &PROX_SGD::LogisticLoss;
    ComputeSGDGrad = &PROX_SGD::ComputeSGDGrad_Logistic;
    }
  else if (strcmp(this->option_args->loss, "squared") == 0) {
    LogLikelihood  = &PROX_SGD::SquaredLoss;
    ComputeSGDGrad = &PROX_SGD::ComputeSGDGrad_Squared;
  }

  // init
  cur_loss_rate = this->option_args->convergence_threshold + 1.0;
  before_loss = 1.0;
  cur_iter = 0;
  after_loss = (this->*LogLikelihood)(_valid, this->V, _w, _w_level);
  train_loss = (this->*LogLikelihood)(_seq, this->V, _w, _w_level);
  fprintf(stdout, "%d\t%d\t%f\t%f\t%u\n", cur_iter, bool_count, train_loss, after_loss, CntNonZero(_w));
  while (bool_count < this->option_args->convergence_threshold_count_train) {

    // 1. Sampling mini-batch data from train datas
    size_t *random_idx = (size_t *)malloc(sizeof(size_t) * mini_batch);
    for (size_t i = 0; i < mini_batch; ++i)
      random_idx[i] = gsl_rng_uniform_int(r, this->N);

    // 2. gradient descent
    //  2.1 calculate gradient for feature weight and weight level
    unordered_map<uint32_t, float> w_grads;
    init_feature_weight(w_grads, this->D, 1);
    float w_level_grad = 0.0f;
    (this->*ComputeSGDGrad)(_seq,
			    _w, _w_level,
			    w_grads, &w_level_grad,
			    random_idx, mini_batch);

    //  2.2 update for feature weight and weight level
    UpdateWeight(w_grads, &w_level_grad,
		 _w, _w_level);


    // 3. likelihood
    after_loss = (this->*LogLikelihood)(_valid, this->V, _w, _w_level);
    train_loss = (this->*LogLikelihood)(_seq, this->V, _w, _w_level);
    //fprintf(_fname, "%d\t%d\t%f\n", cur_iter, bool_count, after_loss);


    // 4. next iteration bool
    if (cur_iter == this->option_args->max_iter) break;
    //cur_loss_rate = (float)(fabs(before_loss - after_loss) / fabs(before_loss));
    cur_loss_rate = (before_loss - after_loss) / before_loss;
    if (cur_loss_rate <= this->option_args->convergence_threshold)
      bool_count += 1;
    else
      bool_count = 0;


    // 5. next iteration update
    before_loss = after_loss;
    cur_iter += 1;
    //Logging("%d\t%d\t%f\t%u", cur_iter, bool_count, after_loss, CntNonZero(_w));
    fprintf(stdout, "%d\t%d\t%f\t%f\t%u\n", cur_iter, bool_count, train_loss, after_loss, CntNonZero(_w));
  }// over while

}// over mini_batch_train



/* ---------------- PROX_SGD ------------------------- */
float PROX_SGD::ProximalOperator(float u, float lambda)
{
  /*
    Naive Proximal Operator
  if (u > t)
    return u - t;
  else if (u < -t)
    return u + t;
  else
    return 0.0f;
  */
  return get_max(0.0f, u-lambda) - get_max(0.0f, -u-lambda);
}

void PROX_SGD::NonUniformTable(float* _prob)
{
  table = (uint16_t *)malloc(sizeof(uint16_t) * this->option_args->table_size);
  float cum_p = 0.0f;
  uint16_t table_idx = 0;

  fprintf(stderr, "Making Non-Uniform Table\n");
  for (uint16_t t = 0; t < this->option_args->m; ++t) {
    cum_p += _prob[t];
    while ( (table_idx < this->option_args->table_size) && \
	    (((float)table_idx / (float)this->option_args->table_size) < cum_p) ) {
      table[table_idx] = t;
      table_idx += 1;
    }
  }
}

void PROX_SGD::get_NonUniformProb(float* prob)
{
  float z = 0.0f;
  for (uint16_t t = 0; t < this->option_args->m; ++t) {
    prob[t] = pow(1.0f - this->option_args->nu * this->option_args->step_size, 
		  this->option_args->m - (t+1));
    z += prob[t];
  }

  fprintf(stderr, "[Non-Uniform Empirical Probablity Sequences]\n\t");
  for (uint16_t t = 0; t < this->option_args->m; ++t) {
    prob[t] /= z;
    fprintf(stderr, "%1.3e ", prob[t]);
  }
  fprintf(stderr, "\n");
}


float PROX_SGD::get_learnig_rate(const arg_params* _option_args,
				 float _cur_iter)
{
  float learning_rate = 0.0;

  learning_rate = _option_args->step_size / sqrt(_cur_iter);
  
  return learning_rate;
}


void PROX_SGD::ComputeSGDGrad_Logistic(data_array* seqs,
				       unordered_map<uint32_t, float>& _w, float* _w_level,
				       unordered_map<uint32_t, float>& _w_sgd_grads, float* _w_level_sgd_grad,
				       size_t* _random_idx, uint32_t _mini_batch)
{
  float inner_product_, pred;

  // calculate gradient for feature weight
  for (size_t i=0; i < _mini_batch; ++i) {
    inner_product_ = 0.0f; pred = 0.0f;
    inner_product_ = inner_product(&seqs, _random_idx[i], _w);
    inner_product_ += *_w_level;
    inner_product_ *= (float)seqs[_random_idx[i]].click;                      // logistic loss
    pred = sigmoid(inner_product_);
    
    // for feature weight
    for (uint32_t k = 0; k < seqs[i].feature_length; ++k) {
      size_t tmp_fid   = seqs[_random_idx[i]].featureid_score[k].featureid;
      float tmp_fscore = seqs[_random_idx[i]].featureid_score[k].score;
      unordered_map<uint32_t, float>::iterator iter = _w_sgd_grads.find(tmp_fid);
      if (iter != _w_sgd_grads.end())
	iter->second -= pred * (float)seqs[_random_idx[i]].click * tmp_fscore; // exist key
    }
    
    // for feature weight level
    *_w_level_sgd_grad -= pred * (float)seqs[_random_idx[i]].click;
  }

  // normalize
  unordered_map<uint32_t, float>::iterator iter = _w_sgd_grads.begin();
  for (; iter != _w_sgd_grads.end(); ++iter) 
    iter->second /= (float)_mini_batch;
  *_w_level_sgd_grad /= (float)_mini_batch;
}

void PROX_SGD::ComputeSGDGrad_Squared(data_array* seqs,
				      unordered_map<uint32_t, float>& _w, float* _w_level,
				      unordered_map<uint32_t, float>& _w_sgd_grads, float* _w_level_sgd_grad,
				      size_t* _random_idx, uint32_t _mini_batch)
{
  float inner_product_, pred;

  // calculate gradient for feature weight
  for (size_t i=0; i < _mini_batch; ++i) {
    // for feature weight
    for (uint32_t k = 0; k < seqs[i].feature_length; ++k) {
      size_t tmp_fid   = seqs[_random_idx[i]].featureid_score[k].featureid;
      float tmp_fscore = seqs[_random_idx[i]].featureid_score[k].score;
      unordered_map<uint32_t, float>::iterator iter = _w_sgd_grads.find(tmp_fid);
      if (iter != _w_sgd_grads.end())
	iter->second -= (float)seqs[_random_idx[i]].click * tmp_fscore; // exist key
    }
    
    // for feature weight level
    *_w_level_sgd_grad -= (float)seqs[_random_idx[i]].click;
  }

  // normalize
  unordered_map<uint32_t, float>::iterator iter = _w_sgd_grads.begin();
  for (; iter != _w_sgd_grads.end(); ++iter) 
    iter->second /= (float)_mini_batch;
  *_w_level_sgd_grad /= (float)_mini_batch;
}

void PROX_SGD::UpdateWeight(unordered_map<uint32_t, float>& w_grads_, float* w_level_grad_,
			    unordered_map<uint32_t, float>& w_, float* w_level_)
{
  // update feature weight
  unordered_map<uint32_t, float>::iterator iter = w_.begin();
  for (; iter != w_.end(); ++iter) {
    float grad  = 0.0f; 
    size_t f_id = iter->first;
    float  f_weight = iter->second;
    unordered_map<uint32_t, float>::iterator tmp_iter = w_grads_.find(f_id);
    if (tmp_iter != w_grads_.end()) {
      // key exist
      grad = tmp_iter->second;
      float u = 0.0f, lambda_ = 0.0f, learning_rate = 0.0f;
      learning_rate = this->option_args->step_size;
      lambda_       = learning_rate * this->option_args->lambda;
      u             = f_weight - learning_rate * grad;
      iter->second  = ProximalOperator(u, lambda_);                    // update
    }
  }
  
  // update feature weight level
  float grad = 0.0f;
  grad = *w_level_grad_;
  float u = 0.0f, lambda_ = 0.0f, learning_rate = 0.0f;
  learning_rate = this->option_args->step_size;
  lambda_       = learning_rate * this->option_args->lambda;
  u             = *w_level_ - learning_rate * grad;
  *w_level_     = ProximalOperator(u, lambda_);                        // update
}

float PROX_SGD::get_max(float a, float b)
{
  if (a > b) {
    return a;
  } else {
    return b;
  }
}

float PROX_SGD::get_min(float a, float b)
{
  if (a > b) {
    return b;
  } else {
    return a;
  }
}

 
/* ---------------- Loss Function ------------------- */
float PROX_SGD::LogisticLoss(data_array* __seq, uint32_t length,
			     unordered_map<uint32_t, float> &__w, float *__w_level)
{
  float loss = 0.0f;
  for (uint32_t i = 0; i < length; i++) {
    
    float inner_product_ = 0.0f;
    inner_product_ = inner_product(&__seq, i, __w);
    inner_product_ += *__w_level; 
    inner_product_ *= (float)__seq[i].click;
    loss += log( 1.0f + exp(-1.0f * inner_product_) );
  }
  return loss / (float)length + this->option_args->lambda * L1Norm(__w);
}

float PROX_SGD::SquaredLoss(data_array* __seq, uint32_t length,
			    unordered_map<uint32_t, float> &__w, float *__w_level)
{
  float loss = 0.0f;
  for (uint32_t i = 0; i < length; i++) {
    
    float inner_product_ = 0.0f;
    inner_product_ = inner_product(&__seq, i, __w);
    inner_product_ += *__w_level; 
    inner_product_ *= (float)__seq[i].click;
    loss += pow(1.0f - inner_product_, 2.0f);
  }
  return loss / (float)length + this->option_args->lambda * L1Norm(__w);
}


/* ---------------- Elementary Function ------------------- */
float PROX_SGD::inner_product(data_array **_seq, uint32_t idx, 
			      unordered_map<uint32_t, float>& fw)
{ 
  uint32_t key;
  float value; 
  float tmp = 0.0;
  for (uint32_t j = 0; j < (*_seq + idx)->feature_length; ++j) {

    // key   : feature id
    // value : feature's score
    key = 0; key = (*_seq + idx)->featureid_score[j].featureid;
    value = 0.0f; value = (*_seq + idx)->featureid_score[j].score;
    unordered_map<uint32_t, float>::iterator w_iter = fw.find(key);
    if (w_iter != fw.end())
      tmp += value * w_iter->second;

  }

  return tmp;
}

float PROX_SGD::L1Norm(unordered_map<uint32_t, float> & _w)
{
  float tmp = 0.0f;
  unordered_map<uint32_t, float>::iterator iter = _w.begin();
  for (; iter != _w.end(); ++iter)
    tmp += fabs(iter->second);
  return tmp;
}

float PROX_SGD::SquaredNorm(float* a, uint32_t dim)
{
  float _sum = 0.0f;
  for (uint32_t j=0; j < dim; ++j)
    _sum += pow(a[j], 2.0);
  //show_feature_weight(a, dim);
  return _sum;
}

float PROX_SGD::sigmoid(float z)
{
  if (z >  6.) {
    return 1.0;
  } else if (z < -6.) {
    return 0.0;
  } else {
    return 1.0 / (1 + exp(-z));
  }
}



/*
float PROX_SGD::Acc(vector<float>& pred,VectorXd &l){
  int t =0;
  float loss =0;
  for(int i = 0; i< pred.size();i++){
    loss += (l(i) - pred[i]) * (l(i) - pred[i]);
    int s = pred[i] > 0.5 ? 1 : 0;
    if(s == l(i)){
      t++;
    }
  }
  //return (float)t/pred.size();
  return loss / pred.size();
}

void PROX_SGD::predict(MatrixXd& _x,VectorXd& _l,vector<float>& ret){
  for(int i = 0; i < _x.rows(); i++){
    float inner_product_ = inner_product(_x, i);
    float pred = sigmoid(inner_product_ + this->w_level);
    ret.push_back(pred);
  }
}
*/

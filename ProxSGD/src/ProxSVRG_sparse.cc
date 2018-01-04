#include "ProxSVRG_sparse.h"


PROX_SVRG::PROX_SVRG(uint32_t _N, uint32_t _D,
		     uint32_t _V,
		     arg_params* _option_args):
  /*
     N : Num of data
     D : Num of feature
   */
  N(_N), 
  D(_D),
  V(_V),
  option_args(_option_args)
{}


void PROX_SVRG::train(gsl_rng *r, FILE *fname, 
		      data_array *seq, data_array *valid,
		      unordered_map<uint32_t, float> &w, float *w_level)
{
  // PROX_SVRG
  Logging("[Proximal SVRG with L1]");
  // Init
  steps_prob = (float *)malloc(sizeof(float) * this->option_args->m);
  init_vector(steps_prob, (uint32_t)this->option_args->m);
  get_NonUniformProb(steps_prob);
  NonUniformTable(steps_prob);
  gsl_ran_shuffle(r, table, this->option_args->table_size, sizeof(uint16_t));
  
  //Logging("CurIter\tBoolCount\tLoss\tNonZeros");
  fprintf(stdout, "CurIter\tBoolCount\tTrainLoss\tValidLoss\tNonZeros\n");
  opt_sparse(r, fname,
	     seq, valid,
	     w, w_level);
}

void PROX_SVRG::init_feature_weight(unordered_map<uint32_t, float>& _w, uint32_t len, uint8_t bool_zero)
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

uint32_t PROX_SVRG::CntNonZero(unordered_map<uint32_t, float>& _w)
{
  uint32_t cnt = 0;
  unordered_map<uint32_t, float>::iterator iter = _w.begin();
  for (; iter != _w.end(); ++iter)
    if (iter->second > 0.0f) 
      cnt += 1;
  return cnt;
}

void PROX_SVRG::show_feature_weight(unordered_map<uint32_t, float>& _w)
{
  unordered_map<uint32_t, float>::iterator iter = _w.begin();
  for (; iter != _w.end(); ++iter) 
    fprintf(stdout, "%u:%f\n", iter->first , iter->second);
}

void PROX_SVRG::init_vector(float* a, uint32_t len)
{
  for (uint32_t i=0; i < len; i++)
    a[i] = 0.0;
}


/* ---------------- Main ------------------------- */
void PROX_SVRG::opt_sparse(gsl_rng* _r, FILE *_fname,
			   data_array*_seq, data_array*_valid,
			   unordered_map<uint32_t, float>& _w, float *_w_level)
{
  float before_loss, after_loss, cur_loss_rate, train_loss;
  uint16_t outer_loop, inner_loop;
  uint8_t bool_count = 0;

  gsl_rng_type *T = (gsl_rng_type *)gsl_rng_mt19937; // random generator
  gsl_rng *r = gsl_rng_alloc(T);                     // random gererator pointer
  gsl_rng_set (r, time(NULL));                       // initialize seed for random generator by sys clock

  uint32_t mini_batch = (uint32_t)(this->option_args->mini_batch_rate * this->N);
  if (mini_batch < 1)
    mini_batch = 1;

  // select loss and gradient func
  if (strcmp(this->option_args->loss, "logistic") == 0) {
    LogLikelihood          = &PROX_SVRG::LogisticLoss;
    ComputeFullGrad        = &PROX_SVRG::ComputeFullGrad_Logistic;
    ComputeGradEstimator   = &PROX_SVRG::ComputeGradEstimator_Logistic;
  }
  else if (strcmp(this->option_args->loss, "squared") == 0) {
    LogLikelihood          = &PROX_SVRG::SquaredLoss;
    ComputeFullGrad        = &PROX_SVRG::ComputeFullGrad_Squared;
    ComputeGradEstimator   = &PROX_SVRG::ComputeGradEstimator_Squared;
  }
  else {
    LogLikelihood          = &PROX_SVRG::LogisticLoss;
    ComputeFullGrad        = &PROX_SVRG::ComputeFullGrad_Logistic;
    ComputeGradEstimator   = &PROX_SVRG::ComputeGradEstimator_Logistic;
  }

  // init
  cur_loss_rate = this->option_args->convergence_threshold + 1.0;
  before_loss = 1.0;
  outer_loop = 0;
  after_loss = (this->*LogLikelihood)(_valid, this->V, _w, _w_level);
  train_loss = (this->*LogLikelihood)(_seq, this->V, _w, _w_level);
  fprintf(stdout, "%d\t%d\t%f\t%f\t%u\n", outer_loop, bool_count, train_loss, after_loss, CntNonZero(_w));
  // outer loop
  while (bool_count < this->option_args->convergence_threshold_count_train) {

    // 1. calculate full gradient : \tilda{\mu_w}1w and \tilda{\mu_w_level}
    unordered_map<uint32_t, float> w_full_grads;
    init_feature_weight(w_full_grads, this->D, 1);
    float w_level_full_grad = 0.0f;
    (this->*ComputeFullGrad)(_seq, 
			     _w, _w_level,
			     w_full_grads, &w_level_full_grad);


    // inner loop
    // 2. sampling the number of the inner loop by non-uniform distribution
    uint16_t num_inner_loop = 0;
    num_inner_loop = gsl_rng_uniform_int(r, this->option_args->table_size);
    num_inner_loop = table[num_inner_loop];  // with probability (1-nu * step_size)^(m-t) / beta)
    if (this->option_args->update_option == 2)
      num_inner_loop = gsl_rng_uniform_int(r, num_inner_loop+1);

    // 3. inner loop
    unordered_map<uint32_t, float> w_tmp;
    init_feature_weight(w_tmp, this->D, 1);
    copy_feature_weight(_w, w_tmp);
    float w_level_tmp = *_w_level;

    inner_loop = 0;
    while (inner_loop < num_inner_loop) {

      // 3.1 select single sample uniformly
      uint32_t idx = gsl_rng_uniform_int(r, this->N);

      // 3.2 calculate gradient estimator
      unordered_map<uint32_t, float> w_grad_estimator;
      init_feature_weight(w_grad_estimator, this->D, 1);
      float w_level_grad_estimator = 0.0f;
      (this->*ComputeGradEstimator)(_seq, idx,
				    _w, _w_level,
				    w_tmp, &w_level_tmp,
				    w_full_grads, &w_level_full_grad,
				    w_grad_estimator, &w_level_grad_estimator);
      
      // 3.3 update
      UpdateWeight_InnerLoop(w_tmp, &w_level_tmp,
			     w_grad_estimator, &w_level_grad_estimator);
      inner_loop += 1;
    } // over inner loop

    // update outer loop
    UpdateWeight_OuterLoop(w_tmp, &w_level_tmp,
			   _w, _w_level);

    // 3. likelihood
    after_loss = (this->*LogLikelihood)(_valid, this->V, _w, _w_level);
    train_loss = (this->*LogLikelihood)(_seq, this->V, _w, _w_level);
    //fprintf(_fname, "%d\t%d\t%f\n", outer_loop, bool_count, after_loss);
    outer_loop += 1;
    fprintf(stdout, "%d\t%d\t%f\t%f\t%u\n", outer_loop, bool_count, train_loss, after_loss, CntNonZero(_w));

    // 4. next iteration bool
    if (outer_loop == this->option_args->max_iter) break;
    cur_loss_rate = (before_loss - after_loss) / before_loss;
    if (cur_loss_rate <= this->option_args->convergence_threshold)
      bool_count += 1;
    else
      bool_count = 0;


    // 5. next iteration update
    before_loss = after_loss;
    //Logging("%d\t%d\t%f\t%u", outer_loop, bool_count, after_loss, CntNonZero(_w));
  }// over while

}// over mini_batch_train



/* ---------------- PROX_SVRG ------------------------- */
float PROX_SVRG::ProximalOperator(float u, float lambda)
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

void PROX_SVRG::NonUniformTable(float* _prob)
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

void PROX_SVRG::get_NonUniformProb(float* prob)
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


float PROX_SVRG::get_learnig_rate(const arg_params* _option_args,
				  float _cur_iter)
{
  float learning_rate = 0.0;

  learning_rate = _option_args->step_size / sqrt(_cur_iter);
  
  return learning_rate;
}

void PROX_SVRG::ComputeFullGrad_Logistic(data_array* seqs, 
					 unordered_map<uint32_t, float>& _w, float* _w_level,
					 unordered_map<uint32_t, float>& _w_full_grads, float* _w_level_full_grad)
{
  float inner_product_, pred;

  for (size_t i=0; i < this->N; ++i) {
    inner_product_ = 0.0f; pred = 0.0f;
    inner_product_ = inner_product(&seqs, i, _w);
    inner_product_ += *_w_level; 
    inner_product_ *= (float)seqs[i].click;                      // logistic loss
    pred = sigmoid(inner_product_);
    
    // for feature weight
    for (uint32_t k = 0; k < seqs[i].feature_length; ++k) {
      size_t tmp_fid   = seqs[i].featureid_score[k].featureid;
      float tmp_fscore = seqs[i].featureid_score[k].score;
      unordered_map<uint32_t, float>::iterator iter = _w_full_grads.find(tmp_fid);
      if (iter != _w_full_grads.end())
	iter->second -= pred * (float)seqs[i].click * tmp_fscore; // exist key
    }
    
    // for feature weight level
    *_w_level_full_grad -= pred * (float)seqs[i].click;
  }

  // normalize
  unordered_map<uint32_t, float>::iterator iter = _w_full_grads.begin();
  for (; iter != _w_full_grads.end(); ++iter) 
    iter->second /= (float)this->N;
  *_w_level_full_grad /= (float)this->N;
}

void PROX_SVRG::ComputeFullGrad_Squared(data_array* seqs, 
					unordered_map<uint32_t, float>& _w, float* _w_level,
					unordered_map<uint32_t, float>& _w_full_grads, float* _w_level_full_grad)
{
  float inner_product_, pred;

  for (size_t i=0; i < this->N; ++i) {
    // for feature weight
    for (uint32_t k = 0; k < seqs[i].feature_length; ++k) {
      size_t tmp_fid   = seqs[i].featureid_score[k].featureid;
      float tmp_fscore = seqs[i].featureid_score[k].score;
      unordered_map<uint32_t, float>::iterator iter = _w_full_grads.find(tmp_fid);
      if (iter != _w_full_grads.end())
	iter->second -= (float)seqs[i].click * tmp_fscore; // exist key
    }
    
    // for feature weight level
    *_w_level_full_grad -= (float)seqs[i].click;
  }

  // normalize
  unordered_map<uint32_t, float>::iterator iter = _w_full_grads.begin();
  for (; iter != _w_full_grads.end(); ++iter) 
    iter->second /= (float)this->N;
  *_w_level_full_grad /= (float)this->N;
}

void PROX_SVRG::ComputeGradEstimator_Logistic(data_array* seqs, uint32_t _idx,
					      unordered_map<uint32_t, float> &pre_w, float *pre_w_level,
					      unordered_map<uint32_t, float> &cur_w, float *cur_w_level,
					      unordered_map<uint32_t, float> &_w_full_grad, float *_w_level_full_grad,
					      unordered_map<uint32_t, float> &_w_grad_estimator, float *_w_level_grad_estimator)
{
  float inner_product_, pred;

  unordered_map<uint32_t, float> w_grad_cur;   // w_grad_cur : s-th outer loop and k-th inner loop
  float w_level_grad_cur=0.0f;
  unordered_map<uint32_t, float> w_grad_pre;   // w_grad_cur : (s-1)-th outer loop
  float w_level_grad_pre=0.0f;
  init_feature_weight(w_grad_cur, this->D, 1);
  init_feature_weight(w_grad_pre, this->D, 1);

  // w_grad_cur
  inner_product_ = 0.0f; pred = 0.0f;
  inner_product_ = inner_product(&seqs, _idx, cur_w);
  inner_product_ += *cur_w_level; 
  inner_product_ *= (float)seqs[_idx].click;                      // logistic loss
  pred = sigmoid(inner_product_);
  for (uint32_t k = 0; k < seqs[_idx].feature_length; ++k) {
      size_t tmp_fid   = seqs[_idx].featureid_score[k].featureid;
      float tmp_fscore = seqs[_idx].featureid_score[k].score;
      unordered_map<uint32_t, float>::iterator iter = w_grad_cur.find(tmp_fid);
      if (iter != w_grad_cur.end())
	iter->second -= pred * (float)seqs[_idx].click * tmp_fscore; // exist key
  }
  w_level_grad_cur -= pred * (float)seqs[_idx].click;

  // w_grad_pre
  inner_product_ = 0.0f; pred = 0.0f;
  inner_product_ = inner_product(&seqs, _idx, pre_w);
  inner_product_ += *pre_w_level; 
  inner_product_ *= (float)seqs[_idx].click;                      // logistic loss
  pred = sigmoid(inner_product_);
  for (uint32_t k = 0; k < seqs[_idx].feature_length; ++k) {
      size_t tmp_fid   = seqs[_idx].featureid_score[k].featureid;
      float tmp_fscore = seqs[_idx].featureid_score[k].score;
      unordered_map<uint32_t, float>::iterator iter = w_grad_pre.find(tmp_fid);
      if (iter != w_grad_pre.end())
	iter->second -= pred * (float)seqs[_idx].click * tmp_fscore; // exist key
  }
  w_level_grad_pre -= pred * (float)seqs[_idx].click;

  // gradient estimator
  //  _w_grad_estimator
  unordered_map<uint32_t, float>::iterator iter = _w_grad_estimator.begin();
  for (; iter != _w_grad_estimator.end(); ++iter) {
    // w_grad_cur
    unordered_map<uint32_t, float>::iterator w_grad_cur_iter = w_grad_cur.find(iter->first);
    if (w_grad_cur_iter != w_grad_cur.end())
      iter->second = w_grad_cur_iter->second;

    // w_grad_pre
    unordered_map<uint32_t, float>::iterator w_grad_pre_iter = w_grad_pre.find(iter->first);
    if (w_grad_pre_iter != w_grad_pre.end())
      iter->second -= w_grad_pre_iter->second;

    // full gradient
    unordered_map<uint32_t, float>::iterator _w_full_grad_iter = _w_full_grad.find(iter->first);
    if (_w_full_grad_iter != _w_full_grad.end())
      iter->second += _w_full_grad_iter->second;
  }
  //  _w_level_grad_estimator
  *_w_level_grad_estimator = *cur_w_level - *pre_w_level + *_w_level_full_grad;
}

void PROX_SVRG::ComputeGradEstimator_Squared(data_array* seqs, uint32_t _idx,
					     unordered_map<uint32_t, float> &pre_w, float *pre_w_level,
					     unordered_map<uint32_t, float> &cur_w, float *cur_w_level,
					     unordered_map<uint32_t, float> &_w_full_grad, float *_w_level_full_grad,
					     unordered_map<uint32_t, float> &_w_grad_estimator, float *_w_level_grad_estimator)
{
  float inner_product_, pred;

  unordered_map<uint32_t, float> w_grad_cur;   // w_grad_cur : s-th outer loop and k-th inner loop
  float w_level_grad_cur=0.0f;
  unordered_map<uint32_t, float> w_grad_pre;   // w_grad_cur : (s-1)-th outer loop
  float w_level_grad_pre=0.0f;
  init_feature_weight(w_grad_cur, this->D, 1);
  init_feature_weight(w_grad_pre, this->D, 1);

  // w_grad_cur
  for (uint32_t k = 0; k < seqs[_idx].feature_length; ++k) {
      size_t tmp_fid   = seqs[_idx].featureid_score[k].featureid;
      float tmp_fscore = seqs[_idx].featureid_score[k].score;
      unordered_map<uint32_t, float>::iterator iter = w_grad_cur.find(tmp_fid);
      if (iter != w_grad_cur.end())
	iter->second -= (float)seqs[_idx].click * tmp_fscore; // exist key
  }
  w_level_grad_cur -= (float)seqs[_idx].click;

  // w_grad_pre
  for (uint32_t k = 0; k < seqs[_idx].feature_length; ++k) {
      size_t tmp_fid   = seqs[_idx].featureid_score[k].featureid;
      float tmp_fscore = seqs[_idx].featureid_score[k].score;
      unordered_map<uint32_t, float>::iterator iter = w_grad_pre.find(tmp_fid);
      if (iter != w_grad_pre.end())
	iter->second -= (float)seqs[_idx].click * tmp_fscore; // exist key
  }
  w_level_grad_pre -= (float)seqs[_idx].click;

  // gradient estimator
  //  _w_grad_estimator
  unordered_map<uint32_t, float>::iterator iter = _w_grad_estimator.begin();
  for (; iter != _w_grad_estimator.end(); ++iter) {

    // iter->second = w_grad_cur - w_grad_pre_iter + _w_full_grad_iter

    // w_grad_cur
    unordered_map<uint32_t, float>::iterator w_grad_cur_iter = w_grad_cur.find(iter->first);
    if (w_grad_cur_iter != w_grad_cur.end())
      iter->second = w_grad_cur_iter->second;

    // w_grad_pre
    unordered_map<uint32_t, float>::iterator w_grad_pre_iter = w_grad_pre.find(iter->first);
    if (w_grad_pre_iter != w_grad_pre.end())
      iter->second -= w_grad_pre_iter->second;

    // full gradient
    unordered_map<uint32_t, float>::iterator _w_full_grad_iter = _w_full_grad.find(iter->first);
    if (_w_full_grad_iter != _w_full_grad.end())
      iter->second += _w_full_grad_iter->second;
  }
  //  _w_level_grad_estimator
  *_w_level_grad_estimator = *cur_w_level - *pre_w_level + *_w_level_full_grad;
}

void PROX_SVRG::UpdateWeight_InnerLoop(unordered_map<uint32_t, float> &_w, float *_w_level,
				       unordered_map<uint32_t, float> &_w_grad, float *_w_level_grad)
{
  // feature weight
  unordered_map<uint32_t, float>::iterator iter = _w.begin();
  for (; iter != _w.end(); ++iter) {
    float grad  = 0.0f; 
    size_t f_id = iter->first;
    float  f_weight = iter->second;
    unordered_map<uint32_t, float>::iterator tmp_iter = _w_grad.find(f_id);
    if (tmp_iter != _w_grad.end()) {
      //keyが存在する
      grad = tmp_iter->second;
      float u = 0.0f, lambda_ = 0.0f, learning_rate = 0.0f;
      learning_rate = this->option_args->step_size;
      lambda_       = learning_rate * this->option_args->lambda;
      u             = f_weight - learning_rate * grad;
      iter->second  = ProximalOperator(u, lambda_);                    // update
    }
  }

  // feture weight level
  float grad = 0.0f;
  grad = *_w_level_grad;
  float u = 0.0f, lambda_ = 0.0f, learning_rate = 0.0f;
  learning_rate = this->option_args->step_size;
  lambda_       = learning_rate * this->option_args->lambda;
  u             = *_w_level - learning_rate * grad;
  *_w_level     = ProximalOperator(u, lambda_);                        // update
}

void PROX_SVRG::UpdateWeight_OuterLoop(unordered_map<uint32_t, float> &w_source, float *w_level_source,
				       unordered_map<uint32_t, float> &w_target, float *w_level_target)
{
  // Updatee Option 1
  if (this->option_args->update_option == 1)
    copy_feature_weight(w_source, w_target);
  *w_level_target = *w_level_source;
}

float PROX_SVRG::get_max(float a, float b)
{
  if (a > b) {
    return a;
  } else {
    return b;
  }
}

float PROX_SVRG::get_min(float a, float b)
{
  if (a > b) {
    return b;
  } else {
    return a;
  }
}

 
/* ---------------- Loss Function ------------------- */
float PROX_SVRG::LogisticLoss(data_array* __seq, uint32_t length,
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

float PROX_SVRG::SquaredLoss(data_array* __seq, uint32_t length,
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
float PROX_SVRG::inner_product(data_array **_seq, uint32_t idx, 
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

float PROX_SVRG::L1Norm(unordered_map<uint32_t, float> & _w)
{
  float tmp = 0.0f;
  unordered_map<uint32_t, float>::iterator iter = _w.begin();
  for (; iter != _w.end(); ++iter)
    tmp += fabs(iter->second);
  return tmp;
}

float PROX_SVRG::SquaredNorm(float* a, uint32_t dim)
{
  float _sum = 0.0f;
  for (uint32_t j=0; j < dim; ++j)
    _sum += pow(a[j], 2.0);
  //show_feature_weight(a, dim);
  return _sum;
}

float PROX_SVRG::sigmoid(float z)
{
  if (z >  6.) {
    return 1.0;
  } else if (z < -6.) {
    return 0.0;
  } else {
    return 1.0 / (1 + exp(-z));
  }
}


void PROX_SVRG::copy_feature_weight(unordered_map<uint32_t, float> &source, 
				    unordered_map<uint32_t, float> &target)
{
  unordered_map<uint32_t, float>::iterator iter = source.begin();
  for (; iter != source.end(); ++iter) {
    unordered_map<uint32_t, float>::iterator target_iter = target.find(iter->first);
    if (target_iter != target.end())
      target_iter->second = iter->second;
  }
}



/*
float PROX_SVRG::Acc(vector<float>& pred,VectorXd &l){
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

void PROX_SVRG::predict(MatrixXd& _x,VectorXd& _l,vector<float>& ret){
  for(int i = 0; i < _x.rows(); i++){
    float inner_product_ = inner_product(_x, i);
    float pred = sigmoid(inner_product_ + this->w_level);
    ret.push_back(pred);
  }
}
*/

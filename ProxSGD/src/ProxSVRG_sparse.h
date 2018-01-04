#ifndef PROX_SVRG_H
#define PROX_SVRG_H

#include <math.h>

#include "utils.h"
#include "read_file_sparse.h"

#include <unordered_map>
using namespace std;

class PROX_SVRG{
public:
  arg_params* option_args;     // option_parser's args
  uint16_t* table;
  /* <- option_args
   float lambda;                // regularization
   float eta;                   // learning rate
   float epsilon;               // adagrad's epsilon
   float clip;                  // threshold in clipping
   string clip_method;           // clipping method, clipping, squeese, euclidian
   unsigned int iteration;       // max iteration
   unsigned int mini_batch;      // mini_batch size
   float convergence_threshold; // early stopping
   unsigned int batch_sgd;       // if 1, optimization is SGD
  */

  PROX_SVRG(uint32_t, uint32_t,
	    uint32_t,
	    arg_params*);

  void train(gsl_rng*, FILE*,
	     data_array*, data_array*,
	     unordered_map<uint32_t, float>&, float*);
  void init_feature_weight(unordered_map<uint32_t, float>&, uint32_t, uint8_t bool_zero=0);
  uint32_t CntNonZero(unordered_map<uint32_t, float>&);
  void show_feature_weight(unordered_map<uint32_t, float>&);
  void init_vector(float* a, uint32_t len);
  /*
    gsl_rng* = _r        : 
    FILE* = _fnamae      :
    data_array* = _X, _y : designe matrix
    RowVectorXf* = _w    : feature weight
    float* = _w_level    : feature weight level
   */
  //void predict(MatrixXd& _x,VectorXd& _l,vector<float>& ret);

  ~PROX_SVRG()
    {
      // do nothing
    }

private:
  uint32_t N;                        // the num of samples
  uint32_t D;                        // the num of features
  uint32_t V;                        // the num of validation
  float *steps_prob;                 // 


  /* ---------------- Main ------------------------- */
  void  opt_sparse(gsl_rng* _r, FILE* _fname,
		   data_array* seq, data_array* valid,
		   unordered_map<uint32_t, float>& _w, float* _w_level);

  /* ---------------- PROX_SVRG ------------------------- */
  float ProximalOperator(float u, float lambda);
  void  NonUniformTable(float*);
  void  get_NonUniformProb(float*);
  float get_learnig_rate(const arg_params* _option_args, 
			 float _cur_iter);

  // full gradient
  void  (PROX_SVRG::*ComputeFullGrad)(data_array* seqs, 
				      unordered_map<uint32_t, float>&, float*,
				      unordered_map<uint32_t, float>&, float*);
  void  ComputeFullGrad_Logistic(data_array* seqs, 
				 unordered_map<uint32_t, float>&, float*,
				 unordered_map<uint32_t, float>&, float*);
  void  ComputeFullGrad_Squared(data_array* seqs, 
				unordered_map<uint32_t, float>&, float*,
				unordered_map<uint32_t, float>&, float*);

  // gradient estimator
  void  (PROX_SVRG::*ComputeGradEstimator)(data_array*, uint32_t,
					   unordered_map<uint32_t, float>&, float*,
					   unordered_map<uint32_t, float>&, float*,
					   unordered_map<uint32_t, float>&, float*, 
					   unordered_map<uint32_t, float>&, float*);
  void  ComputeGradEstimator_Logistic(data_array*, uint32_t,
				      unordered_map<uint32_t, float>&, float*,
				      unordered_map<uint32_t, float>&, float*,
				      unordered_map<uint32_t, float>&, float*, 
				      unordered_map<uint32_t, float>&, float*);
  void  ComputeGradEstimator_Squared(data_array*, uint32_t,
				     unordered_map<uint32_t, float>&, float*,
				     unordered_map<uint32_t, float>&, float*,
				     unordered_map<uint32_t, float>&, float*, 
				     unordered_map<uint32_t, float>&, float*);
  void  UpdateWeight_InnerLoop(unordered_map<uint32_t, float>&, float*,
			       unordered_map<uint32_t, float>&, float*);
  void  UpdateWeight_OuterLoop(unordered_map<uint32_t, float>& w_source, float* w_level_source,
			       unordered_map<uint32_t, float>& w_target, float* w_level_target);
  float get_max(float a, float b);
  float get_min(float a, float b);

  /* ---------------- Loss Function ------------------- */
  float (PROX_SVRG::*LogLikelihood)(data_array*, uint32_t,
				    unordered_map<uint32_t, float>&, float*);
  float LogisticLoss(data_array*, uint32_t,
		     unordered_map<uint32_t, float>&, float*);
  float SquaredLoss(data_array*, uint32_t,
		    unordered_map<uint32_t, float>&, float*);

  /* ---------------- Elementary Function ------------------- */
  float inner_product(data_array**, uint32_t, unordered_map<uint32_t, float>&);
  float L1Norm(unordered_map<uint32_t, float>&);
  float SquaredNorm(float*, uint32_t);
  float sigmoid(float z);
  void  copy_feature_weight(unordered_map<uint32_t, float>& source, 
			    unordered_map<uint32_t, float>& target);
};

#endif

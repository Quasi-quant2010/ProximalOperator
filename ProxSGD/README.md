# command
- ProxSGD
	- ./ProxSGD --train ~/data/libsvm/sparse/a1a.train --valid ~/data/libsvm/sparse/a1a.valid --step_size 0.01 --lambda 0.00001 --max_iter 1000 --convergence_threshold_count_train 5 --m 20 --nu 0.0 --loss logistic 1>stdout.log 2>stderr.log;
- ProxSVRG
	- ./ProxSVRG --train ~/data/libsvm/sparse/a1a.train --valid ~/data/libsvm/sparse/a1a.valid --step_size 0.001 --lambda 0.00001 --max_iter 1000 --convergence_threshold_count_train 5 --m 20 --nu 0.0 --loss logistic 1>stdout.log 2>stderr.log;

# Loss
- logistic loss
- squared loss

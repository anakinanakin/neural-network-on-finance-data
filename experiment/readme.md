#### Directories

`2-class_experiment` (**2-class**)/   run with gpu, batch_size=512

​	Label by trading strategy(redmine #3855). 

​	Label each day separately. Delete no trading part in the beginning of each day, leave only 2 class(red: buy, green: sell)



`3-class_experiment` (**3-class**)/

​	Label by close price trend algorithm(redmine #3855: Benchmark Dataset)

​	Label each day separately.

  	  `finished_exp`/

​			`experiment1`(**3exp1**)/	batch_size=32

​				cpu and gpu give different results with random seed fixed. Use gpu as default.

​					`experiment1(cpu)`/ 	run with cpu

​					`experiment1(gpu)`/	run with gpu

​			`experiment2` (**3exp2**)/	run with gpu

​				Larger batch_size reduces experiment time, but need more epochs to converge. Use batch_size=512 as default.						

​					`result32`/	batch_size=32

​					`result512`/	batch_size=512

​					`result512_full`/	batch_size=512

​					`result2048`/	batch_size=2048

​					`	result2048_full`/	batch_size=2048

​			`experiment3`(**3exp3**)/	run with gpu, batch_size=512

​			`experiment4`(**3exp4**)/	run with gpu, batch_size=512



​    	`unfinished_exp`/

​			`experiment_trade_label`/	2304 experiments, add trading label evaluation

​				window_size = [30,50,100]
​				prediction_k = [10,20,30,40]
​				feature_num = [4,5]
​				label_threshold = [0.0003,0.0006,0.0009,0.0012]
​				lstm_units = [16,32,64]
​				learning_rate = [0.01,0.001]
​				epsilon = [1,0.0000001]
​				regularizer = [0,0.001]

​			`experiment_no_trade_label`/	2304 experiments, same configuration as above

​			`experiment_old`/	864 experiments

​				input_list = [30,50,100]
​				pred_list = [30,50]
​				feature_list = [4,5]
​				threshold_list = [0.0002,0.0003,0.0004,0.0006,0.0008,0.0010]
​				lstm_list = [16,32,64]
​				lr_list = [0.01,0.001]
​				regularizer_list = [0,0.001]

​					`cpu_gpu_compare`/	Compare the result of cpu and gpu when random seed fixed

​							`acc&loss`/	See readme.txt, cpu and gpu have different results

​							`speed`/	Both cpu and gpu training time decrease when batch size grows, but gpu decreases faster with CuDNNLSTM. To conclude, batch_size=512 is a suitable decision.





#### Experiment Spec

`data set:` SMP500 mid=1 (24hrs)

`training period:` 2010/01~2010/06

`validation period:` 2010/07

`testing period (if implemented):` 2010/08

`all hyperparameters:` window_size, prediction_k, feature_num ,label_threshold, lstm_units, learning_rate, epsilon, regularizer, batch_size

​		`window_size:` the time-step of input tensor, range 10~400

​		`prediction_k:` predict the mean of future k minutes rise, fall or flat, range 10~200

​		`feature_num:` OHLC->4 features, OHLC+volume->5 features

​		`label_threshold`: the bp threshold to determine rise, fall or flat, range 0~0.005

​		`lstm_units:` units number of lstm layer, range 16,32,64

​		`learning_rate`: learning rate of the optimizer, range 0.01, 0.001

​		`epsilon`: epsilon of the Adam optimizer, range 1, 1E-07

​		`regularizer:` the regularization parameter, the higher the more it penalizes large weight, range 0, 0.001, 0.01. Use only kernel regularizer with L1 norm

​		`batch_size:` batch size of training

`other explanation:` 

1. batch normalization is used before all activation functions
2. input OHLC(calculate mean and std jointly) normalization with z-score, volume normalization with z-score



||2-class|3exp1(gpu)|3exp2(512_full)|3exp3|3exp4|
|---|---|:--|---|---|---|
|**purpose**|1. run experiments with the best profit-1bp labeling<br />2. see how window_size affect the performance|1. overview of hyperparams effect<br />2. test large regularizer(0.01)|1. see the effect of prediction_k, feature_num, and label_threshold|1. overview of hyperparams effect|1. see the effect of window_size, lstm_units, learning_rate, epsilon, regularizer|
|**config count**|20|12|8|48|71|
|**non-fixed hyperparams**|window_size, <br />label_threshold| window_size, <br />prediction_k, <br />label_threshold, <br />lstm_units, <br />learning_rate, <br />regularizer |prediction_k, <br />feature_num, <br />label_threshold|window_size, <br />prediction_k, <br />feature_num,<br />label_threshold, <br />lstm_units, <br />learning_rate, <br />epsilon,<br />regularizer|window_size, <br />lstm_units, <br />learning_rate, <br />epsilon, <br />regularizer|
|**fixed hyperparams**|prediction_k=10,<br />feature_num=5,<br />lstm_units=32, learning_rate=0.01,<br />epsilon=1,<br />regularizer=0.001,<br />batch_size=512| feature_num=4,<br />epsilon=1,<br />batch_size=32 |window_size=30,<br /> lstm_units=16,<br />learning_rate=0.001,<br />epsilon=1,<br />regularizer=0.001,<br />batch_size=512|batch_size=512|k=10,<br />feature_num=5,<br />label_threshold=0.0012,<br />batch_size=512|
|**noticeable task id**|20 (best validation accuracy and f1)| 7441 (regularizer too big),<br />792(best validation accuracy) |62,134 (best validation accuracy)|8056 (best validation accuracy and f1)|1719 (best validation f1)|
|**conclusion**|1. 2-class experiments have better performance than 3-class in general, because it avoids imbalanced data and fragmentary label problem<br />2. large window_size have slightly better performance than small window_size<br />3. label_threshold is not significant in this experiment| 1. regularizer 0.01 is too big, constraining the training progress<br />2. small label_threshold like 0.0003 makes it hard to train, because the labels are fragmentary. In contrast, large label_threshold like 0.001 achieves higher validation accuracy due to imbalanced data(many flat label), but low f1-score |1. label_threshold 0.001 gives better accuracy than 0.0003, but validation f1 is both low(0.45)<br />2. prediction_k and feature_num is not significant in this experiment|1. large prediction_k(110) combine with large label_threshold(0.002) makes the label not fragmentary and not imbalanced, therefore easier to train and predict.<br />2. lstm_units, learning_rate, epsilon, regularizer is not significant in this experiment|1. imbalanced data due to low prediction_k (10) and high label_threshold (0.0012), giving high validation accuracy (0.9, predict many flat), and low validation f1 (0.4)<br />2. use f1 as criteria, epsilon 1E-07 is better than 1, regularizer 0 is better than 0.001, other hyperparams are not significant|

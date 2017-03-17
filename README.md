# TextBoxes-TensorFlow
TextBoxes re-implementation using tensorflow.
This project is greatly inspired by [slim project](https://github.com/tensorflow/models/tree/master/slim)
And many functions are modified based on [SSD-tensorflow project](https://github.com/balancap/SSD-Tensorflow)
Later, we will overwrite this project so make it more
flexiable and modularized.

Author: 
	Daitao Xing : dx383@nyu.edu
	Jin Huang   : jh5442@nyu.edu

# Progress
2017/ 03/14  

data_processing phase finished
Test：

	1. Download the dataset， put 1/ folder and gt.mat uner ddata/sythtext/ folder（will wirte script）   
	2. python datasets/data2record.py    
	3. python image_processing.py    
	
output： batch_size * 300 * 300 * 3 image

2017/ 03/17  

Finish the design of training(can start training)	

	python train.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --save_summaries_secs=60 \
    --save_interval_secs=600 \
    --weight_decay=0.0005 \
    --optimizer=adam \
    --learning_rate=0.001 \
    --batch_size=32

# Problems to be solved： 
	1. Need to redesign visualization		
	2. image_processing can be improved
		
# Next steps:
 
1. traing on other datasets
2. fine tunes
3. test
4. automatic downloading datasets and so on


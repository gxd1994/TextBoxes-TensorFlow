# TextBoxes-TensorFlow
TextBoxes re-implement using tensorflow
This project is great inspired by [slim project](https://github.com/tensorflow/models/tree/master/slim)
And many functions are modified based on [SSD-tensorflow project](https://github.com/balancap/SSD-Tensorflow)
Later, we will overwrite this project so make it more
flexiable and modularized.

Author: 
	Daitao Xing : dx383@nyu.edu
	Jin Huang   : jh5442@nyu.edu

# 当前进度
2017/ 03/14  

目前完成了data_processing 的阶段
测试：

	1. 下载数据， 将1/ 文件夹和gt.mat 文件放在data/sythtext/ 文件夹下（后面写脚本自动实现这一步）    
	2. python datasets/data2record.py    
	3. python image_processing.py    
	
output： batch_size * 300 * 300 * 3 的图片
  
2017/ 03/17  

完成train 的设计，现在可以开始训练
	
	python train.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --save_summaries_secs=60 \
    --save_interval_secs=600 \
    --weight_decay=0.0005 \
    --optimizer=adam \
    --learning_rate=0.001 \
    --batch_size=32
问题： 
		1. 可视化部分需要重新设计
		2. image_processing 可以重写

# 下一步和整体流程 
 
1. traing on other datasets
2. fine tunes
3. test
4. 自动化脚本（下载数据之类的）


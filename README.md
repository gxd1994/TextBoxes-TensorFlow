# TextBoxes-TensorFlow
TextBoxes re-implement using tensorflow

# 当前进度
2017/ 03/14
目前完成了data_processing 的阶段
测试：
	1. 下载数据， 将1/ 文件夹和gt.mat 文件放在data/sythtext/ 文件夹下（后面写脚本自动实现这一步）  
	2. python datasets/data2record.py  
	3. python image_processing.py  

output： batch_size * 300 * 300 * 3 的图片
  

problem:  
1. 目前还有一个小bug，bbox在归一化后会出现>1 或者<0 的值，需要去掉  
2. 生成的batch 将shuffle设置为True  
3. 目前bboxes没有输出，因为需要将其转化为正负样本之后才能放到train_batch中与images一起输出  

# 下一步和整体流程  
	1. 设计Textbox 网络  
		1. 处理bbox部分
			这一部分很庞大，也很繁琐，设计的细节很多
			1.1 生成anchor bbox
			1.2 将image和bbox输入转化为正负样本输出
			1.3 nonmax 
			1.4 待补充
		2. ssd网络
		3. loss function
		4. 其他(优化方法等) 

	2. 设计graph，用于在GPU（多GPU）上运行
		此部分参考slim 框架
	3. 设计test 脚本（这一部分暂时略过）


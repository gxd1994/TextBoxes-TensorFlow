"""
This script will test all functions and scripts in data pre-processing phase
Test functions includes:
	image_processing.
		data <- tfrecord
		image_buffer, label_index, bbox, name <-  parse_example
		image,labels,bbox <- image_processing
			image <- distorted_image 
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as skio

import tf_extended as tfe
from image_processing import *
import cv2

def visualize_bbox(image, bboxes):
	"""
	Input: image (height, width, channels)
		   bboxes (numof bboxes, 4) in order(ymin, xmin, ymax, xmax)
		          range(0,1) 
	"""
	numofbox = bboxes.shape[0]
	width = image.shape[1]
	height = image.shape[0]
	def norm(x):
		if x < 0:
			x = 0
		else:
			if x > 1:
				x = 1
		return x
	xmin = [int(norm(i) * width) for i in bboxes[:,1]]
	ymin = [int(norm(i) * height) for i in bboxes[:,0]]
	ymax = [int(norm(i) * height) for i in bboxes[:,2]]
	xmax = [int(norm(i) * width) for i in bboxes[:,3]]

	for i in range(numofbox):
		image = cv2.rectangle(image,(xmin[i],ymin[i]),
							 (xmax[i],ymax[i]),(0,0,0))
	skio.imshow(image)
	skio.show()




if __name__ == "__main__":
	data_dir = '/Users/xiaodiu/Documents/github/projecttextbox/TextBoxes-TensorFlow/data/sythtext/'
	file_name = data_dir + '1.tfrecord'
	## test if file_name exists  
	
	example = tf.python_io.tf_record_iterator(file_name).next()
	image_buffer, label, bboxes, name= parse_example(example)
	image,label,bboxes = image_processing(image_buffer, bboxes,label,
										 train= True, thread_id = 0)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		Image, label, bboxes = sess.run([image, label, bboxes])
		print label.shape
		print bboxes
		#print name
		#print width
		#print height
		print Image.shape
		visualize_bbox(Image, bboxes)
		skio.imshow(Image)
		skio.show()


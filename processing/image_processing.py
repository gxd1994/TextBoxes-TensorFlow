
import tensorflow as tf
import tf_extended as tfe
import os
import matplotlib.pyplot as plt
import skimage.io as skio
import cv2
import numpy as np


def image_processing(image, bbox,labels, text_shape,train = True):
	Height = text_shape[0]
	Width = text_shape[1]
	image = tf.image.convert_image_dtype(image, dtype=tf.float32)
	if train:
		image,labels,bbox = distorted_image(image, Height,labels,Width,bbox)
	else:
		image = eval_image(image, Height, Width)

	return image, labels, bbox

def distorted_image(image, height,labels,width,bbox,scope = None):
	# Each bounding box has shape [1, num_boxes, box coords] and
	# the coordinates are ordered [ymin, xmin, ymax, xmax].

	# Display the bounding box in the first thread only.
	with tf.name_scope(scope, 'distorted_bounding_box_crop',
	 [image, bbox,height,width]):
		
		bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
						tf.shape(image),
						bounding_boxes=bbox,
						min_object_covered=0.1,
						aspect_ratio_range=(0.9,1.1),
						area_range=(0.1,1.0),
						max_attempts=200,
						use_image_if_no_bounding_boxes=True)

		distort_bbox = distort_bbox[0, 0]

		# Crop the image to the specified bounding box.
		cropped_image = tf.slice(image, bbox_begin, bbox_size)
		# Restore the shape since the dynamic slice loses 3rd dimension.
		
		distorted_image = tf.image.resize_images(cropped_image, [height, width],
											 method=tf.image.ResizeMethod.BILINEAR)
		distorted_image.set_shape([height, width, 3])

		distorted_image = tf.image.random_flip_left_right(distorted_image)
		# Randomly distort the colors.
		distorted_image = distort_color(distorted_image)


		bboxes = tfe.bboxes_resize(distort_bbox, bbox)
		print "labels: %s " % (labels)
		label, bboxes = tfe.bboxes_filter_overlap(labels, bboxes,threshold = 0.4)
		print "bboxes: %s " % (bboxes)
		return distorted_image, label, bboxes
	


def eval_image(image, height, width, scope=None):
	"""Prepare one image for evaluation.

	Args:
	image: 3-D float Tensor
	height: integer
	width: integer
	scope: Optional scope for op_scope.
	Returns:
	3-D float Tensor of prepared image.
	"""
	with tf.name_scope(scope, 'eval_image',[image, height, width]):
	# Crop the central region of the image with an area containing 87.5% of
	# the original image.
		image = tf.image.central_crop(image, central_fraction=0.875)

		# Resize the image to the original height and width.
		image = tf.expand_dims(image, 0)
		image = tf.image.resize_bilinear(image, [height, width],
										 align_corners=False)
		image = tf.squeeze(image, [0])
	return image


def distort_color(image, scope=None):
	"""Distort the color of the image.

	Each color distortion is non-commutative and thus ordering of the color ops
	matters. Ideally we would randomly permute the ordering of the color ops.
	Rather then adding that level of complication, we select a distinct ordering
	of color ops for each preprocessing thread.

	Args:
	image: Tensor containing single image.
	thread_id: preprocessing thread ID.
	scope: Optional scope for op_scope.
	Returns:
	color-distorted image
	"""
	color_ordering = np.random.randint(2)
	if color_ordering == 0:
	  image = tf.image.random_brightness(image, max_delta=32. / 255.)
	  image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
	  image = tf.image.random_hue(image, max_delta=0.2)
	  image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
	elif color_ordering == 1:
	  image = tf.image.random_brightness(image, max_delta=32. / 255.)
	  image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
	  image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
	  image = tf.image.random_hue(image, max_delta=0.2)

	# The random_* ops do not necessarily clamp.
	image = tf.clip_by_value(image, 0.0, 1.0)
	return image

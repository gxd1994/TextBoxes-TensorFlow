
""" 
This framework is based on SSD_tensorlow(https://github.com/balancap/SSD-Tensorflow)
Add descriptions
"""

import math
from collections import namedtuple

import numpy as np
import tensorflow as tf

import tf_extended as tfe
from nets import custom_layers
from nets import textbox_common

slim = tf.contrib.slim

# =========================================================================== #
# Text class definition.
# =========================================================================== #
TextboxParams = namedtuple('TextboxParameters', 
										['img_shape',
										 'num_classes',
										 'feat_layers',
										 'feat_shapes',
										 'scale_range',
										 'anchor_ratios',
										 'normalizations',
										 'prior_scaling',
										 'step',
										 'scales'
										 ])

class TextboxNet(object):
	"""
	Implementation of the Textbox 300 network.

	The default features layers with 300x300 image input are:
	  conv4_3 ==> 38 x 38
	  fc7 ==> 19 x 19
	  conv6_2 ==> 10 x 10
	  conv7_2 ==> 5 x 5
	  conv8_2 ==> 3 x 3
	  pool6 ==> 1 x 1
	The default image size used to train this network is 300x300.
	"""
	default_params = TextboxParams(
		img_shape=(300, 300),
		num_classes=2,
		feat_layers=['conv4', 'conv7', 'conv8', 'conv9', 'conv10', 'global'],
		feat_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
		scale_range=[0.20, 0.90],
		anchor_ratios=[1,2,3,5,7,10],
		normalizations=[20, -1, -1, -1, -1, -1],
		prior_scaling=[0.1, 0.1, 0.2, 0.2],
		step = 0.14 ,
		scales = [0.2, 0.34, 0.48, 0.62, 0.76, 0.90]
		)

	def __init__(self, params=None):
		"""
		Init the Textbox net with some parameters. Use the default ones
		if none provided.
		"""
		if isinstance(params, TextboxParams):
			self.params = params
		else:
			self.params = self.default_params
			#self.params.step = (scale_range[1] - scale_range[0])/ 5 
			#self.params.scales = [scale_range[0] + i* self.params.step for i in range(6)]

	# ======================================================================= #
	def net(self, inputs,
			is_training=True,
			dropout_keep_prob=0.5,
			reuse=None,
			scope='text_box_300'):
		"""
		Text network definition.
		"""
		r = text_net(inputs,
					feat_layers=self.params.feat_layers,
					normalizations=self.params.normalizations,
					is_training=is_training,
					dropout_keep_prob=dropout_keep_prob,
					reuse=reuse,
					scope=scope)
		# Update feature shapes (try at least!)
		"""
		if update_feat_shapes:
			shapes = ssd_feat_shapes_from_net(r[0], self.params.feat_shapes)
			self.params = self.params._replace(feat_shapes=shapes)
		"""
		return r

	def arg_scope(self, weight_decay=0.0005, data_format='NHWC'):
		"""Network arg_scope.
		"""
		return ssd_arg_scope(weight_decay, data_format=data_format)

	def arg_scope_caffe(self, caffe_scope):
		"""Caffe arg_scope used for weights importing.
		"""
		return ssd_arg_scope_caffe(caffe_scope)

	# ======================================================================= #
	'''
	def update_feature_shapes(self, predictions):
		"""Update feature shapes from predictions collection (Tensor or Numpy
		array).
		"""
		shapes = ssd_feat_shapes_from_net(predictions, self.params.feat_shapes)
		self.params = self.params._replace(feat_shapes=shapes)
	'''

	def anchors(self, img_shape, dtype=np.float32):
		"""Compute the default anchor boxes, given an image shape.
		"""
		return textbox_achor_all_layers(img_shape,
									  self.params.feat_shapes,
									  self.params.anchor_ratios,
									  self.params.scales,
									  0.5,
									  dtype)

	def bboxes_encode(self, bboxes, anchors,
					  scope='text_bboxes_encode'):
		"""Encode labels and bounding boxes.
		"""
		return textbox_common.tf_text_bboxes_encode(
						bboxes, anchors,
						matching_threshold=0.1,
						prior_scaling=self.params.prior_scaling,
						scope=scope)

	def losses(self, logits, localisations,
			   glocalisations, gscores,
			   match_threshold=0.1,
			   negative_ratio=3.,
			   alpha=1.,
			   label_smoothing=0.,
			   scope='ssd_losses'):
		"""Define the SSD network losses.
		"""
		return ssd_losses(logits, localisations,
						  glocalisations, gscores,
						  match_threshold=match_threshold,
						  negative_ratio=negative_ratio,
						  alpha=alpha,
						  label_smoothing=label_smoothing,
						  scope=scope)



def text_net(inputs,
			feat_layers=TextboxNet.default_params.feat_layers,
			normalizations=TextboxNet.default_params.normalizations,
			is_training=True,
			dropout_keep_prob=0.5,
			reuse=None,
			scope='text_box_300'):
	end_points = {}
	with tf.variable_scope(scope, 'text_box_300', [inputs], reuse=reuse):
		# Original VGG-16 blocks.
		net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
		end_points['conv1'] = net
		net = slim.max_pool2d(net, [2, 2], scope='pool1')
		# Block 2.
		net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
		end_points['conv2'] = net
		net = slim.max_pool2d(net, [2, 2], scope='pool2')
		# Block 3.
		net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
		end_points['conv3'] = net
		net = slim.max_pool2d(net, [2, 2], scope='pool3')
		# Block 4.
		net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
		end_points['conv4'] = net
		net = slim.max_pool2d(net, [2, 2], scope='pool4')
		# Block 5.
		net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
		end_points['conv5'] = net
		net = slim.max_pool2d(net, [3, 3], stride=1, scope='pool5')

		# Additional SSD blocks.
		# Block 6: let's dilate the hell out of it!
		net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6')
		end_points['conv6'] = net
		# Block 7: 1x1 conv. Because the fuck.
		net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
		end_points['conv7'] = net

		# Block 8/9/10/11: 1x1 and 3x3 convolutions stride 2 (except lasts).
		end_point = 'conv8'
		with tf.variable_scope(end_point):
			net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
			net = custom_layers.pad2d(net, pad=(1, 1))
			net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3', padding='VALID')
		end_points[end_point] = net
		end_point = 'conv9'
		with tf.variable_scope(end_point):
			net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
			net = custom_layers.pad2d(net, pad=(1, 1))
			net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
		end_points[end_point] = net
		end_point = 'conv10'
		with tf.variable_scope(end_point):
			net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
			net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
		end_points[end_point] = net
		end_point = 'global'
		with tf.variable_scope(end_point):
			net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
			net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
		end_points[end_point] = net

		# Prediction and localisations layers.
		predictions = []
		logits = []
		localisations = []
		for i, layer in enumerate(feat_layers):
			with tf.variable_scope(layer + '_box'):
				p, l = text_multibox_layer(layer,
										  end_points[layer],
										  normalizations[i])
			#predictions.append(prediction_fn(p))
			logits.append(p)
			localisations.append(l)

		return localisations, logits, end_points


def text_multibox_layer(layer,
					   inputs,
					   normalization=-1):
	"""
	Construct a multibox layer, return a class and localization predictions.
	The  most different between textbox and ssd is the prediction shape
	where textbox has prediction score shape (38,38,2,6)
	and location has shape (38,38,2,6,4)
	besise,the kernel for fisrt 5 layers is 1*5 and padding is (0,2)
	kernel for the last layer is 1*1 and padding is 0
	"""
	net = inputs
	if normalization > 0:
		net = custom_layers.l2_normalization(net, scaling=True)
	# Number of anchors.
	num_anchors = 6
	num_classes = 2
	# Location.
	num_loc_pred = 2*num_anchors * 4
	if(layer == 'global'):
		loc_pred = slim.conv2d(net, num_loc_pred, [1, 1], activation_fn=None, padding = 'VALID',
						   scope='conv_loc')
	else:
		loc_pred = slim.conv2d(net, num_loc_pred, [1, 5], activation_fn=None, padding = 'SAME',
						   scope='conv_loc')
	#loc_pred = custom_layers.channel_to_last(loc_pred)
	loc_pred = tf.reshape(loc_pred, loc_pred.get_shape().as_list()[:-1] + [2,num_anchors,4])
	# Class prediction.
	scores_pred = 2 * num_anchors * num_classes
	if(layer == 'global'):
		sco_pred = slim.conv2d(net, scores_pred, [1, 1], activation_fn=None, padding = 'VALID',
						   scope='conv_cls')
	else:
		sco_pred = slim.conv2d(net, scores_pred, [1, 5], activation_fn=None, padding = 'SAME',
						   scope='conv_cls')
	#cls_pred = custom_layers.channel_to_last(cls_pred)
	sco_pred = tf.reshape(sco_pred, sco_pred.get_shape().as_list()[:-1] + [2,num_anchors,num_classes])
	return sco_pred, loc_pred



## produce anchor for one layer
# each feature point has 12 default textboxes(6 boxes + 6 offsets boxes)
# aspect ratios = (1,2,3,5,7,10)
# feat_size :
	# conv4_3 ==> 38 x 38
	# fc7 ==> 19 x 19
	# conv6_2 ==> 10 x 10
	# conv7_2 ==> 5 x 5
	# conv8_2 ==> 3 x 3
	# pool6 ==> 1 x 1

def textbox_anchor_one_layer(img_shape,
							 feat_size,
							 ratios,
							 scale,
							 offset = 0.5,
							 dtype=np.float32):
	# Follow the papers scheme
	# 12 ahchor boxes with out sk' = sqrt(sk * sk+1)
	y, x = np.mgrid[0:feat_size[0], 0:feat_size[1]] + 0.5
	y = y.astype(dtype) / feat_size[0]
	x = x.astype(dtype) / feat_size[1]
	x_offset = x
	y_offset = y + offset
	x_out = np.stack((x, x_offset), -1)
	y_out = np.stack((y, y_offset), -1)
	y_out = np.expand_dims(y_out, axis=-1)
	x_out = np.expand_dims(x_out, axis=-1)


	# 
	num_anchors = 6
	h = np.zeros((num_anchors, ), dtype=dtype)
	w = np.zeros((num_anchors, ), dtype=dtype)
	for i ,r in enumerate(ratios):
		h[i] = scale / math.sqrt(r) / feat_size[0]
		w[i] = scale * math.sqrt(r) / feat_size[1]
	return y_out, x_out, h, w



## produce anchor for all layers
def textbox_achor_all_layers(img_shape,
						   layers_shape,
						   anchor_ratios,
						   scales,
						   offset=0.5,
						   dtype=np.float32):
	"""
	Compute anchor boxes for all feature layers.
	"""
	layers_anchors = []
	for i, s in enumerate(layers_shape):
		anchor_bboxes = textbox_anchor_one_layer(img_shape, s,
												 anchor_ratios,
												 scales[i],
												 offset=offset, dtype=dtype)
		layers_anchors.append(anchor_bboxes)
	return layers_anchors

def ssd_arg_scope(weight_decay=0.0005, data_format='NHWC'):
	"""Defines the VGG arg scope.

	Args:
	  weight_decay: The l2 regularization coefficient.

	Returns:
	  An arg_scope.
	"""
	with slim.arg_scope([slim.conv2d, slim.fully_connected],
						activation_fn=tf.nn.relu,
						weights_regularizer=slim.l2_regularizer(weight_decay),
						weights_initializer=tf.contrib.layers.xavier_initializer(),
						biases_initializer=tf.zeros_initializer()):
		with slim.arg_scope([slim.conv2d, slim.max_pool2d],
							padding='SAME',
							data_format=data_format):
			with slim.arg_scope([custom_layers.pad2d,
								 custom_layers.l2_normalization,
								 custom_layers.channel_to_last],
								data_format=data_format) as sc:
				return sc


# =========================================================================== #
# Caffe scope: importing weights at initialization.
# =========================================================================== #
def ssd_arg_scope_caffe(caffe_scope):
	"""Caffe scope definition.

	Args:
	  caffe_scope: Caffe scope object with loaded weights.

	Returns:
	  An arg_scope.
	"""
	# Default network arg scope.
	with slim.arg_scope([slim.conv2d],
						activation_fn=tf.nn.relu,
						weights_initializer=caffe_scope.conv_weights_init(),
						biases_initializer=caffe_scope.conv_biases_init()):
		with slim.arg_scope([slim.fully_connected],
							activation_fn=tf.nn.relu):
			with slim.arg_scope([custom_layers.l2_normalization],
								scale_initializer=caffe_scope.l2_norm_scale_init()):
				with slim.arg_scope([slim.conv2d, slim.max_pool2d],
									padding='SAME') as sc:
					return sc


# =========================================================================== #
# Text loss function.
# =========================================================================== #
def ssd_losses(logits, localisations,
			   glocalisations, gscores,
			   match_threshold=0.1,
			   negative_ratio=3.,
			   alpha=1.,
			   label_smoothing=0.,
			   scope=None):
	"""Loss functions for training the text box network.


	Arguments:
	  logits: (list of) predictions logits Tensors;
	  localisations: (list of) localisations Tensors;
	  glocalisations: (list of) groundtruth localisations Tensors;
	  gscores: (list of) groundtruth score Tensors;
	"""
	with tf.name_scope(scope, 'text_loss'):
		l_cross_pos = []
		l_cross_neg = []
		l_loc = []
		for i in range(len(logits)):
			dtype = logits[i].dtype
			with tf.name_scope('block_%i' % i):
				# Determine weights Tensor.
				pmask = gscores[i] > match_threshold
				ipmask = tf.cast(pmask, tf.int32)
				fpmask = tf.cast(pmask, dtype)
				n_positives = tf.reduce_sum(fpmask)

				# Negative mask
				# Number of negative entries to select.
				n_neg = tf.cast(negative_ratio * n_positives, tf.int32)

				nvalues = tf.where(tf.cast(1-ipmask,tf.bool), gscores[i], np.zeros(gscores[i].shape))
				nvalues_flat = tf.reshape(nvalues, [-1])
				val, idxes = tf.nn.top_k(nvalues_flat, k=n_neg)
				minval = val[-1]
				# Final negative mask.
				nmask = nvalues > minval
				fnmask = tf.cast(nmask, dtype)
				inmask = tf.cast(nmask, tf.int32)
				# Add cross-entropy loss.
				with tf.name_scope('cross_entropy_pos'):
					loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[i],
																		  labels=ipmask)
					loss = tf.losses.compute_weighted_loss(loss, fpmask)
					l_cross_pos.append(loss)

				with tf.name_scope('cross_entropy_neg'):
					loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[i],
																		  labels=inmask)
					loss = tf.losses.compute_weighted_loss(loss, fnmask)
					l_cross_neg.append(loss)

				# Add localization loss: smooth L1, L2, ...
				with tf.name_scope('localization'):
					# Weights Tensor: positive mask + random negative.
					weights = tf.expand_dims(alpha * fpmask, axis=-1)
					loss = custom_layers.abs_smooth(localisations[i] - glocalisations[i])
					loss = tf.losses.compute_weighted_loss(loss, weights)
					l_loc.append(loss)

		# Additional total losses...
		with tf.name_scope('total'):
			total_cross_pos = tf.add_n(l_cross_pos, 'cross_entropy_pos')
			total_cross_neg = tf.add_n(l_cross_neg, 'cross_entropy_neg')
			total_cross = tf.add(total_cross_pos, total_cross_neg, 'cross_entropy')
			total_loc = tf.add_n(l_loc, 'localization')

			# Add to EXTRA LOSSES TF.collection
			tf.add_to_collection('EXTRA_LOSSES', total_cross_pos)
			tf.add_to_collection('EXTRA_LOSSES', total_cross_neg)
			tf.add_to_collection('EXTRA_LOSSES', total_cross)
			tf.add_to_collection('EXTRA_LOSSES', total_loc)












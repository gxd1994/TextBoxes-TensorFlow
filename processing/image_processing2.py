"""
This file pre-process images from a datasets and
output batch iamges and labels(bboxes)

parse examples from tfrecord
	1.parse_example

Pre-processing images :
	1. crop and pad images randomly
	2. crop and pad bbox
	3. Transform images and bboxes to input/output vectors

"""

import tensorflow as tf
import tf_extended as tfe
import os
import matplotlib.pyplot as plt
import skimage.io as skio
import cv2


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 1,
							"""Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('Height', 300,
							"""Provide square images of this size.""")
tf.app.flags.DEFINE_integer('Width', 300,
							"""Provide square images of this size.""")
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
							"""Number of preprocessing threads per tower. """
							"""Please make this a multiple of 4.""")
tf.app.flags.DEFINE_integer('num_readers', 1,
							"""Number of parallel readers during train.""")

# Images are preprocessed asynchronously using multiple threads specified by
# --num_preprocss_threads and the resulting processed images are stored in a
# random shuffling queue. The shuffling queue dequeues --batch_size images
# for processing on a given Inception tower. A larger shuffling queue guarantees
# better mixing across examples within a batch and results in slightly higher
# predictive performance in a trained model. Empirically,
# --input_queue_memory_factor=16 works well. A value of 16 implies a queue size
# of 1024*16 images. Assuming RGB 299x299 images, this implies a queue size of
# 16GB. If the machine is memory limited, then decrease this factor to
# decrease the CPU memory footprint, accordingly.

tf.app.flags.DEFINE_integer('input_queue_memory_factor', 1,
							"""Size of the queue of preprocessed images. """
							"""Default is ideal but try smaller values, e.g. """
							"""4, 2 or 1, if host memory is constrained. See """
							"""comments in code for more details.""")


def distorted_inputs(data_files, batch_size=None, num_preprocess_threads=None):
	"""Generate batches of distorted versions of ImageNet images.

	Use this function as the inputs for training a network.

	Distorting images provides a useful technique for augmenting the data
	set during training in order to make the network invariant to aspects
	of the image that do not effect the label.

	Args:
	dataset: instance of Dataset class specifying the dataset.
	batch_size: integer, number of examples in batch
	num_preprocess_threads: integer, total number of preprocessing threads but
	  None defaults to FLAGS.num_preprocess_threads.

	Returns:
	images: Images. 4D tensor of size [batch_size, FLAGS.image_size,
									   FLAGS.image_size, 3].
	labels: 1-D integer Tensor of [batch_size].
	"""
	if not batch_size:
		batch_size = FLAGS.batch_size

	# Force all input processing onto CPU in order to reserve the GPU for
	# the forward inference and back-propagation.
	with tf.device('/cpu:0'):
		images ,box,name= batch_inputs(
			data_files, batch_size, train=True,
			num_preprocess_threads=num_preprocess_threads,
			num_readers=FLAGS.num_readers)
	return images,box,name

def parse_example(example_serialized):
	"""
	One example proto containing following fields
	'image/height': int64_feature(shape[0]),
	'image/width': int64_feature(shape[1]),
	'image/channels': int64_feature(shape[2]),
	'image/shape': int64_feature(shape),
	'image/object/bbox/xmin': float_feature(xmin),
	'image/object/bbox/xmax': float_feature(xmax),
	'image/object/bbox/ymin': float_feature(ymin),
	'image/object/bbox/ymax': float_feature(ymax),
	'image/object/bbox/label': int64_feature(label),
	'image/format': bytes_feature('jpeg'),
	'image/encoded': bytes_feature(image_data.tostring()),

	Input : example_serialized

	Ouput: 
		Image_buffer
	"""
	feature_map = {
		'image/height': tf.FixedLenFeature([1], tf.int64),
		'image/width': tf.FixedLenFeature([1], tf.int64),
		'image/channels': tf.FixedLenFeature([1], tf.int64),
		'image/shape': tf.FixedLenFeature([3], tf.int64),
		'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
		'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
		'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
		'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
		'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
		'image/format': tf.FixedLenFeature([], tf.string, default_value='jpeg'),
		'image/encoded': tf.FixedLenFeature([], tf.string, default_value=''),
		'image/name': tf.VarLenFeature(dtype = tf.string),
	}
	features = tf.parse_single_example(example_serialized, feature_map)
	#image = tf.decode_raw(features['image/encoded'], tf.uint8)
	xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
	ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
	xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
	ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)
	bboxes = tf.concat([ymin, xmin, ymax, xmax],0)
	bboxes = tf.expand_dims(bboxes,0)
	bboxes = tf.transpose(bboxes, [0,2,1])
	Image_buffer = features['image/encoded']
	label = tf.expand_dims(features['image/object/bbox/label'].values, 0)
	width = tf.cast(features['image/height'], dtype=tf.int64)
	height = tf.cast(features['image/width'], dtype=tf.int64)
	name = tf.cast(features['image/name'], dtype = tf.string)
	print "name %s" % (name) 
	return Image_buffer, label, bboxes, name



def image_processing(image_buffer, bbox,labels, train,thread_id = 0):
	image = decode_jpeg(image_buffer)
	Height = FLAGS.Height
	Width = FLAGS.Width

	if train:
		image,labels,bbox = distorted_image(image, Height,labels,Width,bbox,thread_id)
	else:
		image = eval_image(image, Height, Width)

	return image, labels, bbox

def distorted_image(image, height,labels,width,bbox,thread_id,scope = None):
	# Each bounding box has shape [1, num_boxes, box coords] and
	# the coordinates are ordered [ymin, xmin, ymax, xmax].

	# Display the bounding box in the first thread only.
	with tf.name_scope(scope, 'distorted_bounding_box_crop',
	 [image, bbox,height,width]):
		if not thread_id:
			image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
													bbox)
			tf.summary.image('image_with_bounding_boxes', image_with_box)


		bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
						tf.shape(image),
						bounding_boxes=bbox,
						min_object_covered=0.1,
						aspect_ratio_range=(0.9,1.1),
						area_range=(0.1,1.0),
						max_attempts=200,
						use_image_if_no_bounding_boxes=True)

		if not thread_id:
			image_with_distorted_box = tf.image.draw_bounding_boxes(
					tf.expand_dims(image, 0), distort_bbox)
			tf.summary.image('images_with_distorted_bounding_box',
					   image_with_distorted_box)

		distort_bbox = distort_bbox[0, 0]

		# Crop the image to the specified bounding box.
		cropped_image = tf.slice(image, bbox_begin, bbox_size)
		# Restore the shape since the dynamic slice loses 3rd dimension.
		
		distorted_image = tf.image.resize_images(cropped_image, [height, width],
											 method=tf.image.ResizeMethod.BILINEAR)
		distorted_image.set_shape([height, width, 3])
		if not thread_id:
			tf.summary.image('cropped_resized_image',
					   tf.expand_dims(distorted_image, 0))
		distorted_image = tf.image.random_flip_left_right(distorted_image)
		# Randomly distort the colors.
		distorted_image = distort_color(distorted_image, thread_id)

		if not thread_id:
			tf.summary.image('final_distorted_image',
					   tf.expand_dims(distorted_image, 0))
		# Update bounding boxes: resize and filter out.

		bboxes = tfe.bboxes_resize(distort_bbox, bbox)
		print "labels: %s " % (labels)
		label, bboxes = tfe.bboxes_filter_overlap(labels, bboxes,threshold = 0.4)

		return distorted_image, label, bboxes



def decode_jpeg(image_buffer, scope=None):
	"""Decode a JPEG string into one 3-D float image Tensor.

	Args:
	image_buffer: scalar string Tensor.
	scope: Optional scope for op_scope.
	Returns:
	3-D float Tensor with values ranging from [0, 1).
	"""
	with tf.name_scope(scope, 'decode_jpeg',[image_buffer]):
	# Decode the string as an RGB JPEG.
	# Note that the resulting image contains an unknown height and width
	# that is set dynamically by decode_jpeg. In other words, the height
	# and width of image is unknown at compile-time.
		image = tf.image.decode_jpeg(image_buffer, channels=3)
	# After this point, all image pixels reside in [0,1)
	# until the very end, when they're rescaled to (-1, 1).  The various
	# adjust_* ops all require this range for dtype float.
		image = tf.image.convert_image_dtype(image, dtype=tf.float32)
		print 'image after decode %s' % (image)
	return image	


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


def distort_color(image, thread_id=0, scope=None):
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
  with tf.name_scope( scope, 'distort_color',[image]):
	color_ordering = thread_id % 2

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


def batch_inputs(data_files, batch_size, train, num_preprocess_threads=None,num_readers=4):

	"""Contruct batches of training or evaluation examples from the image dataset.
	Args:
	dataset: instance of Dataset class specifying the dataset.
	  See dataset.py for details.
	batch_size: integer
	train: boolean
	num_preprocess_threads: integer, total number of preprocessing threads
	num_readers: integer, number of parallel readers

	Returns:
	images: 4-D float Tensor of a batch of images
	labels: 1-D integer Tensor of [batch_size].

	Raises:
	ValueError: if data is not found
	"""

	#print 1
	with tf.name_scope('batch_processing'):
		if data_files is None:
		  raise ValueError('No data files found for this dataset')

		# Create filename_queue
		if train:
		  filename_queue = tf.train.string_input_producer(data_files,num_epochs = 2,
														  shuffle=True,
														  capacity=16)
		else:
		  filename_queue = tf.train.string_input_producer(data_files, num_epochs = 2,
														  shuffle=False,
														  capacity=1)
		if num_preprocess_threads is None:
		  num_preprocess_threads = FLAGS.num_preprocess_threads

		if num_preprocess_threads % 4:
		  raise ValueError('Please make num_preprocess_threads a multiple '
						   'of 4 (%d % 4 != 0).', num_preprocess_threads)

		if num_readers is None:
		  num_readers = FLAGS.num_readers

		if num_readers < 1:
		  raise ValueError('Please make num_readers at least 1')

		# Approximate number of examples per shard.
		
		examples_per_shard = 512

		# Size the random shuffle queue to balance between good global
		# mixing (more examples) and memory use (fewer examples).
		# 1 image uses 299*299*3*4 bytes = 1MB
		# The default input_queue_memory_factor is 16 implying a shuffling queue
		# size: examples_per_shard * 16 * 1MB = 17.6GB

		min_queue_examples = examples_per_shard * FLAGS.input_queue_memory_factor
		if train:
		  examples_queue = tf.RandomShuffleQueue(
			  capacity=min_queue_examples + 3 * batch_size,
			  min_after_dequeue=min_queue_examples,
			  dtypes=[tf.string])
		else:
		  examples_queue = tf.FIFOQueue(
			  capacity=examples_per_shard + 3 * batch_size,
			  dtypes=[tf.string])

		# Create multiple readers to populate the queue of examples.
		if num_readers > 1:
		  enqueue_ops = []
		  for _ in range(num_readers):
			reader = tf.TFRecordReader()
			_, value = reader.read(filename_queue)
			enqueue_ops.append(examples_queue.enqueue([value]))

		  tf.train.queue_runner.add_queue_runner(
			  tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
		  example_serialized = examples_queue.dequeue()
		else:
		  reader = tf.TFRecordReader()
		  _, example_serialized = reader.read(filename_queue)

		images_and_labels = []
		for thread_id in range(num_preprocess_threads):
		  # Parse a serialized Example proto to extract the image and metadata.
		  image_buffer, label_index, bbox, name= parse_example(example_serialized)
		  image,labels,bbox = image_processing(image_buffer, bbox,label_index,
		  										train, thread_id)
		  
		  images_and_labels.append([image, bbox[1,:],name])

		images ,box,names= tf.train.batch_join(
			images_and_labels,
			batch_size=batch_size,
			capacity=2 * num_preprocess_threads * batch_size)
		print 'box shape %s' % (box.shape)

		# Reshape images into these desired dimensions.
		
		print 'image batch phase %s' % (images)
		height = FLAGS.Height
		width = FLAGS.Width
		depth = 3

		#images = tf.cast(images, tf.float32)
		#images = tf.reshape(images, shape=[batch_size, height, width, depth])

		print 'image reshape %s' % (images)

		# Display the training images in the visualizer.

		tf.summary.image('images', images)

	return images, box, names



def main(_):
	data_dir = '/Users/xiaodiu/Documents/github/projecttextbox/TextBoxes-TensorFlow/data/sythtext/'
	tf_record_pattern = os.path.join(data_dir, '*.tfrecord')
	data_files = tf.gfile.Glob(tf_record_pattern)
	print data_files
	images ,box,name= distorted_inputs(data_files)
	print images.shape
	
	with tf.Session() as sess:
	    sess.run(tf.global_variables_initializer())
	    sess.run(tf.local_variables_initializer())
	    coord = tf.train.Coordinator()
	    threads = tf.train.start_queue_runners(coord=coord)
	    #print sess.run(shape)
	    img = sess.run(images)
	    boxb = sess.run(box)
	    name = sess.run(name)
	    print name
	    print img.shape
	    print img[0,:,:,:]
	    #skio.imshow(img[1,:,:,:])
	    image = img[0,:,:,:]
	    xmin = int(boxb[0,1] * 300)
	    ymin = int(boxb[0,0] * 300)
	    xmax = int(boxb[0,3] * 300)
	    ymax = int(boxb[0,2] * 300)
	    skio.imshow(cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(0,0,0)))
	    skio.show()
	    skio.imshow(skio.imread(data_dir+ name))
	    skio.show()
	    coord.request_stop()
	    coord.join(threads)
	 
if __name__ == '__main__':
	tf.app.run()





import datasets.sythtextprovider as sythtext
import tensorflow as tf
slim = tf.contrib.slim
import cv2
#import matplotlib.pyplot as plt
from PIL import Image
from datasets.sythtextprovider import get_datasets
"""
data_dir = '/Users/xiaodiu/Documents/github/projecttextbox/TextBoxes-TensorFlow/data/sythtext/'
file = data_dir + '1.tfrecord'


tfrecord_file_queue = tf.train.string_input_producer([file] ,num_epochs = 1,name='queue',shuffle = True)
reader = tf.TFRecordReader()
_, tfrecord_serialized = reader.read(tfrecord_file_queue)
# label and image are stored as bytes but could be stored as
# int64 or float64 values in a serialized tf.Example protobuf.
tfrecord_features = tf.parse_single_example(tfrecord_serialized,
features={
    'image/height': tf.FixedLenFeature([1], tf.int64),
    'image/width': tf.FixedLenFeature([1], tf.int64),
    'image/channels': tf.FixedLenFeature([1], tf.int64),
    'image/shape': tf.FixedLenFeature([], tf.int64),
    'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
    'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
    'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
    'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
    'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
    #'image/object/bbox/label_text' : tf.VarLenFeature(dtype=tf.string),
    'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
    'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
}, name='features')
# image was saved as uint8, so we have to decode as uint8.

image = tf.decode_raw(tfrecord_features['image/encoded'], tf.uint8)
shape = tf.cast(tfrecord_features['image/shape'], tf.int64)
#image = tf.reshape(image, shape)
height = tf.cast(tfrecord_features['image/height'],tf.int64)
width = tf.cast(tfrecord_features['image/width'],tf.int64)

"""
dataset_dir = '/Users/xiaodiu/Documents/github/projecttextbox/TextBoxes-TensorFlow/data/sythtext/'
dataset = get_datasets(dataset_dir)

provider = slim.dataset_data_provider.DatasetDataProvider(
                    dataset,
                    num_readers=1,
                    common_queue_capacity=20 * 32,
                    common_queue_min=10 * 32,
                    shuffle=True)
            # Get for SSD network: image, labels, bboxes.
[image,shape, height, width,glabels, gbboxes,] = provider.get(['image','shape', 'height',
												  'width',
                                                  'object/label',
                                                  'object/bbox'])
#image = tf.decode_raw(image, tf.uint8)
#height = tf.cast(features['height'], tf.int32)
#width = tf.cast(features['width'], tf.int32)
#image = tf.reshape(image, tf.pack([height,width,3]))


#print image
print shape
print glabels
print gbboxes
with tf.Session() as sess:

	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)
	#print sess.run(shape)
	#img = sess.run(image)
	#print img.shape
	print sess.run([height,width])
	print sess.run(shape)
	print sess.run(gbboxes).shape
	print sess.run(glabels)
	
	
	coord.request_stop()
	coord.join(threads)




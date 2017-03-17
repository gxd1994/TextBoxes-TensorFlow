## an initial version
## Transform the tfrecord to slim data provider format

import numpy 
import tensorflow as tf
import os
slim = tf.contrib.slim




ITEMS_TO_DESCRIPTIONS = {
    'image': 'slim.tfexample_decoder.Image',
    'shape': 'shape',
    'height': 'height',
    'width': 'width',
    'object/bbox': 'box',
    'object/label': 'label'
}
SPLITS_TO_SIZES = {
    'train': 4262,
}
NUM_CLASSES = 2



def get_datasets(data_dir,file_pattern = '*.tfrecord'):
    file_patterns = os.path.join(data_dir, file_pattern)
    print 'file_path: {}'.format(file_patterns)
    reader = tf.TFRecordReader
    keys_to_features = {
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

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        #'image': slim.tfexample_decoder.Tensor('image/encoded'),
        'shape': slim.tfexample_decoder.Tensor('image/shape'),
        'height': slim.tfexample_decoder.Tensor('image/height'),
        'width': slim.tfexample_decoder.Tensor('image/width'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
                ['xmin', 'ymin', 'xmax', 'ymax'], 'image/object/bbox/'),
        'object/label': slim.tfexample_decoder.Tensor('image/object/bbox/label'),
        #'objext/txt': slim.tfexample_decoder.Tensor('image/object/bbox/label_text'),
      }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_names = None


    return slim.dataset.Dataset(
        data_sources=file_patterns,
        reader=reader,
        decoder=decoder,
        num_samples=SPLITS_TO_SIZES['train'],
        items_to_descriptions=ITEMS_TO_DESCRIPTIONS,
        num_classes=NUM_CLASSES,
        labels_to_names=labels_to_names)
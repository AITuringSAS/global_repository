import hashlib
import io
import os
import glob
import random
from shutil import copy2

from absl import app
from absl import flags
from absl import logging
from lxml import etree

import PIL.Image
import PIL.ImageOps
import tensorflow.compat.v1 as tf
import cv2
import json

# import dataset.tfrecord_util as tfrecord_util
# import dataset.create_yaml as create_yaml
import tfrecord_util
import create_yaml

import tempfile


flags.DEFINE_string('data_dir', 'temp', 'Root directory to raw dataset')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_integer('num_shards', 10, 'Number of shards for output file.')
flags.DEFINE_boolean('use_data_augmentation', False, 'Define the use of data augmentation')
flags.DEFINE_float('perc_split_training', 1, 'Define the percentage to the training set')


FLAGS = flags.FLAGS

GLOBAL_IMG_ID = 0  # global image id.
GLOBAL_ANN_ID = 0  # global annotation id.
tmpDir = tempfile.TemporaryDirectory()


def unique_list(list_):
    # intilize a null list
    unique_list = []

    # traverse for all elements
    for x in list_:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list


def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path:  # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def create_labelmapDict_patch(list_all_images, path_dataset):
    list_all_classes = []
    for idx, name_image_ in enumerate(list_all_images):
        # print(name_image_)
        _, tail = os.path.split(name_image_)
        temp_obj = []
        name_file_xml_all = os.path.join(path_dataset, 'LABELS', tail[0:-3] + 'xml')
        # print(name_file_xml_all)
        if os.path.exists(name_file_xml_all):
            with tf.gfile.GFile(name_file_xml_all, 'rb') as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            data = tfrecord_util.recursive_parse_xml_to_dict(xml)['annotation']
            if 'object' in data:
                for obj in data['object']:
                    name_in_obj_ = obj['name'].replace(' ', '').strip()
                    if name_in_obj_ != 'INCOMPLETAS':
                        list_all_classes.append(name_in_obj_)
                        temp_obj.append(obj)
    list_all_classes = unique_list(list_all_classes)
    list_all_classes.sort()
    # list_classes = [dict_patch[name_] for name_ in list_old]
    list_all_classes.insert(0, 'background')
    labelmap_ = {el: k for k, el in enumerate(list_all_classes)}
    return labelmap_


def get_image_id(filename):
    del filename
    global GLOBAL_IMG_ID
    GLOBAL_IMG_ID += 1
    return GLOBAL_IMG_ID


def dict_to_tf_example(data,
                       label_map_dict,
                       ignore_difficult_instances=False):
    full_path = os.path.join(FLAGS.data_dir, 'IMAGENES', data['filename'])[0:-3] + 'jpg'
    image_ = cv2.imread(full_path)
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()
    image_id = get_image_id(data['filename'])
    width = int(image_.shape[1])  # int(data['size']['width'])
    height = int(image_.shape[0])  # int(data['size']['height'])
    image_id = get_image_id(data['filename'])
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    area = []
    classes = []
    classes_text = []
    if 'object' in data:
        for obj in data['object']:
            name_in_obj_ = obj['name'].replace(' ', '').strip()
            if name_in_obj_ in label_map_dict:
                x_pos = [int(obj['bndbox']['xmax']), int(obj['bndbox']['xmin'])]
                y_pos = [int(obj['bndbox']['ymax']), int(obj['bndbox']['ymin'])]
                xmin.append((float(min(x_pos))) / width)
                ymin.append((float(min(y_pos))) / height)
                xmax.append((float(max(x_pos))) / width)
                ymax.append((float(max(y_pos))) / height)
                area.append((xmax[-1] - xmin[-1]) * (ymax[-1] - ymin[-1]))
                classes_text.append(name_in_obj_.replace(' ', '').encode('utf8'))
                classes.append(int(label_map_dict[name_in_obj_]))  # label_map_dict[obj['name']])

    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/height':
                tfrecord_util.int64_feature(height),
                'image/width':
                tfrecord_util.int64_feature(width),
                'image/filename':
                tfrecord_util.bytes_feature(data['filename'].encode('utf8')),
                'image/source_id':
                tfrecord_util.bytes_feature(str(image_id).encode('utf8')),
                'image/key/sha256':
                tfrecord_util.bytes_feature(key.encode('utf8')),
                'image/encoded':
                tfrecord_util.bytes_feature(encoded_jpg),
                'image/format':
                tfrecord_util.bytes_feature('jpeg'.encode('utf8')),
                'image/object/bbox/xmin':
                tfrecord_util.float_list_feature(xmin),
                'image/object/bbox/xmax':
                tfrecord_util.float_list_feature(xmax),
                'image/object/bbox/ymin':
                tfrecord_util.float_list_feature(ymin),
                'image/object/bbox/ymax':
                tfrecord_util.float_list_feature(ymax),
                'image/object/area':
                tfrecord_util.float_list_feature(area),
                'image/object/class/text':
                tfrecord_util.bytes_list_feature(classes_text),
                'image/object/class/label':
                tfrecord_util.int64_list_feature(classes),
            }))
    return example


def main(_):
    total_images_training = 0
    writers = [
        tf.python_io.TFRecordWriter(FLAGS.output_path + '-%05d-of-%05d.tfrecord' %
                                    (i, FLAGS.num_shards))
        for i in range(FLAGS.num_shards)
    ]
    list_all_images = glob.glob(os.path.join(FLAGS.data_dir, 'IMAGENES', '*.jpg'))
    random.shuffle(list_all_images)  # shuffle list of images
    num_samp_train = int(FLAGS.perc_split_training * len(list_all_images))
    aituring_label_map_dict = create_labelmapDict_patch(list_all_images, FLAGS.data_dir)
    label_map_dict = aituring_label_map_dict
    class_main_names = list(label_map_dict.keys())[1::]
    labels_by_class_dict = dict.fromkeys(class_main_names, 0)
    create_yaml.create_yaml(class_main_names, FLAGS.output_path + '_config.yaml', FLAGS.use_data_augmentation)
    list_img_test = list_all_images[num_samp_train::]
    for idx, name_image_ in enumerate(list_all_images[0:num_samp_train]):
        _, tail = os.path.split(name_image_)
        data_total = {}
        name_file_xml_all = os.path.join(FLAGS.data_dir, 'LABELS', tail[0:-3] + 'xml')
        if os.path.exists(name_file_xml_all):
            with tf.gfile.GFile(name_file_xml_all, 'rb') as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            data_total = tfrecord_util.recursive_parse_xml_to_dict(xml)['annotation']
            if 'object' in data_total:
                for obj in data_total['object']:
                    name_in_obj_ = obj['name'].replace(' ', '')
                    if name_in_obj_ in labels_by_class_dict:
                        labels_by_class_dict[name_in_obj_] += 1
        if idx % 100 == 0:
            logging.info('On image %d of %d', idx, int(len(list_all_images)))
        if bool(data_total):
            total_images_training += 1
            tf_example = dict_to_tf_example(
                data_total,
                label_map_dict)
            writers[idx % FLAGS.num_shards].write(tf_example.SerializeToString())
    path_, _ = os.path.split(FLAGS.output_path)
    with open(os.path.join(path_, 'Num_Classes.json'), 'w') as outfile:
        json.dump(labels_by_class_dict, outfile)
    with open(os.path.join(FLAGS.data_dir, 'list_img_test.txt'), 'w') as out_txt:
        out_txt.writelines("%s\n" % img for img in list_img_test)

    for writer in writers:
        writer.close()
    print('---Aqui Total #####------')
    print(total_images_training)
    file_num_train = open(FLAGS.output_path + '_samplesTrain.txt', "w")
    file_num_train.write(str(total_images_training))
    file_num_train.close()


if __name__ == '__main__':
    app.run(main)

import argparse
import os
import numpy as np


def define_parameters():
  ################################################################################
  # TFRECORD FLAGS
  ################################################################################
  my_params = argparse.ArgumentParser(description='List the content of a folder')
  my_params.add_argument('--URL_DATASET', metavar='dataset', type=str,
                         default='', help='path/to/dataset', required=True)
  my_params.add_argument('--output_path', metavar='tfrecord-name', type=str,
                         default=os.path.join(os.getcwd(), 'tmp', 'tfrecords', 'train_tfrecord'), help='Filename to save output TFRecord', required=False)
  my_params.add_argument('--num_shards', metavar='shards', type=int, default=10,
                         help='Number of shards for output file', required=False)
  my_params.add_argument('--use_data_augmentation', metavar='True/False', type=bool, default=False,
                         help='Define the use of data augmentation', required=False)
  my_params.add_argument('--perc_split_training', metavar='0.0/1.0', type=float, default=0.9,
                         help='Define the percentage training set', required=False)

  ################################################################################
  # TRAINING FLAGS
  ################################################################################
  my_params.add_argument('--mode', metavar='train/eval', type=str, default='train',
                         help='Mode to run: train or eval (default: train)', required=False)
  my_params.add_argument('--train_file_pattern', metavar='path/to/tfrecord/folder', type=str, default=os.path.join(os.getcwd(), 'tmp', 'tfrecords', 'train_tfrecord*.tfrecord'),
                         help='Glob for training data files', required=False)
  my_params.add_argument('--BACKBONE_REF', metavar='efficientdet-d#', type=str, default='efficientdet-d1',
                         help='Backbone reference', required=True)
  my_params.add_argument('--MODEL_CKPTS', metavar='path/to/save/model', type=str, default=os.path.join(*[os.getcwd(), 'tmp', 'result']),
                         help='Location to save trained model', required=True)
  my_params.add_argument('--backbone_ckpt', metavar='path/to/checkpoint (default: efficientdet-d1)', type=str, default=os.path.join(os.getcwd(), 'tmp', 'efficientdet-d1'),
                         help='Location of the ResNet50 checkpoint to use for model', required=False)
  my_params.add_argument('--BATCH_SIZE', metavar='size', type=np.int64, default=64,
                         help='Define the global training batch size', required=True)
  my_params.add_argument('--NUM_EPOCHS', metavar='epochs', type=float, default=None,
                         help='Define the Number of epochs for training', required=True)
  my_params.add_argument('--hparams', metavar='path to config.yaml (tfrecord folder location usually)', type=str, default=os.path.join(os.getcwd(), 'tmp', 'tfrecords', 'train_tfrecord_config.yaml'),
                         help='Comma separated k=v pairs of hyperparameters or a module. containing attributes to use as hyperparameters', required=False)
  my_params.add_argument('--num_examples_per_epoch', metavar='size', type=float, default=4455,
                         help='Define the Number of examples in one epoch', required=False)

  ################################################################################
  # FROZEN MODEL FLAGS
  ################################################################################

  return my_params

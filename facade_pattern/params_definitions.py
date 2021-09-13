"""params_definitions module"""

import os
import argparse
import subprocess
import numpy as np


class PipelineTools(object):
  """PipelineTools
    This class provide the flags definitions for the main command line 

    Attributes
    ----------
    paths : dict
        define the paths to manage the workspace

    Methods
    -------
    define_parameters():
        Define the parameters for the command line

    download_and_uncompress_backbone(backbone_name: str):
        Download and uncompress the backbone given a backbone name in the URL list
  """

  def __init__(self):

    self.paths = {

        'WORKSPACE': os.path.join(os.getcwd(), 'tmp'),
        'DATASET_DIR': os.path.join(os.getcwd(), 'tmp', 'dataset'),
        'DATASET_FILE': os.path.join(os.getcwd(), 'tmp', 'dataset', 'dataset.zip'),
        'MODEL_OUTPUT': os.path.join(os.getcwd(), 'tmp', 'result'),
        'FROZEN_MODEL_DIR': os.path.join(os.getcwd(), 'tmp', 'frozen_model'),
        'TFRECORDS_DIR': os.path.join(os.getcwd(), 'tmp', 'tfrecords'),
        'BACKBONE_CKPT': {
            'DIR': os.path.join(os.getcwd(), 'tmp', 'ckpt'),
            'efficientdet-d0': os.path.join(os.getcwd(), 'tmp', 'ckpt', 'efficientdet-d0'),
            'efficientdet-d1': os.path.join(os.getcwd(), 'tmp', 'ckpt', 'efficientdet-d1'),
            'efficientdet-d2': os.path.join(os.getcwd(), 'tmp', 'ckpt', 'efficientdet-d2'),
            'efficientdet-d3': os.path.join(os.getcwd(), 'tmp', 'ckpt', 'efficientdet-d3')
        },
        'BACKBONE_CKPT_URL': {
            'efficientdet-d0':
                {
                    'URL': 'https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d0.tar.gz',
                    'TAR': 'efficientdet-d0.tar.gz'
                },
            'efficientdet-d1':
                {
                    'URL': 'https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d1.tar.gz',
                    'TAR': 'efficientdet-d1.tar.gz'
                },
            'efficientdet-d2':
                {
                    'URL': 'https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d2.tar.gz',
                    'TAR': 'efficientdet-d2.tar.gz'
                },
            'efficientdet-d3':
                {
                    'URL': 'https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d3.tar.gz',
                    'TAR': 'efficientdet-d3.tar.gz'
                }
        }
    }

  def define_parameters(self):
    """define_parameters method
        This method define the command line flags to be used in the main script 

        Parameters:
            None

        Returns:
            None
    """
    my_params = argparse.ArgumentParser(description='List the content of a folder')

    ################################################################################
    # CONFIG FILE FLAG
    ################################################################################
    my_params.add_argument("--configfile", help="Specify config file", metavar="FILE")

    ################################################################################
    # TFRECORD FLAGS
    ################################################################################
    my_params.add_argument('--URL_DATASET', metavar='dataset', type=str, action=env_default('URL_DATASET'),
                           help='path/to/dataset', required=False)
    my_params.add_argument('--output_path', metavar='tfrecord-name', type=str,
                           default=os.path.join(self.paths['TFRECORDS_DIR'], 'train_tfrecord'), help='Filename to save output TFRecord', required=False)
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
    my_params.add_argument('--train_file_pattern', metavar='path/to/tfrecord/folder', type=str, default=os.path.join(self.paths['TFRECORDS_DIR'], 'train_tfrecord*.tfrecord'),
                           help='Glob for training data files', required=False)
    my_params.add_argument('--BACKBONE_REF', metavar='efficientdet-d#', type=str, action=env_default('BACKBONE_REF'),
                           help='Backbone reference', required=False)
    my_params.add_argument('--MODEL_CKPTS', metavar='path/to/save/model', type=str, action=env_default('MODEL_CKPTS'),
                           help='Location to save trained model', required=False)
    my_params.add_argument('--backbone_ckpt', metavar='path/to/checkpoint (default: efficientdet-d1)', type=str, default=self.paths['BACKBONE_CKPT']['efficientdet-d1'],
                           help='Location of the ResNet50 checkpoint to use for model', required=False)
    my_params.add_argument('--BATCH_SIZE', metavar='size', type=np.int64, action=env_default('BATCH_SIZE'),
                           help='Define the global training batch size', required=False)
    my_params.add_argument('--NUM_EPOCHS', metavar='epochs', type=float, action=env_default('NUM_EPOCHS'),
                           help='Define the Number of epochs for training', required=False)
    my_params.add_argument('--hparams', metavar='path to config.yaml (tfrecord folder location usually)', type=str, default=os.path.join(self.paths['TFRECORDS_DIR'], 'train_tfrecord_config.yaml'),
                           help='Comma separated k=v pairs of hyperparameters or a module. containing attributes to use as hyperparameters', required=False)
    my_params.add_argument('--num_examples_per_epoch', metavar='size', type=float, default=4455,
                           help='Define the Number of examples in one epoch', required=False)

    ################################################################################
    # FROZEN MODEL FLAGS
    ################################################################################

    return my_params

  def download_and_uncompress_backbone(self, backbone_name):
    """define_parameters method
        This method download a pre-trained model 

        Parameters:
            backbone_name (str): model name

        Returns:
            None
    """
    # Download backbone checkpoints
    subprocess.call(['wget', self.paths['BACKBONE_CKPT_URL'][backbone_name]['URL'],
                     '-O', os.path.join(self.paths['BACKBONE_CKPT']['DIR'], self.paths['BACKBONE_CKPT_URL'][backbone_name]['TAR'])])

    # Uncompress backbone .tar.gz files
    subprocess.call(['tar', '-xzvf', os.path.join(self.paths['BACKBONE_CKPT']['DIR'],
                                                  self.paths['BACKBONE_CKPT_URL'][backbone_name]['TAR']), '-C', self.paths['BACKBONE_CKPT']['DIR']])

    # Delete .tar.gz backbone file
    subprocess.call(['rm', os.path.join(self.paths['BACKBONE_CKPT']['DIR'],
                                        self.paths['BACKBONE_CKPT_URL'][backbone_name]['TAR'])])


# Courtesy of http://stackoverflow.com/a/10551190 with env-var retrieval fixed
class EnvDefault(argparse.Action):
  """An argparse action class that auto-sets missing default values from env
  vars. Defaults to requiring the argument."""

  def __init__(self, envvar, required=True, default=None, **kwargs):
    if not default and envvar:
      if envvar in os.environ:
        default = os.environ[envvar]
    if required and default:
      required = False
    super(EnvDefault, self).__init__(default=default, required=required,
                                     **kwargs)

  def __call__(self, parser, namespace, values, option_string=None):
    setattr(namespace, self.dest, values)


def env_default(envvar):
  """decorator"""
  def wrapper(**kwargs):
    return EnvDefault(envvar, **kwargs)
  return wrapper

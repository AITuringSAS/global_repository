"""params_definitions module"""

import os
import tarfile
import argparse
import numpy as np
from zipfile import ZipFile


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
        my_params = argparse.ArgumentParser(description="""Command line options.""",
                                            add_help=True)

        my_params.add_argument('-v', '--version', action='version', version='%(prog)s 1.0')

        group1 = my_params.add_argument_group("tfrecord parameters")
        group2 = my_params.add_argument_group("training parameters")
        group3 = my_params.add_argument_group("TPU support parameters")
        group4 = my_params.add_argument_group("Model evaluation support parameters")

        ################################################################################
        # CONFIG FILE FLAG
        ################################################################################
        my_params.add_argument("--configfile", help="Specify config file", metavar="FILE")

        ################################################################################
        # TFRECORD FLAGS
        ################################################################################
        group1.add_argument('--URL_DATASET', metavar='dataset', type=str, action=env_default('URL_DATASET'),
                            help='URL from S3', required=False)
        group1.add_argument('--output_path', metavar='tfrecord-name', type=str,
                            default=os.path.join(self.paths['TFRECORDS_DIR'], 'train_tfrecord'), help='Filename to save output TFRecord', required=False)
        group1.add_argument('--num_shards', metavar='shards', type=int, default=10,
                            help='Number of shards for output file', required=False)
        group1.add_argument('--use_data_augmentation', metavar='True/False', type=bool, default=False,
                            help='Define the use of data augmentation', required=False)
        group1.add_argument('--perc_split_training', metavar='0.0/1.0', type=float, default=0.9,
                            help='Define the percentage training set', required=False)

        ################################################################################
        # TRAINING FLAGS
        ################################################################################
        group2.add_argument('--mode', metavar='train/eval or train_and_eval', type=str, default='train',
                            help='Mode to run: train or eval (default: train)', required=False)
        group2.add_argument('--train_file_pattern', metavar='path/to/tfrecord/folder', type=str, default=os.path.join(self.paths['TFRECORDS_DIR'], 'train_tfrecord*.tfrecord'),
                            help='Glob for training data files', required=False)
        group2.add_argument('--BACKBONE_REF', metavar='efficientdet-d#', type=str, action=env_default('BACKBONE_REF'),
                            help='Backbone reference', required=False)
        group2.add_argument('--MODEL_CKPTS', metavar='path/to/save/model', type=str, action=env_default('MODEL_CKPTS'),
                            help='Location to save trained model', required=False)
        group2.add_argument('--backbone_ckpt', metavar='path/to/checkpoint (default: efficientdet-d1)', type=str, default=self.paths['BACKBONE_CKPT']['efficientdet-d1'],
                            help='Location of the ResNet50 checkpoint to use for model', required=False)
        group2.add_argument('--BATCH_SIZE', metavar='size', type=np.int64, action=env_default('BATCH_SIZE'),
                            help='Define the global training batch size', required=False)
        group2.add_argument('--NUM_EPOCHS', metavar='epochs', type=float, action=env_default('NUM_EPOCHS'),
                            help='Define the Number of epochs for training', required=False)
        group2.add_argument('--hparams', metavar='path to config.yaml (tfrecord folder location usually)', type=str, default=os.path.join(self.paths['TFRECORDS_DIR'], 'train_tfrecord_config.yaml'),
                            help='Comma separated k=v pairs of hyperparameters or a module. containing attributes to use as hyperparameters', required=False)
        group2.add_argument('--num_examples_per_epoch', metavar='size', type=float, default=4455,
                            help='Define the Number of examples in one epoch', required=False)
        group2.add_argument('--ckpt', metavar='checkpoint', type=float, default=None,
                            help='Start training from this EfficientDet checkpoint.', required=False)
        group2.add_argument('--profile', metavar='True/False', type=bool, default=False,
                            help='Profile training performance.', required=False)

        #-------- TPU support
        group3.add_argument('--tpu', metavar='TPU-name or URL', type=str, default=None,
                            help='The Cloud TPU to use for training. This should be either the name used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470', required=False)
        group3.add_argument('--tpu_zone', metavar='zone', type=str, default=None,
                            help='GCE zone where the Cloud TPU is located in. If not specified, we will attempt to automatically detect the GCE project from metadata.', required=False)
        group3.add_argument('--strategy', metavar='tpu/gpus', type=str, default=None,
                            help='Training: gpus for multi-gpu, if None, use TF default.', required=False)
        group3.add_argument('--use_xla', metavar='True/False', type=bool, default=False,
                            help='Use XLA even if strategy is not tpu. If strategy is tpu, always use XLA, and this flag has no effect.', required=False)
        group3.add_argument('--num_cores', metavar='number of cores (default: 8)', type=int, default=8,
                            help='Number of TPU cores for training.', required=False)
        group3.add_argument('--use_spatial_partition', metavar='True/False', type=bool, default=False,
                            help='Use spatial partition.', required=False)
        group3.add_argument('--num_cores_per_replica', metavar='number', type=int, default=2,
                            help='Number of TPU cores per replica when using spatial partition.', required=False)
        group3.add_argument('--iterations_per_loop', metavar='number', type=int, default=1000,
                            help='Number of iterations per TPU training loop.', required=False)

        #-------- GCP project reference
        my_params.add_argument('--gcp_project', metavar='project_name', type=str, default=None,
                               help='Project name for the Cloud TPU-enabled project. If not specified, we will attempt to automatically detect the GCE project from metadata.', required=False)

        #-------- Model evaluation
        group4.add_argument('--val_file_pattern', metavar='path/to/eval*tfrecords', type=str, default=None,
                            help='Glob for evaluation tfrecords.', required=False)
        group4.add_argument('--eval_batch_size', metavar='size', type=int, default=1,
                            help='global evaluation batch size.', required=False)
        group4.add_argument('--eval_samples', metavar='samples', type=int, default=5000,
                            help='Number of samples for eval.', required=False)
        group4.add_argument('--eval_after_train', metavar='True/False', type=bool, default=True,
                            help='Run one eval after the training finishes.', required=False)
        group4.add_argument('--min_eval_interval', metavar='seconds', type=int, default=180,
                            help='Minimum seconds between evaluations.', required=False)
        group4.add_argument('--eval_timeout', metavar='seconds', type=int, default=None,
                            help='Maximum seconds between checkpoints before evaluation terminates.', required=False)
        group4.add_argument('--run_epoch_in_child_process', metavar='number', type=bool, default=False,
                            help='This option helps to rectify CPU memory leak. If True, every epoch is run in a separate process for train and eval and memory will be cleared. Drawback: need to kill 2 processes if trainining needs to be interrupted.', required=False)

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
        os.system(' '.join(['wget', '--no-check-certificate', self.paths['BACKBONE_CKPT_URL'][backbone_name]['URL'],
                            '-O', os.path.join(self.paths['BACKBONE_CKPT']['DIR'], self.paths['BACKBONE_CKPT_URL'][backbone_name]['TAR'])]))

        # Uncompress backbone .tar.gz files
        tar = tarfile.open(os.path.join(self.paths['BACKBONE_CKPT']['DIR'],
                                        self.paths['BACKBONE_CKPT_URL'][backbone_name]['TAR']))
        tar.extractall(self.paths['BACKBONE_CKPT']['DIR'])
        tar.close()

        # Delete .tar.gz backbone file
        os.remove(os.path.join(self.paths['BACKBONE_CKPT']['DIR'],
                               self.paths['BACKBONE_CKPT_URL'][backbone_name]['TAR']))

    def download_and_uncompress_dataset(self, dataset_url):
        # Download dataset from S3 (dataset must be public)
        os.system(' '.join(['wget', '--no-check-certificate',
                            dataset_url, '-O', self.paths['DATASET_FILE']]))
        # Unzip dataset
        with ZipFile(self.paths['DATASET_FILE'], 'r') as zipobj:
            zipobj.extractall(self.paths['DATASET_DIR'])
        # Delete .zip dataset
        os.remove(self.paths['DATASET_FILE'])


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

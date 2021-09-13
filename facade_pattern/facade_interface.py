import os
import sys
import absl
import subprocess
sys.path.append(os.path.join(os.getcwd(), 'automl', 'efficientdet'))
sys.path.append(os.path.join(os.getcwd(), 'tfRecordMod'))
from automl.efficientdet import main as efficientdet_train_tf1
from facade_pattern import params_definitions
from tfRecordMod import create_tfrecords


class Inteface(object):
    """docstring for FacadeInteface"""

    def __init__(self):
        self.paths = {

            'WORKSPACE': os.path.join(os.getcwd(), 'tmp'),
            'DATASET_DIR': os.path.join(os.getcwd(), 'tmp', 'dataset'),
            'DATASET_FILE': os.path.join(os.getcwd(), 'tmp', 'dataset', 'dataset.zip'),
            'MODEL_OUTPUT': os.path.join(os.getcwd(), 'tmp', 'result'),
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

        self.args = params_definitions.define_parameters()
        self.args = self.args.parse_args()
        absl.flags.FLAGS.mark_as_parsed()

        try:
            os.mkdir(self.paths['WORKSPACE'])
            os.mkdir(self.paths['DATASET_DIR'])
            os.mkdir(self.paths['MODEL_OUTPUT'])
            os.mkdir(self.paths['TFRECORDS_DIR'])
            os.mkdir(self.paths['BACKBONE_CKPT']['DIR'])
        except OSError:
            pass

        # Download dataset from S3 (dataset must be public)
        subprocess.call(['wget', self.args.URL_DATASET, '-O', self.paths['DATASET_FILE']])
        # Unzip dataset
        subprocess.call(['unzip', '-q', self.paths['DATASET_FILE'], '-d', self.paths['DATASET_DIR']])
        # Delete .zip dataset
        subprocess.call(['rm', self.paths['DATASET_FILE']])
        # Download backbone checkpoints
        self.download_and_uncompress_backbone(self.args.BACKBONE_REF)

    def create_tfrecord(self):
        print("=================================")
        print("--> Creating TFRecord...please...wait!")

        create_tfrecords.flags.FLAGS.data_dir = self.paths['DATASET_DIR']
        create_tfrecords.flags.FLAGS.output_path = self.args.output_path
        create_tfrecords.flags.FLAGS.num_shards = self.args.num_shards
        create_tfrecords.flags.FLAGS.use_data_augmentation = self.args.use_data_augmentation
        create_tfrecords.flags.FLAGS.perc_split_training = self.args.perc_split_training

        create_tfrecords.main('')

        print("--> tfrecord files generated!")
        print()

    def run_training(self):
        print("=================================")
        print("--> Starting training...please...wait!")

        efficientdet_train_tf1.flags.FLAGS.mode = self.args.mode
        efficientdet_train_tf1.flags.FLAGS.train_file_pattern = self.args.train_file_pattern
        efficientdet_train_tf1.flags.FLAGS.model_name = self.args.BACKBONE_REF
        efficientdet_train_tf1.flags.FLAGS.model_dir = os.path.join(self.paths['MODEL_OUTPUT'], self.args.MODEL_CKPTS)
        efficientdet_train_tf1.flags.FLAGS.backbone_ckpt = self.paths['BACKBONE_CKPT'][self.args.BACKBONE_REF]
        efficientdet_train_tf1.flags.FLAGS.train_batch_size = self.args.BATCH_SIZE
        efficientdet_train_tf1.flags.FLAGS.num_epochs = self.args.NUM_EPOCHS
        efficientdet_train_tf1.flags.FLAGS.hparams = self.args.hparams
        efficientdet_train_tf1.flags.FLAGS.num_examples_per_epoch = self.args.num_examples_per_epoch

        efficientdet_train_tf1.main('')

        print("--> training has finished!")
        print()

    def save_frozen_model(self):
        pass

    def download_and_uncompress_backbone(self, backbone_name):
        # Download backbone checkpoints
        subprocess.call(['wget', self.paths['BACKBONE_CKPT_URL'][backbone_name]['URL'],
                         '-O', os.path.join(self.paths['BACKBONE_CKPT']['DIR'], self.paths['BACKBONE_CKPT_URL'][backbone_name]['TAR'])])

        # Uncompress backbone .tar.gz files
        subprocess.call(['tar', '-xzvf', os.path.join(self.paths['BACKBONE_CKPT']['DIR'],
                                                      self.paths['BACKBONE_CKPT_URL'][backbone_name]['TAR']), '-C', self.paths['BACKBONE_CKPT']['DIR']])

        # Delete .tar.gz backbone file
        subprocess.call(['rm', os.path.join(self.paths['BACKBONE_CKPT']['DIR'],
                                            self.paths['BACKBONE_CKPT_URL'][backbone_name]['TAR'])])

import os
import sys
import absl
import subprocess

# Append submodule path to python search path
sys.path.append(os.path.join(os.getcwd(), 'automl', 'efficientdet'))
sys.path.append(os.path.join(os.getcwd(), 'tfRecordMod'))
sys.path.append(os.path.join(os.getcwd(), 'freezeModelMod'))

# Import submodules
from automl.efficientdet import main as efficientdet_train_tf1
from freezeModelMod import freeze_aituring
from facade_pattern import params_definitions
from tfRecordMod import create_tfrecords


class Inteface(object):
    """docstring for FacadeInteface"""

    def __init__(self):

        self.pip_tools = params_definitions.PipelineTools()
        self.paths = self.pip_tools.paths
        self.args = self.pip_tools.define_parameters()
        self.args = self.args.parse_args()
        absl.flags.FLAGS.mark_as_parsed()

        try:
            os.mkdir(self.paths['WORKSPACE'])
            os.mkdir(self.paths['DATASET_DIR'])
            os.mkdir(self.paths['MODEL_OUTPUT'])
            os.mkdir(self.paths['FROZEN_MODEL_DIR'])
            os.mkdir(self.paths['TFRECORDS_DIR'])
            os.mkdir(self.paths['BACKBONE_CKPT']['DIR'])
        except OSError:
            pass

        # Download dataset from S3 (dataset must be public)
        subprocess.call(['wget', self.args.URL_DATASET, '-O', self.paths['DATASET_FILE'], 'y'])
        # Unzip dataset
        subprocess.call(['unzip', '-q', self.paths['DATASET_FILE'], '-d', self.paths['DATASET_DIR'], 'y'])
        # Delete .zip dataset
        subprocess.call(['rm', self.paths['DATASET_FILE']])
        # Download backbone checkpoints
        self.download_and_uncompress_backbone(self.args.BACKBONE_REF)
        print()

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
        print("=================================")
        print("--> Saving frozen model...please...wait!")

        freeze_aituring.flags.FLAGS.path_ckpt = os.path.join(self.paths['MODEL_OUTPUT'], self.args.MODEL_CKPTS)
        freeze_aituring.flags.FLAGS.path_yaml = os.path.join(self.paths['TFRECORDS_DIR'], 'train_tfrecord_config.yaml')
        freeze_aituring.flags.FLAGS.path_output = self.paths['FROZEN_MODEL_DIR']
        freeze_aituring.flags.FLAGS.model_name_ = self.args.BACKBONE_REF

        freeze_aituring.main('')

        print("--> frozen model saved!")
        print()

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

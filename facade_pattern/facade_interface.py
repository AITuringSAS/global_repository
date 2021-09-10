import os
import sys
import absl
sys.path.append(os.path.join(*[os.getcwd(), 'automl', 'efficientdet']))
sys.path.append(os.path.join(*[os.getcwd(), 'tfRecordMod']))
from automl.efficientdet import main as efficientdet_train_tf1
from facade_pattern import params_definitions
from tfRecordMod import create_tfrecords


class Inteface(object):
    """docstring for FacadeInteface"""

    def __init__(self):
        self.args = params_definitions.define_parameters()
        self.args = self.args.parse_args()
        absl.flags.FLAGS.mark_as_parsed()

    def create_tfrecord(self):
        print("=================================")
        print("--> Creating TFRecord...please...wait!")

        create_tfrecords.flags.FLAGS.data_dir = self.args.URL_DATASET
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
        efficientdet_train_tf1.flags.FLAGS.model_dir = os.path.join(
            *[os.getcwd(), 'tmp', 'result', self.args.MODEL_CKPTS])
        efficientdet_train_tf1.flags.FLAGS.backbone_ckpt = self.args.backbone_ckpt
        efficientdet_train_tf1.flags.FLAGS.train_batch_size = self.args.BATCH_SIZE
        efficientdet_train_tf1.flags.FLAGS.num_epochs = self.args.NUM_EPOCHS
        efficientdet_train_tf1.flags.FLAGS.hparams = self.args.hparams
        efficientdet_train_tf1.flags.FLAGS.num_examples_per_epoch = self.args.num_examples_per_epoch

        efficientdet_train_tf1.main('')

        print("--> training has finished!")
        print()

    def save_frozen_model(self):
        pass

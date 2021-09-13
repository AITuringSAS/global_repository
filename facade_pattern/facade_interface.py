"""facade_interface module"""

import os
import sys
import absl
import subprocess
import configparser

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
    """FacadeInteface
    This class provide an interface for the automl/efficientdet repository. It set the params definitions
    from params_definitions.py for the command line managment. The automl efficiendet repository is a git submodule 
    that won't be affected by this interface. We propose a Facade Design Pattern to act as mediator between the commands
    and the internal scripts.


    Attributes
    ----------
    pip_tools : PipelineTools
        param definition object from params_definitions.py
    paths : dict
        define the paths to manage the workspace
    args : sys.argv
        parse the command line arguments


    Methods
    -------
    create_tfrecord():
        Creates the tfrecord files from input dataset.

    run_training():
        Starts the training pipeline.

    save_frozen_model():
        Saves the trained model in frozen format.


    Training with command line parameters
    -------
    python3 main.py --URL_DATASET https://datasets-aiteam.s3.amazonaws.com/DATASET-5-FOTOS.zip \
    --BATCH_SIZE 1 \
    --BACKBONE_REF efficientdet-d0 \
    --NUM_EPOCHS 5 \
    --MODEL_CKPTS efficientdet-d0-Output-folder


    Training with environment variables    
    -------
    export URL_DATASET=https://datasets-aiteam.s3.amazonaws.com/DATASET-5-FOTOS.zip
    export BATCH_SIZE=1
    export BACKBONE_REF=efficientdet-d0
    export NUM_EPOCHS=5
    export MODEL_CKPTS=efficientdet-d0-Output-folder
    python3 main.py 


    Training with configuration file    
    -------
    python3 main.py --configfile params.config

    params.config:
    [Defaults]
    URL_DATASET=https://datasets-aiteam.s3.amazonaws.com/DATASET-5-FOTOS.zip
    BACKBONE_REF=efficientdet-d0
    BATCH_SIZE=1
    NUM_EPOCHS=5
    MODEL_CKPTS=efficientdet-d0-output
    """

    def __init__(self):
        self.pip_tools = params_definitions.PipelineTools()
        self.paths = self.pip_tools.paths
        self.args = self.pip_tools.define_parameters()
        self.args = self.args.parse_args()

        if self.args.configfile:
            config = configparser.ConfigParser()
            config.optionxform = str
            config.read(os.path.join(os.getcwd(), self.args.configfile))
            defaults = config['Defaults']

            args = vars(self.args)
            result = dict(defaults)
            result.update({k: v for k, v in args.items() if v is not None})  # Update if v is not None
            self.args = result.copy()
        else:
            self.args = vars(self.args)

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
        subprocess.call(['wget', self.args['URL_DATASET'], '-O', self.paths['DATASET_FILE']])
        # Unzip dataset
        subprocess.call(['unzip', '-q', self.paths['DATASET_FILE'], '-o', self.paths['DATASET_DIR']])
        # Delete .zip dataset
        subprocess.call(['rm', self.paths['DATASET_FILE']])
        # Download backbone checkpoints
        self.pip_tools.download_and_uncompress_backbone(self.args['BACKBONE_REF'])
        print()

    def create_tfrecord(self):
        """create_tfrecord method
        This method define the command line flags in the main script for the creation of the TFRecords at
        repository/tfRecordMod/create_tfrecords.py

        Parameters:
            None

        Returns:
            None
        """
        print("=================================")
        print("--> Creating TFRecord...please...wait!")

        create_tfrecords.flags.FLAGS.data_dir = self.paths['DATASET_DIR']
        create_tfrecords.flags.FLAGS.output_path = self.args['output_path']
        create_tfrecords.flags.FLAGS.num_shards = self.args['num_shards']
        create_tfrecords.flags.FLAGS.use_data_augmentation = self.args['use_data_augmentation']
        create_tfrecords.flags.FLAGS.perc_split_training = self.args['perc_split_training']

        create_tfrecords.main('')

        print("--> tfrecord files generated!")
        print()

    def run_training(self):
        """run_training method
        This method define the command line flags in the main script for the training step at
        repository/automl/efficientdet/main.py

        Parameters:
            None

        Returns:
            None
        """
        print("=================================")
        print("--> Starting training...please...wait!")

        efficientdet_train_tf1.flags.FLAGS.mode = self.args['mode']
        efficientdet_train_tf1.flags.FLAGS.train_file_pattern = self.args['train_file_pattern']
        efficientdet_train_tf1.flags.FLAGS.model_name = self.args['BACKBONE_REF']
        efficientdet_train_tf1.flags.FLAGS.model_dir = os.path.join(
            self.paths['MODEL_OUTPUT'], self.args['MODEL_CKPTS'])
        efficientdet_train_tf1.flags.FLAGS.backbone_ckpt = self.paths['BACKBONE_CKPT'][self.args['BACKBONE_REF']]
        efficientdet_train_tf1.flags.FLAGS.train_batch_size = self.args['BATCH_SIZE']
        efficientdet_train_tf1.flags.FLAGS.num_epochs = self.args['NUM_EPOCHS']
        efficientdet_train_tf1.flags.FLAGS.hparams = self.args['hparams']
        efficientdet_train_tf1.flags.FLAGS.num_examples_per_epoch = self.args['num_examples_per_epoch']

        efficientdet_train_tf1.main('')

        print("--> training has finished!")
        print()

    def save_frozen_model(self):
        """save_frozen method
        This method define the command line flags in the main script to export the frozen model at
        repository/freezeModelMod/freeze_aituring.py

        Parameters:
            None

        Returns:
            None
        """
        print("=================================")
        print("--> Saving frozen model...please...wait!")

        freeze_aituring.flags.FLAGS.path_ckpt = os.path.join(self.paths['MODEL_OUTPUT'], self.args['MODEL_CKPTS'])
        freeze_aituring.flags.FLAGS.path_yaml = os.path.join(self.paths['TFRECORDS_DIR'], 'train_tfrecord_config.yaml')
        freeze_aituring.flags.FLAGS.path_output = self.paths['FROZEN_MODEL_DIR']
        freeze_aituring.flags.FLAGS.model_name_ = self.args['BACKBONE_REF']

        freeze_aituring.main('')

        print("--> frozen model saved!")
        print()

"""facade_interface module"""
import os
import sys
import absl
import configparser
import numpy as np
import logging

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
        if len(sys.argv) < 2:
            self.args.print_usage()
            sys.exit(1)

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
        logging.info("filename: ", self.paths['DATASET_FILE'])

        # Download dataset from S3
        self.pip_tools.download_and_uncompress_dataset(self.args['URL_DATASET'])

        # Download backbone checkpoints
        self.pip_tools.download_and_uncompress_backbone(self.args['BACKBONE_REF'])
        logging.info('')

    def create_tfrecord(self):
        """create_tfrecord method
        This method define the command line flags in the main script for the creation of the TFRecords at
        repository/tfRecordMod/create_tfrecords.py

        Parameters:
            None

        Returns:
            None
        """
        logging.info("=================================")
        logging.info("--> Creating TFRecord...please...wait!")

        create_tfrecords.flags.FLAGS.data_dir = self.paths['DATASET_DIR']
        create_tfrecords.flags.FLAGS.output_path = self.args['output_path']
        create_tfrecords.flags.FLAGS.num_shards = int(self.args['num_shards'])
        create_tfrecords.flags.FLAGS.use_data_augmentation = bool(self.args['use_data_augmentation'])
        create_tfrecords.flags.FLAGS.perc_split_training = float(self.args['perc_split_training'])

        create_tfrecords.main('')

        logging.info("--> tfrecord files generated!")
        logging.info('')

    def run_training(self):
        """run_training method
        This method define the command line flags in the main script for the training step at
        repository/automl/efficientdet/main.py

        Parameters:
            None

        Returns:
            None
        """
        logging.info("=================================")
        logging.info("--> Starting training...please...wait!")

        #-------------------- GCP project reference standard parameters
        efficientdet_train_tf1.flags.FLAGS.gcp_project = self.args['gcp_project']

        #-------------------- Training standard parameters
        efficientdet_train_tf1.flags.FLAGS.mode = self.args['mode']
        efficientdet_train_tf1.flags.FLAGS.train_file_pattern = self.args['train_file_pattern']
        efficientdet_train_tf1.flags.FLAGS.model_name = self.args['BACKBONE_REF']
        efficientdet_train_tf1.flags.FLAGS.model_dir = os.path.join(
            self.paths['MODEL_OUTPUT'], self.args['MODEL_CKPTS'])
        efficientdet_train_tf1.flags.FLAGS.backbone_ckpt = self.paths['BACKBONE_CKPT'][self.args['BACKBONE_REF']]
        efficientdet_train_tf1.flags.FLAGS.train_batch_size = np.int64(self.args['BATCH_SIZE'])
        efficientdet_train_tf1.flags.FLAGS.num_epochs = int(self.args['NUM_EPOCHS'])
        efficientdet_train_tf1.flags.FLAGS.hparams = self.args['hparams']
        efficientdet_train_tf1.flags.FLAGS.num_examples_per_epoch = int(self.args['num_examples_per_epoch'])

        #-------------------- TPU support
        efficientdet_train_tf1.flags.FLAGS.tpu = self.args['tpu']
        efficientdet_train_tf1.flags.FLAGS.tpu_zone = self.args['tpu_zone']
        efficientdet_train_tf1.flags.FLAGS.strategy = self.args['strategy']
        efficientdet_train_tf1.flags.FLAGS.use_xla = bool(self.args['use_xla'])
        efficientdet_train_tf1.flags.FLAGS.num_cores = int(self.args['num_cores'])
        efficientdet_train_tf1.flags.FLAGS.use_spatial_partition = bool(self.args['use_spatial_partition'])
        efficientdet_train_tf1.flags.FLAGS.num_cores_per_replica = int(self.args['num_cores_per_replica'])
        efficientdet_train_tf1.flags.FLAGS.iterations_per_loop = int(self.args['iterations_per_loop'])

        #-------------------- Model evaluation support
        efficientdet_train_tf1.flags.FLAGS.eval_batch_size = int(self.args['eval_batch_size'])
        efficientdet_train_tf1.flags.FLAGS.eval_samples = int(self.args['eval_samples'])
        efficientdet_train_tf1.flags.FLAGS.eval_after_train = bool(self.args['eval_after_train'])
        efficientdet_train_tf1.flags.FLAGS.min_eval_interval = int(self.args['min_eval_interval'])
        efficientdet_train_tf1.flags.FLAGS.eval_timeout = int(self.args['eval_timeout'])
        efficientdet_train_tf1.flags.FLAGS.run_epoch_in_child_process = bool(self.args['run_epoch_in_child_process'])

        #-------------------- Start training
        efficientdet_train_tf1.main('')

        logging.info("--> training has finished!")
        logging.info('')

    def save_frozen_model(self):
        """save_frozen method
        This method define the command line flags in the main script to export the frozen model at
        repository/freezeModelMod/freeze_aituring.py

        Parameters:
            None

        Returns:
            None
        """
        logging.info("=================================")
        logging.info("--> Saving frozen model...please...wait!")

        freeze_aituring.flags.FLAGS.path_ckpt = os.path.join(self.paths['MODEL_OUTPUT'], self.args['MODEL_CKPTS'])
        freeze_aituring.flags.FLAGS.path_yaml = os.path.join(self.paths['TFRECORDS_DIR'], 'train_tfrecord_config.yaml')
        freeze_aituring.flags.FLAGS.path_output = self.paths['FROZEN_MODEL_DIR']
        freeze_aituring.flags.FLAGS.model_name_ = self.args['BACKBONE_REF']

        freeze_aituring.main('')

        logging.info("--> frozen model saved!")
        logging.info('')

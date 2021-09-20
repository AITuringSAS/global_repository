import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Append submodule path to python search path
sys.path.append(os.path.join(os.getcwd(), 'freeze'))
sys.path.append(os.path.join(os.getcwd(), 'tfrmod'))
sys.path.append(os.path.join(os.getcwd(), 'automl', 'efficientdet'))

# Tools to parse commands, zip and tar.gz file
import tarfile
from zipfile import ZipFile
from absl.flags import argparse_flags

# Custom submodules
from pipeline import command_interface
from automl.efficientdet.main import FLAGS as flags_efficientdet
from tfrmod.create_tfrecords import FLAGS as flags_tfrecord
from freeze.freeze_model import FLAGS as flags_freeze


def download_and_uncompress_backbone(backbone_name, backbone_url, backbone_save_dir):
    """
    define_parameters method
        This method download a pre-trained model 

        Parameters:
            backbone_name (str): model name
            backbone_url (str): backbone url
            backbone_save_dir (str): path to save backbone

        Returns:
            None
    """
    # Download backbone checkpoints
    os.system(' '.join(['wget', '--no-check-certificate', backbone_url, '-O',
                        os.path.join(backbone_save_dir, backbone_name + 'tar.gz')]))

    # Uncompress backbone .tar.gz files
    tar = tarfile.open(os.path.join(backbone_save_dir, backbone_name + 'tar.gz'))
    tar.extractall(backbone_save_dir)
    tar.close()

    # Delete .tar.gz backbone file
    os.remove(os.path.join(backbone_save_dir, backbone_name + 'tar.gz'))


def download_and_uncompress_dataset(dataset_url, save_dir, dataset_name):
    """
    Function to download and uncompress dataset from S3

    Parameters:
            dataset_url (str): url from S3
            save_dir (str): path to download dataset
            dataset_name (str): dataset filename to save

        Returns:
            None
    """

    # Download dataset from S3 (dataset must be public)
    os.system(' '.join(['wget', '--no-check-certificate',
                        dataset_url, '-O', dataset_name]))
    # Unzip dataset
    with ZipFile(dataset_name, 'r') as zipobj:
        zipobj.extractall(save_dir)
    # Delete .zip dataset
    os.remove(dataset_name)


def main():
    """Pipeline

    The following pipeline is the standard procedure 

    1. Set params definitions
    2. Set paths to workspace
    3. Download dataset from S3
    4. Download backbone checkpoint
    5. run create_tfrecord method
    6. run training method
    7. save frozen model 
    """

    # Setting default arguments:
    parser = argparse_flags.ArgumentParser(
        description='Command Line Tools EfficientDet Interface.', fromfile_prefix_chars='@')
    parser.add_argument('--URL_DATASET', help='url from S3.')
    parser.add_argument('--BACKBONE_REF', help='backbone name e.g. efficientdet-d0')
    parser.add_argument('--NUM_EPOCHS', help='number of epochs. e.g 100')
    parser.add_argument('--TRAIN_BATCH_SIZE', help='size of batch for training e.g. 6')
    parser.add_argument('--EVAL_BATCH_SIZE', help='size of batch for evaluation e.g. 6')
    parser.add_argument('--MODEL_CKPTS', help='path to save train model. (default: saved in workspace)')
    parser.add_argument('--NUM_EXAMPLES_PER_EPOCH', help='number of examples for training an epoch')
    
    # Dictionary for managing paths
    paths = {
        'WORKSPACE': os.path.join(os.getcwd(), 'workspace'),
        'DATASET_DIR': os.path.join(os.getcwd(), 'workspace', 'dataset'),
        'MODEL_OUTPUT_DIR': os.path.join(os.getcwd(), 'workspace', 'result'),
        'FROZEN_MODEL_DIR': os.path.join(os.getcwd(), 'workspace', 'frozen_model'),
        'TFRECORD_DIR': os.path.join(os.getcwd(), 'workspace', 'tfrecords'),
        'BACKBONE_CKPT_DIR': os.path.join(os.getcwd(), 'workspace', 'ckpt')
    }

    # Dictionary for managing files
    files = {
        'TFRECORD_SCRIPT': os.path.join(os.getcwd(), 'tfrmod', 'create_tfrecords.py'),
        'EFFICIENTDET_MAIN_SCRIPT': os.path.join(os.getcwd(), 'automl', 'efficientdet', 'main.py'),
        'FREEZE_MAIN_SCRIPT': os.path.join(os.getcwd(), 'freeze', 'freeze_model.py'),
        'HPARAMS_YAML': os.path.join(paths['TFRECORD_DIR'], 'tfrecord_config.yaml'),
        'DATASET_FILE': os.path.join(os.getcwd(), 'workspace', 'dataset', 'dataset.zip'),
    }

    # Dictionary for backbone url managment
    backbone_url = {
        'efficientdet-d0': 'https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d0.tar.gz',
        'efficientdet-d1': 'https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d1.tar.gz',
        'efficientdet-d2': 'https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d2.tar.gz',
        'efficientdet-d3': 'https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d3.tar.gz'
    }

    # check if not arguments are given
    # if not args, check if env parameters were given
    if len(sys.argv) < 2 and not all([os.getenv('URL_DATASET'), os.getenv('BACKBONE_REF'), os.getenv('NUM_EPOCHS'), os.getenv('TRAIN_BATCH_SIZE'), os.getenv('MODEL_CKPTS')]):
        parser.print_usage()
        sys.exit(1)
        
    # Setting default args to workspace
    flags_tfrecord.data_dir = paths['DATASET_DIR']
    flags_tfrecord.output_path = os.path.join(paths['TFRECORD_DIR'], 'tfrecord')

    flags_efficientdet.train_file_pattern = os.path.join(paths['TFRECORD_DIR'], 'tfrecord*.tfrecord')
    flags_efficientdet.hparams = files['HPARAMS_YAML']

    flags_freeze.path_yaml = files['HPARAMS_YAML']
        
    # Default args will be overridden if provided by command line
    args = parser.parse_args()
    
    # catch environment parameters if not detected by command line or config.file
    if not args.URL_DATASET:
        if "URL_DATASET" in os.environ:
            args.URL_DATASET = os.getenv('URL_DATASET')
        else:
            parser.print_usage()
            sys.exit(1)
    if not args.BACKBONE_REF:
        if "BACKBONE_REF" in os.environ:
            args.BACKBONE_REF = os.getenv('BACKBONE_REF')
        else:
            parser.print_usage()
            sys.exit(1)
    if not args.NUM_EPOCHS:
        if "NUM_EPOCHS" in os.environ:
            args.NUM_EPOCHS = int(os.getenv('NUM_EPOCHS'))
        else:
            parser.print_usage()
            sys.exit(1)
    if not args.TRAIN_BATCH_SIZE:
        if "TRAIN_BATCH_SIZE" in os.environ:
            args.TRAIN_BATCH_SIZE = int(os.getenv('TRAIN_BATCH_SIZE'))
        else:
            parser.print_usage()
            sys.exit(1)
    if not args.EVAL_BATCH_SIZE:
        if "EVAL_BATCH_SIZE" in os.environ:
            args.EVAL_BATCH_SIZE = int(os.getenv('EVAL_BATCH_SIZE'))
        else:
            parser.print_usage()
            sys.exit(1)
    if not args.MODEL_CKPTS:
        if "MODEL_CKPTS" in os.environ:
            args.MODEL_CKPTS = os.getenv('MODEL_CKPTS')
        else:
            args.MODEL_CKPTS = paths['MODEL_OUTPUT_DIR']
            sys.exit(1)
    if not args.NUM_EXAMPLES_PER_EPOCH:
        if "NUM_EXAMPLES_PER_EPOCH" in os.environ:
            args.NUM_EXAMPLES_PER_EPOCH = int(os.getenv('NUM_EXAMPLES_PER_EPOCH'))
        else:
            parser.print_usage()
            sys.exit(1)


    # Create workspace directories
    try:
        os.mkdir(paths['WORKSPACE'])
        os.mkdir(paths['DATASET_DIR'])
        os.mkdir(paths['MODEL_OUTPUT_DIR'])
        os.mkdir(paths['FROZEN_MODEL_DIR'])
        os.mkdir(paths['TFRECORD_DIR'])
        os.mkdir(paths['BACKBONE_CKPT_DIR'])
        os.mkdir(args.MODEL_CKPTS)
    except OSError:
        pass

    # Update flags provided by command line
    flags_efficientdet.model_name = args.BACKBONE_REF
    flags_efficientdet.backbone_ckpt = os.path.join(paths['BACKBONE_CKPT_DIR'], args.BACKBONE_REF)
    flags_efficientdet.num_epochs = args.NUM_EPOCHS
    flags_efficientdet.train_batch_size = args.TRAIN_BATCH_SIZE
    flags_efficientdet.eval_batch_size = args.EVAL_BATCH_SIZE
    flags_efficientdet.num_examples_per_epoch = args.NUM_EXAMPLES_PER_EPOCH

    # Get folder name for results
    path_ckpt = os.path.basename(os.path.normpath(args.MODEL_CKPTS))

    # Set output dir
    flags_freeze.path_ckpt = os.path.join(paths['MODEL_OUTPUT_DIR'], path_ckpt)
    flags_freeze.model_name_ = args.BACKBONE_REF
    if not flags_freeze.path_output:
        flags_freeze.path_output = os.path.join(paths['FROZEN_MODEL_DIR'], path_ckpt + '_freeze')


    # Download dataset from S3
    download_and_uncompress_dataset(args.URL_DATASET, paths['DATASET_DIR'], files['DATASET_FILE'])

    # Download backbone checkpoints
    download_and_uncompress_backbone(args.BACKBONE_REF, backbone_url[args.BACKBONE_REF], paths['BACKBONE_CKPT_DIR'])

    # Command Interface
    commandI = command_interface.Inteface(args_tfrecords=flags_tfrecord,
                                          args_efficientdet=flags_efficientdet,
                                          args_freeze=flags_freeze,
                                          paths=paths,
                                          files=files)

    # 1. Create tfrecord files
    commandI.create_tfrecord()

    # 2. Start trainning pipeline
    commandI.run_training()

    # 3. Save trained model
    commandI.save_frozen_model()


if __name__ == '__main__':
    main()

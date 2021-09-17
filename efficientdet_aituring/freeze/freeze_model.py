import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'automl', 'efficientdet'))
# sys.path.append(os.path.join(os.getcwd(), '..','automl', 'efficientdet'))
import inference
import yaml
import tensorflow.compat.v1 as tf
if tf.executing_eagerly():
    tf.compat.v1.disable_eager_execution()
from absl import app
from absl import flags
import shutil
from shutil import copyfile


# path_yaml = '/home/john_ruiz/tfrecords/PostersMXPatch_config.yaml'
# path_ckpt = '/home/john_ruiz/semillas/efficientdet-d2-MXNew'
# path_output = '/home/john_ruiz/semillas/efficientdet-d2-MXNew-freeze'

flags.DEFINE_string(
    'path_ckpt',
    default=None,
    help='Path to the trained checkpoint')
flags.DEFINE_string(
    'path_yaml',
    default=None,
    help='Path to the created yaml file, this file is created during training')
flags.DEFINE_string(
    'path_output',
    default=None,
    help='Path for saving the freeze model.')
flags.DEFINE_string(
    'model_name_',
    default='efficientdet-d2',
    help='Name of model to freeze.')

FLAGS = flags.FLAGS


def parse_from_yaml(yaml_file_path):
    """Parses a yaml file and returns a dictionary."""
    with tf.io.gfile.GFile(yaml_file_path, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)[0]
    return config_dict


def main(_):
    for filename in os.listdir(FLAGS.path_output):
        file_path = os.path.join(FLAGS.path_output, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    config = parse_from_yaml(FLAGS.path_yaml)
    # config['nms_configs'] = {
    #     'method': 'gaussian',
    #     'iou_thresh': None,  # use the default value based on method.
    #     'score_thresh': 0.0,
    #     'sigma': None,
    #     'max_nms_inputs': 0,
    #     'max_output_size': 100,
    # }
    driver = inference.ServingDriver(
        FLAGS.model_name_, FLAGS.path_ckpt, batch_size=1, model_params=config)
    driver.build()
    driver.export(FLAGS.path_output)
    copyfile(FLAGS.path_yaml, os.path.join(FLAGS.path_output, 'labelmaps_config.yaml'))
    path_output_classes, _ = os.path.split(FLAGS.path_yaml)
    if os.path.exists(os.path.join(path_output_classes, "Num_Classes.json")):
        copyfile(os.path.join(path_output_classes, "Num_Classes.json"),
                 os.path.join(FLAGS.path_output, 'num_classes.json'))


if __name__ == '__main__':
    app.run(main)

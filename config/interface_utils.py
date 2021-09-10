import os

config = {

    'TFRECORD_SCRIPT': os.path.join(*[os.getcwd(), 'tfRecordMod', "create_tfrecords.py"]),
    'TFRECORD_PARAMETERS': os.path.join(*[os.getcwd(), 'config', '1-tfrecord-parameters.txt']),

    'TRAINING_SCRIPT': os.path.join(*[os.getcwd(), 'automl', 'efficientdet', 'main.py']),
    'TRAINING_PARAMETERS': os.path.join(*[os.getcwd(), 'config', '2-training-parameters.txt'])

}

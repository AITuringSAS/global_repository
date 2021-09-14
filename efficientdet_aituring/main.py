
from facade_pattern import facade_interface


def main():
    """Pipeline

    The following pipeline is the standard procedure 

    1. Set params definitions
    2. Set paths to workspace
    3. Download dataset from S3
    4. Download backbone checkpoint
    5. Define create_tfrecord method
    6. Define run_training method
    7. Define save_frozen_model method
    """

    # 1-4. Interface
    pipeline = facade_interface.Inteface()

    # 5. Create tfrecord files
    pipeline.create_tfrecord()

    # 6. Start trainning pipeline
    pipeline.run_training()

    # 7. Save trained model
    pipeline.save_frozen_model()


if __name__ == '__main__':
    main()

from facade_pattern import facade_interface


def main():
    pipeline = facade_interface.Inteface()
    pipeline.create_tfrecord()
    pipeline.run_training()


if __name__ == '__main__':
    main()

    # URL_DATASET  *
    # BACKBONE_REF *
    # MODEL_CKPTS *
    # BATCH_SIZE  *
    # NUM_EPOCHS  *

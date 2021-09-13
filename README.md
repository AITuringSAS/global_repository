# Global Repository 
Repositorio global del equipo AITeam para el trabajo y desarrollo de una CNN basada en el modelo del EfficientDet de Google: [efficientdet-google](https://github.com/google/automl)

## Descripción

## Requisitos

### Windows

### Linux

### MacOX

## Instalación
`pip install -r requirements.txt`

### Instalación con conda

### Instalación Local

## Pipeline
```
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
```

## TFRecords
```
python3 tfRecordMod/create_tfrecords.py --data_dir=$temp_dir_dataset \
--output_path=$temp_dir_tfrecords/train_tfrecord \
--num_shards=10 \
--use_data_augmentation=True  \
--perc_split_training=1.0
```
## Training
```
python3 automl/efficientdet/main.py --mode=train \
--training_file_pattern=$temp_dir_tfrecords/train_tfrecord*.tfrecord \
--model_name=$BACKBONE_REF \
--model_dir=/workspace/saved_checkpoints/$MODEL_CKPTS \
--backbone_ckpt=/workspace/ckpts/$BACKBONE_REF \
--train_batch_size=$BATCH_SIZE \
--num_epochs=$NUM_EPOCHS \
--hparams=$temp_dir_tfrecords/train_tfrecord_config.yaml \
--num_examples_per_epoch=4455
```

## Frozen model
```
python3 freezeModelMod/freeze_aituring.py --path_ckpt /workspace/saved_checkpoints/$MODEL_CKPTS \
--path_yaml $temp_dir_tfrecords/train_tfrecord_config.yaml \
--path_output /workspace/saved_checkpoints/$MODEL_CKPTS$prefix_freeze \
--model_name $BACKBONE_REF
```

## Inference



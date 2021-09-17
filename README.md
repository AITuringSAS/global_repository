# Efficientdet Repositorio Refactorizado
Repositorio refactorizado del AITeam para el trabajo y desarrollo de una CNN basada en el modelo del EfficientDet de Google: [efficientdet-google](https://github.com/google/automl)

## Descripción
Este proyecto se basa en un patrón de diseño estructural Facade que permite ocultar la complejidad interna del proyecto [efficientdet-google](https://github.com/google/automl) y expone una interfaz simplificada de uso para el pipeline de entrenamiento, evaluación y manejo de datasets.

Este proyecto utiliza Tensorflow 2.5 y Python 3.8.

Tensorflow 2 requiere CPU con soporte AVX

Pipeline:

1. Conversión dataset de imágenes/labels a formato *TFRecord*
2. Entrenamiento 
3. Conversión modelo entrenado a FrozenModel

**Nota:**
Se recomienda el uso de entorno virtual conda/venv para el manejo de la instalación de paquetes

<img src="./uml/EfficientDetUML.png"/>

## Getting started
Se debe clonar el repositorio de manera **recursiva** para descargar los archivos correspondientes al submodulo [efficientdet-google-fork-version](https://git-codecommit.us-east-1.amazonaws.com/v1/repos/automl)
```
git clone --recursive https://git-codecommit.us-east-1.amazonaws.com/v1/repos/aituring_pipeline_efficientdet
```

## Requisitos

### Windows
- Microsoft C++ Build Tools: [link here](https://visualstudio.microsoft.com/visual-cpp-build-tools)
- wget: [link here](https://eternallybored.org/misc/wget/)
- Agregar a las variables de entorno (opcional) (si se usa miniconda3):
	- `C:\Users\USER-NAME\miniconda3\Scripts`
	- `C:\Users\USER-NAME\miniconda3\Library\bin`

**Note:**
Se debe agregar el path del folder en donde está *wget* a las variables de entorno de windows


### Linux
- gcc >= 9.3.0


## Instalación
Se recomienda el uso de un entorno aislado para la instalación de paquetes. Para este repositorio se utilizó [miniconda](https://docs.conda.io/en/latest/miniconda.html)

```
conda config --set auto_activate_base false
conda config --set env_prompt '({name})
```

### Instalación con conda
```
conda create --prefix=conda-env python=3.8.10
conda activate conda-env/
conda install tensorflow=2.5
conda install tensorflow-model-optimization=0.6 -c hcc
conda install lxml=4.6
conda install pycocotools=2.0.2
conda install opencv=4.5 -c conda-forge
conda install pyyaml=5.4 -c conda-forge
conda install configparser=5.0 -c conda-forge
```

### Instalación con PIP
```
conda create --prefix=conda-env python=3.8
pip install -r requirements.txt
```
**Nota:**
Conda también permite la carga de paquetes desde un archivo de requerimientos utilizando el comando:<br>

`conda install --file requirements.txt`

## Parametros
List de parametros por defecto
```
python3 main.py [-h/--help]
```
Lista completa de parametros
```
python3 main.py [--helpfull]
```


## Test

```
Training with command line parameters
-------
python3 main.py --URL_DATASET https://datasets-aiteam.s3.amazonaws.com/DATASET-5-FOTOS.zip \
--BATCH_SIZE 1 \
--BACKBONE_REF efficientdet-d0 \
--NUM_EPOCHS 5 \
--MODEL_CKPTS efficientdet-d0-output-folder


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
python3 main.py @params.config
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

## Eval
cooming soon...

## Frozen model
```
python3 freezeModelMod/freeze_aituring.py --path_ckpt /workspace/saved_checkpoints/$MODEL_CKPTS \
--path_yaml $temp_dir_tfrecords/train_tfrecord_config.yaml \
--path_output /workspace/saved_checkpoints/$MODEL_CKPTS$prefix_freeze \
--model_name $BACKBONE_REF
```

## Inference
coming soon...



## EfficientDet
EfficientDet Implementation in Keras focused on clean code and
readability. 
Training will be logged with Tensorboard. 
To take a look at the training progress do: **tensorboard --logdir logs**
This repo also includes the option of using wandb.ai for experiment tracking.

### Installation

#### Via PIP (recommended)
Split into 3 ways to install. This is due to the way tensorflow needs
to be installed to correctly work with CUDA. The first installation
does not include tensorflow and is recommended to use.

```shell
pip install efficient-det
```


You can include a [cpu] or [gpu] tag to include the respective tensorflow
version.
Includes tensorflow dependency:

#### Via Docker
Runs with tensorflow:2.3.0-gpu.
Depending on system CUDA version you might need to use
another version. [See this for more info](https://www.tensorflow.org/install/source#gpu
).

Install docker and nvidia container toolkit on host system
```shell
sudo apt-get install -y docker.io nvidia-container-toolkit
```

Build Docker Image

```shell
sudo docker build . -t edet
```


#### From Source
1. Clone Repository

```shell
git clone git@git.hhu.de:zeboz100/efficientdet.git
```

2. Build it

```shell
pip install -r requirements.txt
```


### Usage

#### PIP

Training

```shell
python3 -m efficient_det.run_training --dataset_path /path/to/dataset
```

Run a Hyperparameter Search:
```shell
python3 -m efficient_det.run_hyper_parameter_search --dataset_path 
/path/to/dataset --num_tries 100 --gpus_per_trial 0.5
```


#### Source
Execute all commands in **efficientdet/**

Set PYTHONPATH :

```shell
export PYTHONPATH="$PWD/src"
```

Training
```shell
python3 src/efficient_det/train.py --dataset_path /path/to/dataset/
```

Hyperparameter Search:
```shell
python3 src/efficient_det/train.py --dataset_path /path/to/dataset/
```

#### Docker

Run Container
```shell
sudo docker run --gpus all -it edet bash
```

and then proceed with the PIP instructions.

#### To run all tests

```shell
python3 -m unittest
```

#### To test loaded model
You can test the loaded model via notebook or from a script.


* Notebook in examples/visualize_rsults.ipynb

* You need to set dataset path
* You need to set path to trained model
  
Execute Script from efficientdet/
```shell
python3 example/visualize_results.py
```


# Do Better ImageNet Models Transfer Better?

Hi! We provide simple Tensorflow 2.0 implementations for paper "Do Better Imagnet models Transfer Better?".
There are 3 transfer-learning tasks in this implementations.
- As fixed extractors
	> L2-regularized Logistic Regression + L-BFGS without data augmentation
- Fine-tuned from ImageNet initialization
- Trained from random initialization

# How to use
## Enviroment

    pip install tensorflow-datasets
    conda install tensorflow-gpu

You can install  **tensorflow-gpu & datasets!**

## Parameter Setting
  
You need to change the parameters according to the data set in main.py. (NUM_TRAIN_SAMPLES, IMG_SHAPE, NUM_CLASSES). You can use 2 models & 6 datasets in this implementation or more.

    NUM_GPUS = 2
    BS_PER_GPU = 8
    NUM_EPOCHS = 80
    NUM_TRAIN_SAMPLES = 50000
    TASK=2
    MODEL = "mobilenet_v1" # mobilenet_v1, mobilenet_v2
    IMG_SHAPE = (32, 32, 3)
    NUM_CLASSES = 10
    DATASET = "cifar10" # food101, cifar10, cifar100, sun397, oxford_flowers102, caltech101

BATCH_SIZE = NUM_GPUS * BS_PER*GPU

## Model Summary
|  |  Parameter| Features|Image Size|Top-1acc|
|--|--|--|--|--|
| `MobileNet v1` |3.2M|1024|224|**72.4**|
|`MobileNet v2`|2.2M|1280|224|71.6|

# Result
## Paper
The higher the accuracy in ImageNet, the higher the performance in transfer learning.
| Dataset | Metric |
|--|--|
|`food101`  |top-1  |
|`cifar10`  |  top-1|
|`cifar100`  |  top-1|
|`sun397`  |  top-1|
|`oxford_flowers102`  | mean acc |
|`caltech101`  |mean acc |

### Task 1. Logistic Regression.
| Dataset | mobilenet v1 |mobilenet v2|
|--|--|--|
|`food101`  |  |win!|
|`cifar10`  |  |win!|
|`cifar100`  |  win!||
|`sun397`  | win!||
|`oxford_flowers102`  | win!||
|`caltech101`  | |win!|
### Task 2. Fine-Tuned
| Dataset | mobilenet v1 |mobilenet v2|
|--|--|--|
|`food101`  |win!  ||
|`cifar10`  | win! ||
|`cifar100`  | win! ||
|`sun397`  | win!||
|`oxford_flowers102`  | win!||
|`caltech101`  |win! ||
### Task 3. Trained from Random Initialization
| Dataset | mobilenet v1 |mobilenet v2|
|--|--|--|
|`food101`  |win!  ||
|`cifar10`  |  win!||
|`cifar100`  | win! ||
|`sun397`  |win! ||
|`oxford_flowers102`  |win! ||
|`caltech101`  |win! ||

## My Result.
### Mobilenet v1 vs v2 - acc
| Dataset | Task1 v1 |Task1 v2|Task2 v1 |Task2 v2|Task3 v1 |Task3 v2|
|--|--|--|--|--|--|--|
|`food101`  | - |-| - |-  |- |- |
|`cifar10`  | - |-| **0.8257** |0.8164  |- |- |
|`cifar100`  | - |-| - |-  |- |- |
|`sun397`  | - |-| - |-  |- |- |
|`oxford_flowers102`  | - |-| - |-  |- |- |
|`caltech101`  | - |-| - |-  |- |- |
### Statistical Methods - log odds
| Dataset | Task1 v1 |Task1 v2|Task2 v1 |Task2 v2|Task3 v1 |Task3 v2|
|--|--|--|--|--|--|--|
|`food101`  | - |-| - |-  |- |- |
|`cifar10`  | - |-| **1.5554** |1.4921  |- |- |
|`cifar100`  | - |-| - |-  |- |- |
|`sun397`  | - |-| - |-  |- |- |
|`oxford_flowers102`  | - |-| - |-  |- |- |
|`caltech101`  | - |-| - |-  |- |- |

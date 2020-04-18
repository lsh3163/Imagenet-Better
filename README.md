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
  
You need to change the parameters according to the data set in main.py or Implementation.ipynb. (NUM_TRAIN_SAMPLES, IMG_SHAPE, NUM_CLASSES). You can use 2 models & 6 datasets in this implementation or more.

    NUM_GPUS = 4
    BS_PER_GPU = 64  # Batchsize = 4x64
    NUM_EPOCHS = 200
    TASK=2  # Task 1 : Logistic Regression, Task2 : Transfer learning, Task3 : Random Initialization(Scratch Training)
    MODEL = "mobilenet_v2"  # mobilenet_v1, mobilenet_v2
    DATASET = "cifar10"  # food101, cifar10, cifar100, sun397, oxford_flowers102, caltech101
    learning_rate = 0.01


## Model Summary
|  |  Parameter| Features|Image Size|Top-1acc / Retrained|Top-1acc / Paper |
|--|--|--|--|--|--|
| `MobileNet v1` |3.2M|1024|224|**72.4**|70.4
|`MobileNet v2`|2.2M|1280|224|71.6|**72.0**

# Result
## Paper
The higher the accuracy in ImageNet, the higher the performance in transfer learning.
| Dataset | Metric |Classes|Size(train/test)|
|:--:|:--:|:--:|:--:|
|`food101`  |top-1  |101|75750/25250|
|`cifar10`  |  top-1|10|50000/10000|
|`cifar100`  |  top-1|100|50000/10000|
|`sun397`  |  top-1|397|19850/19850|
|`oxford_flowers102`  | mean acc |102|2040/6149|
|`caltech101`  |mean acc |102|3060/6084|

### Task 1. Logistic Regression.
| Dataset | mobilenet v1 |mobilenet v2|
|:--:|:--:|:--:|
|`food101`  |  |win!|
|`cifar10`  |  |win!|
|`cifar100`  |  win!||
|`sun397`  | win!||
|`oxford_flowers102`  | win!||
|`caltech101`  | |win!|
### Task 2. Fine-Tuned
| Dataset | mobilenet v1 |mobilenet v2|
|:--:|:--:|:--:|
|`food101`  |win!  ||
|`cifar10`  | win! ||
|`cifar100`  | win! ||
|`sun397`  | win!||
|`oxford_flowers102`  | win!||
|`caltech101`  |win! ||
### Task 3. Trained from Random Initialization 
| Dataset | mobilenet v1 |mobilenet v2|
|:--:|:--:|:--:|
|`food101`  |win!  ||
|`cifar10`  |  win!||
|`cifar100`  | win! ||
|`sun397`  |win! ||
|`oxford_flowers102`  |win! ||
|`caltech101`  |win! ||

## My Result.
### 1. Tensorflow 2.0

#### Setting
- [x] Preprocessing - (Scale to [-1, 1])
- [ ] Dropout
- [ ] Data Augmentation
- [x] Optimizer - SGD (Momentum 0.9, Nestrov True, Weight Decay 1e-6)
- [x] Batchsize - 256 
- [x] Image Size - 224 x 224 (except cifar10, 100 - 32 x 32)
- [x] Epoch : 200 (~10000 iterations)  
 
#### Mobilenet v1 vs v2 - acc
| Dataset | Task1 v1 |Task1 v2|Task2 v1 |Task2 v2|Task3 v1 |Task3 v2|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|`food101`  | - |-| - |-  |- |- |
|`cifar10`  | 0.2185 |**0.2868**| **0.8257** |0.8164  |**0.7788**|0.7635 |
|`cifar100`  | 0.0561 |**0.0936**| **0.5752** |0.5414  |**0.3015** |0.2861 |
|`sun397`  | - |-| - |-  |- |- |
|`oxford_flowers102`  | - |-| **0.5572** |0.5250  |- |- |
|`caltech101`  | - |-| **0.8664** |0.8616  |0.5303 |**0.5408** |

#### Statistical Methods - log odds
| Dataset | Task1 v1 |Task1 v2|Task2 v1 |Task2 v2|Task3 v1 |Task3 v2|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|`food101`  | - |-| - |-  |- |- |
|`cifar10`  | -1.2744 |**-0.9109**| **1.5554** |1.4921  |**1.2586** |1.1719|
|`cifar100`  | -2.8224 |**-2.2704**| **0.3030** |0.1659  |**-0.8401** |-0.9144 |
|`sun397`  | - |-| - |-  |- |- |
|`oxford_flowers102`  | - |-| **0.2298** |0.1000  |- |- |
|`caltech101`  | - |-| **1.8694** |1.8286  |0.1213 |**0.1635** |

### 2. Pytorch
#### Setting
- [x] Preprocessing - (Scale to [0, 1])
- [ ] Dropout
- [ ] Data Augmentation
- [x] Optimizer - SGD (Momentum 0.9, Nestrov True, Weight Decay 1e-6)
- [x] Batchsize - 128 
- [x] Image Size - 224 x 224 
- [x] Epoch : 100 (~10000 iterations)  
 
#### Mobilenet v1 vs v2 - acc
| Dataset | Task1 v1 |Task1 v2|Task2 v1 |Task2 v2|Task3 v1 |Task3 v2|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|`food101`  | - |-| - |-  |- |- |
|`cifar10`  | - |-| - |-  |0.8910 |- |
|`cifar100`  | - |-| - |-  |- |- |
|`sun397`  | - |-| - |-  |- |- |
|`oxford_flowers102`  | - |-| - |-  |- |- |
|`caltech101`  | - |-| - |-  |- |- |

#### Statistical Methods - log odds
| Dataset | Task1 v1 |Task1 v2|Task2 v1 |Task2 v2|Task3 v1 |Task3 v2|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|`food101`  | - |-| - |-  |- |- |
|`cifar10`  | - |-| - |-  |2.100 |- |
|`cifar100`  | - |-| - |-  |- |- |
|`sun397`  | - |-| - |-  |- |- |
|`oxford_flowers102`  | - |-| - |-  |- |- |
|`caltech101`  | - |-| - |-  |- |- |
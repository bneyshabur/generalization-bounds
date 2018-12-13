# Exploring Generalization in Deep Learning
This repository contains the code to train convolutional and fully connected networks and compute various measures and generalization bounds on convolutional and fully connected networks. The architecture, training procedure and some of the measures are chosen the same the following paper:

**[Exploring Generalization in Deep Learning.](https://arxiv.org/abs/1805.12076)**  
[Behnam Neyshabur](https://www.neyshabur.net), [Srinadh Bhojanapalli](http://ttic.uchicago.edu/~srinadh/), [David McAllester](http://ttic.uchicago.edu/~dmcallester/), [Nathan Srebro](http://www.ttic.edu/srebro).  
Neural Information Processing Systems (NIPS), 2017


## Usage
1. Install *Python 3.6* and *PyTorch 0.4.1*.
2. Clone the repository:
   ```
   git clone https://github.com/bneyshabur/over-parametrization.git
   ```
3. As a simple example, the following command trains a VGG-like architecture on *CIFAR10* dataset and then computes several measures/bounds on the learned network:
   ```
   python main.py --dataset CIFAR10 --model vgg
   ```
## Main Inputs Arguments
* `--no-cuda`: disables cuda training
* `--datadir`: path to the directory that contains the datasets (default: datasets)
* `--dataset`: name of the dataset(options: MNIST | CIFAR10 | CIFAR100 | SVHN, default: CIFAR10). If the dataset is not in the desired directory, it will be downloaded.
* `--model`: architecture(options: vgg | fc, default: vgg). You can add your own model to the `models` directory.

## Reported Measures
After training the network, following measures can be computed. Please see the file `measures.py` for explanation of each measure:
* `L_pq norm`: product of Lpq norm of layers
* `L_p-path norm`: Lp-path norm of the network
* `L_p operator norm`: product of Lp norm of eigen-values of layers

We also compute and report the following generalization bounds:
* `L_1,inf bound`: Generalization bound by Bartlett and Mendelson 2002 (depth dependency from Golowich et al. 2018).
* `L_3,1.5 bound`: Generalization bound by Neyshabur et al. 2015 (depth dependency from Golowich et al. 2018).
* `Frobenious bound`: Generalization bound by Neyshabur et al. 2015 (depth dependency from Golowich et al. 2018).
* `Spec_L1 bound`: Generalization bound by Bartlett et al. 2017.
* `Spec_Fro bound`: Generalization bound by Neyshabur et al. 2018.

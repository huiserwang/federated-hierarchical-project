# Game Theory Project

Experiments are produced on MNIST and CIFAR10.

## Requirments
Install all the packages from requirements.txt:

```
pip -r install requirements.txt
```

* Python >= 3.6
* Pytorch >= 1.2.0
* Torchvision >= 0.4.0
* Numpy>=1.15.4

## Data
* Download train and test datasets on your own, otherwise they will be automatically downloaded from torchvision datasets.
* Experiments are run on MNIST and Cifar.

## Running the experiments

Federated experiment involves training a global model using many local models.

* To run the hierarchical federated learning experiment with CIFAR on CNN:
```
python federated_hierarchical_main.py --model=cnn --dataset=cifar --gpu=0 --local_ep 2 --epochs=10
```

- To run the hierarchical federated learning experiment with CIFAR on MLP:

```
python federated_hierarchical_main.py --model=mlp --dataset=cifar --gpu=0 --local_ep 2 --epochs=10
```

- To run the hierarchical federated learning experiment with MNIST on CNN:

```
python federated_hierarchical_main.py --model=cnn --dataset=mnist --gpu=0 --local_ep 2 --epochs=10
```

- To run the hierarchical federated learning experiment with MNIST on MLP:

```
python federated_hierarchical_main.py --model=mlp --dataset=mnist --gpu=0 --local_ep 2 --epochs=10
```

You can change the default values of other parameters to simulate different conditions. Refer to the options section.

## Options
The default values for various paramters parsed to the experiment are given in ```options.py```. Details are given some of those parameters:

* ```--dataset:```  Default: 'mnist'. Options: 'mnist',  'cifar'
* ```--model:```    Default: 'mlp'. Options: 'mlp', 'cnn'
* ```--gpu:```      Default: None (runs on CPU). Can also be set to the specific gpu id (i.e. 0).
* ```--epochs:```   Default: 10. Number of rounds of training.
* ```--lr:```       Learning rate set to 0.01 by default.
* ```--verbose:```  Detailed log outputs. Activated by default, set to 0 to deactivate.
* ```--seed:```     Random Seed. Default set to 1.

#### Federated Parameters
* ```--num_fv:``` Number of FL Vehicles. Default is 6.

* ```--num_fr:``` Number of FL RSUs. Default is 5.

* ```--local_ep:``` Number of local training epochs in each user. Default is 2.

* ```--local_bs:``` Batch size of local updates in each user. Default is 32.


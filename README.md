# Distributed Training with Tensorflow

The machine learning training tasks are both computation expensive as well as time consuming, especially the models with millions or sometimes
billions of parameters. Also the size of training dataset used for such huge models is in GBs or TBs. Such types of models which are memory intensive are not 
quite possible to train with traditiona means. Distributed training is one of the solutions to handle such problem. Now we got speciallized hardwares likes GPUs and TPUs
which makes the model training process much faster as they are designed to do parallel processing.

In case of distributed training we got two main types:

**1. Data parallelism** - Model itself is replicated over multiple devices and each replica is trained on slice of batch of data \
**2. Model parallelism** - If the model size is too huge to contain on single device, we shard the model and distribute it over multiple devices.

In this post we are going to discuss about Data parallelism options available in tensorflow.

In data parallelism we have two options:
**1. Synchronous** - In synchronous training each replica receives a different slice of input batch and trains only on that data laters all the gradients are aggregated and updated.
**2. Asynchronous** - In asynchronous training all workers train independently and variables are updated asynchronously.

In order to support distributed training, TensorFlow has *MirroredStrategy, TPUStrategy, MultiWorkerMirroredStrategy, ParameterServerStrategy, CentralStorageStrategy*, as well as other strategies available.

In this post we will cover the synchronous methods in tensorflow.

- ### MirroredStrategy

tf.distribute.MirroredStrategy supports synchronous distributed training on multiple GPUs on one machine. It creates one replica per GPU device. Each variable in the model is mirrored across all the replicas. Together, these variables form a single conceptual variable called MirroredVariable. These variables are kept in sync with each other by applying identical updates. \
Efficient all-reduce algorithms are used to communicate the variable updates across the devices. All-reduce aggregates tensors across all the devices by adding them up, and makes them available on each device. \
NCCL is the best all reduce if you are using multiple GPUs for training.

The below steps are carried out in a typical MirroredStrategy execution flow:
- All the variables and the model graph is replicated on the replicas.
- Input is evenly distributed across the replicas.
- Each replica calculates the loss and gradients for the input it received.
- The gradients are synced across all the replicas by summing them.
- After the sync, the same update is made to the copies of the variables on each replica.






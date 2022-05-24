# Distributed Training with Tensorflow

The machine learning training tasks are both computation expensive as well as time consuming, especially the models with millions or sometimes
billions of parameters. Also the size of training dataset used for such huge models is in GBs or TBs. Such types of models which are memory intensive are not 
quite possible to train with traditional means. Distributed training is one of the solutions to handle such problem. Now we got speciallized hardware's likes GPUs and TPUs
which makes the model training process much faster as they are designed to do parallel processing.

In case of distributed training, we got two main types:

**1. Data parallelism** - Model itself is replicated over multiple devices and each replica is trained on slice of batch of data \
**2. Model parallelism** - If the model size is too huge to contain on single device, we shard the model and distribute it over multiple devices.

In this post we are going to discuss about Data parallelism options available in tensorflow.

In data parallelism we have two options:
**1. Synchronous** - In synchronous training each replica receives a different slice of input batch and trains only on that data laters all the gradients are aggregated and updated. \
**2. Asynchronous** - In asynchronous training all workers train independently and variables are updated asynchronously.

In order to support distributed training, TensorFlow has *MirroredStrategy, TPUStrategy, MultiWorkerMirroredStrategy, ParameterServerStrategy, CentralStorageStrategy*, as well as other strategies available.

In this post we will cover the synchronous methods in tensorflow.

- ### MirroredStrategy

tf.distribute.MirroredStrategy supports synchronous distributed training on multiple GPUs on one machine. It creates one replica per GPU device. Each variable in the model is mirrored across all the replicas. Together, these variables form a single conceptual variable called MirroredVariable. These variables are kept in sync with each other by applying identical updates. \
Efficient all-reduce algorithms are used to communicate the variable updates across the devices. All-reduce aggregates tensors across all the devices by adding them up, and makes them available on each device. \
NCCL is the best all reduce if you are using multiple GPUs for training.

The below steps are carried out in a typical MirroredStrategy execution flow:
> > - All the variables and the model graph is replicated on the replicas.
> > - Input is evenly distributed across the replicas.
> > - Each replica calculates the loss and gradients for the input it received.
> > - The gradients are synced across all the replicas by summing them.
> > - After the sync, the same update is made to the copies of the variables on each replica.


- ### MultiWorkerMirroredStrategy

The MultiWorkerMirroredStrategy support the model training over multiple machines essentially with multiple GPUs attached to them. This strategy is quite similar to Mirrored strategy as per execution is considered. One thing that we need to specify in this scenario is **TF_CONFIG** environment variable. This variable must be set in the environment variables which will direct the program for using multiple workers. \
There are two components of a TF_CONFIG variable: 'cluster' and 'task' \
- A 'cluster' is the same for all workers and provides information about the training cluster, which is a dict consisting of different types of jobs, such as 'worker' or 'chief'. 
- A 'task' provides information on the current task and is different for each worker. It specifies the 'type' and 'index' of that worker. \

Here is an example for TPU_CONFIG :

```
tf_config = {
    'cluster': {
        'worker': ['localhost:12345', 'localhost:23456']
    },
    'task': {'type': 'worker', 'index': 0}
}
```
In case you are using **VertexAI** - The ML service provided by Google Cloud Platform, you don’t have to worry about this environment variable, it is internally set by VertexAI itself. \
Another most important thing to remember while using MultiWorkerMirroredStrategy is we need to initiate the strategy at the beginning of the programs and all the tensor related operations must be carried out after the initialization.\
If you encounter *RuntimeError: Collective ops must be configured at program startup*, try creating the instance of MultiWorkerMirroredStrategy at the beginning of the program and put the code that may create ops after the strategy is instantiated.


- ### TPUStrategy

TPUs are specialized hardware developed by Google to carry out the heavy ML workloads faster. Currently TPUs are available in GCP, Google Colab and Kaggle. TPUs are much faster as compared with the some contemporary GCPs, So that is saves time as well cost. Most of the google services like translate, Photos and Gmail are powered by TPUs. \
If you want to train your model using TPUs there TPUStrategy in tensorflow. \
In terms of distributed training architecture, TPUStrategy is the same MirroredStrategy—it implements synchronous distributed training. TPUs provide their own implementation of efficient all-reduce and other collective operations across multiple TPU cores, which are used in TPUStrategy. \
If you want to run TPU training on Google Colab you have to change the runtime to Use TPUs. \
If you are running the ML workload in any GCP service like **AIPlatform** or **VertexAI**, you just have to select the hardware with appropriate TPUs available in the region and rest is the same. \
GCP also provides TPU nodes in that case you may need to specify the name of the tpu or 'local' in TPUClusterResolver. In case of GCP or Colab we don't need to. \
One more thing which is important is it is good practice to initialize the TPUStrategy at the beginning of the code to avoid any errors.
 

#### Notes:
- I have added simple mnist model training using above mentioned strategies in code, In case of multiworker you have set *TF_CONFIG* environment variable, if you are using GCP services like VertexAI you can skip the *TF_CONFIG*. \
Above code is tested on *python=3.8*. \
In case you are using custom training you need to specify the reduction technique in case of losses.
- For an example, let's say you have 4 GPU's and a batch size of 64. One batch of input is distributed across the replicas (4 GPUs), each replica getting an input of size 16.
- The model on each replica does a forward pass with its respective input and calculates the loss. Now, instead of dividing the loss by the number of examples in its respective input (BATCH_SIZE_PER_REPLICA = 16), the loss should be divided by the GLOBAL_BATCH_SIZE (64).
- This needs to be done because after the gradients are calculated on each replica, they are synced across the replicas by summing them.
- If using tf.keras.losses classes (as in the example below), the loss reduction needs to be explicitly specified to be one of NONE or SUM. AUTO and SUM_OVER_BATCH_SIZE are disallowed when used with tf.distribute.Strategy.


#### References:
https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras \
https://www.tensorflow.org/tutorials/distribute/keras \
https://www.tensorflow.org/guide/tpu \
https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy \
https://www.tensorflow.org/guide/distributed_training











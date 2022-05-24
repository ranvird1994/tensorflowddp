import tensorflow as tf
import tensorflow_datasets as tfds

print(tf.__version__)

resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
tpu_strategy = tf.distribute.TPUStrategy(resolver)

datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)

train, test = datasets['train'], datasets['test']

BUFFER_SIZE = 1000
BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

def preprocess_image(img, lbl):
    img = tf.cast(img, tf.float32)/255
    return img, lbl

train_dataset = train.map(preprocess_image).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
val_dataset = test.map(preprocess_image).batch(BATCH_SIZE)

def model():
    return tf.keras.Sequential([
              tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
              tf.keras.layers.MaxPooling2D(),
              tf.keras.layers.Conv2D(32, 3, activation='relu'),
              tf.keras.layers.MaxPooling2D(),
              tf.keras.layers.Conv2D(16, 3, activation='relu'),
              tf.keras.layers.MaxPooling2D(),
              tf.keras.layers.Flatten(),
              tf.keras.layers.Dense(64, activation='relu'),
              tf.keras.layers.Dense(10)
          ])

with tpu_strategy.scope():
    cnn_model = model()
    cnn_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])

cnn_model.fit(train_dataset,
              epochs=1,
              validation_data=eval_dataset,
              steps_per_epoch=len(train_dataset),
              validation_steps=len(eval_dataset))

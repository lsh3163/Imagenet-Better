import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
import numpy as np
import tensorflow_datasets as tfds

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)

NUM_GPUS = 8
BS_PER_GPU = 4
NUM_EPOCHS = 200
TASK=2
MODEL = "mobilenet_v1" # mobilenet_v1, mobilenet_v2
NUM_CLASSES = 397
DATASET = "sun397" # food101, cifar10, cifar100, sun397, oxford_flowers102, caltech101

def normalize_img(image, label):
    image = tf.image.resize(image, (224, 224))
    return tf.cast(image, tf.float32) / 127.5 - 1, label

if TASK==1:
    include_top=False
    weights="imagenet"
    trainable=False
elif TASK==2:
    include_top=False
    weights="imagenet"
    trainable=True
elif TASK==3:
    include_top=False
    weights=None
    trainable=True


if MODEL=="mobilenet_v1":
    base_model = tf.keras.applications.MobileNet(input_shape=None,
                                            include_top=include_top,
                                            weights=weights)
elif MODEL=="mobilenet_v2":
    base_model = tf.keras.applications.MobileNetV2(input_shape=None,
                                                 include_top=include_top,
                                                 weights=weights)

# Freeze the pre-trained model weights
base_model.trainable = trainable

# Trainable classification head
maxpool_layer = tf.keras.layers.GlobalMaxPooling2D()
prediction_layer = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')

# Layer classification head with feature detector
model = tf.keras.Sequential([
    base_model,
    maxpool_layer,
    prediction_layer
])

learning_rate = 0.01
# Compile the model
metrics = [tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name="top_K")]
# metrics = ["acc"]
model.compile(#optimizer=tf.keras.optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True),
              optimizer=tf.keras.optimizers.Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
              loss='sparse_categorical_crossentropy',
              metrics=metrics
)
print(model.summary())

# (x, y), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
(train_dataset, test_dataset), ds_info = tfds.load(
    DATASET,
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)


ds_train = train_dataset.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(BS_PER_GPU * NUM_GPUS)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = test_dataset.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(BS_PER_GPU * NUM_GPUS)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)


history = model.fit(ds_train,
          validation_data=ds_test,
          validation_freq=1,
          epochs=NUM_EPOCHS, shuffle=True
          # callbacks=[tf.keras.callbacks.TensorBoard(log_dir="./", update_freq="batch", histogram_freq=1)]
)



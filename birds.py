import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import glob

imgs_path = glob.glob("./birds/*/*.jpg")
# imgs_path[:5]
# len(imgs_path)
# img_p = imgs_path[1000]
# img_p.split('/')[2].split(".")[1]
all_labels = [img_p.split('/')[2].split(".")[1] for img_p in imgs_path]

label_names = np.unique(all_labels)
# len(label_names)
label_names_enu = enumerate(label_names)
label_to_index = dict((i, k) for (k, i) in label_names_enu)
index_to_label = dict((k, i) for (i, k) in label_to_index.items())

all_labels = [label_to_index.get(name) for name in all_labels]

np.random.seed(2022)
random_index = np.random.permutation(len(imgs_path))

imgs_path = np.array(imgs_path)[random_index]
all_labels = np.array(all_labels)[random_index]

i = int(len(imgs_path)*0.2)
train_path = imgs_path[i:]
train_label = all_labels[i:]
test_path = imgs_path[:i]
test_label = all_labels[:i]

train_ds = tf.data.Dataset.from_tensor_slices((train_path, train_label))
test_ds = tf.data.Dataset.from_tensor_slices((test_path, test_label))

def load_img(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image,(256, 256))
    image = tf.cast(image, tf.float32)
    image = image / 255
    return image, label

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.map(load_img,num_parallel_calls=AUTOTUNE)
test_ds = test_ds.map(load_img,num_parallel_calls=AUTOTUNE)

BATCH_SIZE = 32
train_ds = train_ds.repeat().shuffle(300).batch(BATCH_SIZE)
test_ds = test_ds.batch(BATCH_SIZE)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(64,(3,3), input_shape=(256,256,3),activation="relu"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(64,(3,3),activation="relu"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(128,(3,3),activation="relu"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(128,(3,3),activation="relu"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(512,(3,3),activation="relu"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(512,(3,3),activation="relu"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(512,(3,3),activation="relu"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(512,(3,3),activation="relu"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(512,(3,3),activation="relu"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Dense(1024, activation="relu"))
model.add(tf.keras.layers.Dense(200))

model.summary()

model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["acc"]
        )

STEPS_PER_EPOCH = int(len(train_path)//BATCH_SIZE)
VAL_STEPS_PER_EPOCH = int(len(test_path)/BATCH_SIZE)
history = model.fit(
        train_ds,
        batch_size=BATCH_SIZE,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=test_ds,
        validation_steps=VAL_STEPS_PER_EPOCH
        )

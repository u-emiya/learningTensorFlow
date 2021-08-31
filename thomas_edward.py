import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE

import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import keras

from tensorflow import keras

import pathlib
data_dir = pathlib.Path('/Users/uemiya/.keras/datasets/train_animation/character')
print(data_dir)
print(type(data_dir))
image_count= len(list(data_dir.glob('*/*.jpg')))
print(image_count)
#image_count is 200

CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])
print(CLASS_NAMES)
#['edward' 'thomas']


BATCH_SIZE = 10
IMG_HEIGHT = 192
IMG_WIDTH = 192
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)
TOTAL_EPOCH=15



image_generator=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     classes = list(CLASS_NAMES))
image_batch, label_batch = next(train_data_gen)

def show_batch(image_batch, label_batch):
  plt.figure(figsize=(10,10))
  for n in range(32):
      ax = plt.subplot(6,6,n+1)
      plt.imshow(image_batch[n])
      plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
      plt.axis('off')
  plt.show()

#show_batch(image_batch, label_batch)
list_ds=tf.data.Dataset.list_files(str(data_dir/'*/*'))

def get_label(file_path):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  #label_to_index.get('daisy')

  return  keras.backend.cast(parts[-2] == CLASS_NAMES,"int64")

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def decode_label(label):
    return keras.backend.argmax(label)

def process_path(file_path):

  label = get_label(file_path)
  label=decode_label(label)

  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)

  return img, label

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)


def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
  # This is a small dataset, only load it once, and keep it in memory.
  # use `.cache(filename)` to cache preprocessing work for datasets that don't
  # fit in memory.
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()

  ds = ds.shuffle(buffer_size=shuffle_buffer_size)

  # Repeat forever
  ds = ds.repeat()

  ds = ds.batch(BATCH_SIZE)

  # `prefetch` lets the dataset fetch batches in the background while the model
  # is training.
  ds = ds.prefetch(buffer_size=AUTOTUNE)

  return ds

train_ds = prepare_for_training(labeled_ds)
#show_batch(image_batch.numpy(), label_batch.numpy())
mobile_net=tf.keras.applications.MobileNetV2(input_shape=(192,192,3),include_top=False)
mobile_net.trainable=False

def change_range(image,label):
    return 2*image-1, label

keras_ds = train_ds.map(change_range)

model = tf.keras.Sequential([
    mobile_net,
    tf.keras.layers.GlobalAveragePooling2D(),
    #tf.keras.layers.Dropout(0.5 ),
    tf.keras.layers.Dense(128,kernel_regularizer=keras.regularizers.l2(0.001),activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(CLASS_NAMES),activation='softmax')
])
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])

from tensorflow.keras.callbacks import EarlyStopping
early_stop=keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=3)


train_history=model.fit(keras_ds,validation_data=next(iter(keras_ds)),epochs=TOTAL_EPOCH,
 steps_per_epoch=STEPS_PER_EPOCH,verbose=1,callbacks=[early_stop])

#model.save('thomas_edward.h5')

test_dir=pathlib.Path('/Users/uemiya/.keras/datasets/train_animation/test')
test_image_count= len(list(test_dir.glob('*/*.jpg')))

test_ls_ds=tf.data.Dataset.list_files(str(test_dir/'*/*'))
test_lbl_ds = test_ls_ds.map(process_path, num_parallel_calls=AUTOTUNE)

test_ds = prepare_for_training(test_lbl_ds)

test_ds = test_ds.map(change_range)
test_img_batch, test_lbl_batch = next(iter(test_ds))

test_loss,test_acc=model.evaluate(test_img_batch, test_lbl_batch ,verbose=2)
print('\nTest accuracy:',test_acc)

history_dict=train_history.history
print(history_dict.keys())

print(len(train_history.epoch))

plt.plot(range(1, len(train_history.epoch)+1), train_history.history['accuracy'], "-o")
plt.plot(range(1, len(train_history.epoch)+1), train_history.history['val_accuracy'], "-o")
plt.title('model accuracy')
plt.ylabel('accuracy')  # Y軸ラベル
plt.xlabel('epoch')  # X軸ラベル
plt.grid()
plt.legend(['accuracy', 'val_accuracy'], loc='best')
plt.show()

from PIL import Image

pre = model.predict(test_img_batch, batch_size=BATCH_SIZE,verbose=1)
i=0
for label  in test_lbl_batch:
  print("Label      : ", label.numpy())
  print("probability: ", np.argmax(pre[i]))
  print("probability: ", pre[i])
  i=i+1

import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE

import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import keras

from tensorflow import keras

'''
get_file...origin=url,fname=ファイルの名前,untar=取得するファイルを解凍するかどうか
data_root_origにダウンロードしたデータのパスが保存される。
'''
import pathlib
data_dir = tf.keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
                                         fname='flower_photos', untar=True)
print(data_dir)
print(type(data_dir))

data_dir = pathlib.Path(data_dir)
print(data_dir)
print(type(data_dir))
image_count= len(list(data_dir.glob('*/*.jpg')))
#image_count is 3670

CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])
#['roses' 'sunflowers' 'daisy' 'dandelion' 'tulips']


'''
定義を行う
np.ceil...切り上げ、つまり3670/32=114.6…　→115となる
'''
BATCH_SIZE = 32
IMG_HEIGHT = 192
IMG_WIDTH = 192
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)
TOTAL_EPOCH=15



'''
画像を読み込んでいるらしい
全ての画像をイテレータとして読み込んでいる。batch_size,target_sizeなどを指定することで、
next()メソッドで読み込む際の量を定める（のかな？）
'''
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
'''
全ての画像のパスを読み込む
'''
list_ds=tf.data.Dataset.list_files(str(data_dir/'*/*'))
'''
画像パスの表示
for f in list_ds.take(10):
    print(f.numpy())
'''
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

'''
list_dsには画像のパスが保存されている。
list_ds.mapにより、入力データセット(画像のパス)の各要素にprocess_pathメソッドを適用させている
この前処理はmap変換により並列処理が実現され高速化となる。
labeled_dsには3670全てのデータが入っていて、
imageには画像データが、labelにはその画像のインデックスがtrue、他はfalseの配列が格納されている。
'''
# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

'''
画像データの形とラベルの表示
ex)
Image shape:  (192, 192, 3)
Label:  4
Image shape:  (192, 192, 3)
Label:  3

for image, label in labeled_ds.take(2):
  print("Image shape: ", image.numpy().shape)
  print("Label: ", label.numpy())
'''

'''
train_dataを指定したバッチサイズでランダムに選び出し用意するメソッド
'''
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

'''
prepare_for_trainingメソッドによって､どのような形にデータセットがなったのか確認
ex)
0
Image shape:  (32, 192, 192, 3)
Label:  [3 4 0 2 2 0 4 1 4 2 2 1 2 3 3 3 3 3 1 0 2 4 0 4 3 3 3 1 1 3 0 4]
1
Image shape:  (32, 192, 192, 3)
Label:  [0 0 4 2 3 3 3 0 4 4 1 4 3 2 0 2 4 4 3 4 1 1 4 3 1 1 2 4 4 4 3 2]

print('train_ds')
i=0
for image, label in train_ds.take(2):
  print(i)
  print("Image shape: ", image.numpy().shape)
  print("Label: ", label.numpy())
  i=i+1
'''


#show_batch(image_batch.numpy(), label_batch.numpy())
'''
import time
default_timeit_steps = 1000

def timeit(ds, steps=default_timeit_steps):
  start = time.time()
  it = iter(ds)
  for i in range(steps):
    batch = next(it)
    if i%10 == 0:
      print('.',end='')
  print()
  end = time.time()

  duration = end-start
  print("{} batches: {} s".format(steps, duration))
  print("{:0.5f} Images/s".format(BATCH_SIZE*steps/duration))

timeit(train_data_gen)

timeit(train_ds)
'''

'''
今回はMobileNetV2というCNNを入力層に使用する。
include_top...このネットワークの出力層側にある全結合層を含むかどうかの設定
'''
mobile_net=tf.keras.applications.MobileNetV2(input_shape=(192,192,3),include_top=False)
mobile_net.trainable=False

def change_range(image,label):
    return 2*image-1, label

keras_ds = train_ds.map(change_range)

'''
mobile_net層でどのような変換がなされるのか確認
ex)
(32, 6, 6, 1280)
(32, 192, 192, 3)
[2 2 3 3 4 1 0 4 2 1 1 4 4 1 1 2 3 1 1 2 1 1 1 1 1 4 0 0 2 2 4 4]

image_batch, label_batch = next(iter(keras_ds))
feature_map_batch = mobile_net(image_batch)
print(feature_map_batch.shape)
print(image_batch.shape)
print(label_batch.numpy())
'''

'''
GlobalAveragePooling2Dってなんぞや
mobile_netによって得られたベクトル表現を圧縮する。
(どこかの二つの次元に対して)平均値を割り出し圧縮する。つまり２次元分圧縮される
'''
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

test_dir=pathlib.Path('/Users/uemiya/.keras/datasets/test')
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

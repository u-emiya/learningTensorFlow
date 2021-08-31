from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

AUTOTUNE=tf.data.experimental.AUTOTUNE

'''
get_file...origin=url,fname=ファイルの名前,untar=取得するファイルを解凍するかどうか
data_root_origにダウンロードしたデータのパスが保存される。
'''
import pathlib
data_root_orig=tf.keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
                                       fname='flower_photos', untar=True)
data_root=pathlib.Path(data_root_orig)
print(data_root)

for item in data_root.iterdir():
    print(item)

import random
all_image_paths=list(data_root.glob('*/*'))
all_image_paths=[str(path) for path in all_image_paths]
random.shuffle(all_image_paths)

image_count=len(all_image_paths)
print(image_count)

import IPython.display as display
from PIL import Image
roses=list(data_root.glob('roses/*'))
for image_path in roses[:3]:
    im=(Image.open(str(image_path)))
    #im.show()

label_names=sorted(item.name for item in data_root.glob('*/') if item.is_dir())
print(label_names)
label_to_index=dict((name,index) for index,name in enumerate(label_names))
print(label_to_index)

all_image_labels= [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]
print("First 10 labels indices: ", all_image_labels[:10])

img_path=all_image_paths[0]
img_raw=tf.io.read_file(img_path)
print(repr(img_raw)[:100]+"...")
'''
画像のテンソルにデコードをする
'''
img_tensor=tf.image.decode_image(img_raw)
print(img_tensor.shape)
print(img_tensor.dtype)
'''
全ての画像を192*192に変形し、その後正規化する
'''
img_final=tf.image.resize(img_tensor,[192,192])
img_final=img_final/255.0
print(img_final.shape)
print(img_final.numpy().min())
print(img_final.numpy().max())


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image /= 255.0  # normalize to [0,1] range

    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

import os
attributions = (data_root/"LICENSE.txt").open(encoding='utf-8').readlines()[4:]
attributions = [line.split(' CC-BY') for line in attributions]
attributions = dict(attributions)

def caption_image(image_path):
  image_rel = pathlib.Path(image_path).relative_to(data_root)
  return "Image (CC BY 2.0) " + ' - '.join(attributions[str(image_rel)].split(' - ')[:-1])


import matplotlib.pyplot as plt

image_path = all_image_paths[0]
label = all_image_labels[0]

plt.imshow(load_and_preprocess_image(img_path))
plt.grid(False)
plt.xlabel(caption_image(img_path))
plt.title(label_names[label].title())
print()

path_ds=tf.data.Dataset.from_tensor_slices(all_image_paths)
image_ds=path_ds.map(load_and_preprocess_image,num_parallel_calls=AUTOTUNE)

import matplotlib.pyplot as plt
'''
plt.figure(figsize=(8,8))
for n,image in enumerate(image_ds.take(4)):
  plt.subplot(2,2,n+1)
  plt.imshow(image)
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.xlabel(caption_image(all_image_paths[n]))
  plt.show()
'''
label_ds=tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels,tf.int64))
for label in label_ds.take(10):
     print(label_names[label.numpy()])

image_label_ds=tf.data.Dataset.zip((image_ds,label_ds))

ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))

# The tuples are unpacked into the positional arguments of the mapped function
# タプルは展開され、マップ関数の位置引数に割り当てられます
def load_and_preprocess_from_path_label(path, label):
  return load_and_preprocess_image(path), label

image_label_ds = ds.map(load_and_preprocess_from_path_label)
print(image_label_ds)

BATCH_SIZE = 32

# シャッフルバッファのサイズをデータセットとおなじに設定することで、データが完全にシャッフルされる
# ようにできます。
ds = image_label_ds.shuffle(buffer_size=image_count)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
# `prefetch`を使うことで、モデルの訓練中にバックグラウンドでデータセットがバッチを取得できます。
ds = ds.prefetch(buffer_size=AUTOTUNE)

mobile_net=tf.keras.applications.MobileNetV2(input_shape=(192,192,3),include_top=False)
mobile_net.trainable=False

def change_range(image,label):
    return 2*image-1, label

keras_ds = ds.map(change_range)
image_batch, label_batch = next(iter(keras_ds))
feature_map_batch = mobile_net(image_batch)
print(feature_map_batch.shape)
'''
GlobalAveragePooling2Dってなんぞや
mobile_netによって得られたベクトル表現を圧縮する。
(どこかの二つの次元に対して)平均値を割り出し圧縮する。つまり２次元分圧縮される
'''
model = tf.keras.Sequential([
    mobile_net,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(len(label_names))])
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])

STEPS_PER_EPOC=tf.math.ceil(len(all_image_paths)/BATCH_SIZE).numpy()

image_batch, label_batch = next(iter(keras_ds))
feature_map_batch = mobile_net(image_batch)

print(feature_map_batch.shape)
print(image_batch.shape)
print(label_batch)

#model.fit(keras_ds, epochs=1,  steps_per_epoch=STEPS_PER_EPOC)

from tensorflow.keras.models import load_model
import numpy as np
from keras.preprocessing.image import img_to_array,load_img
import  tensorflow as tf

img_name='test_thomas1'
model_file_name='thomas_edward'


model=load_model('/Users/uemiya/Documents/thomas_edward.h5')
img_path=('/Users/uemiya/.keras/datasets/train_animation/prediction/test_thomas2.jpg')

img=img_to_array(load_img(img_path,target_size=(192,192)))
img_nad=img_to_array(img)/255
img_nad=img_nad[None,...]

label=['edward','thomas']
pred=model.predict(img_nad,batch_size=1,verbose=0)
score=np.max(pred)
i=np.argmax(pred)
print(label)
pred_label=label[i]
print('name:',pred_label)
print('score:',score)

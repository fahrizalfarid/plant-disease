## Optimized Model for arm architecture

## Requirements
`pip install tensorflow==2.6.0`

## Run
```python
from tensorflow.keras.preprocessing.image import img_to_array,load_img
import tensorflow as tf
import numpy as np
from json import load
import os


imgsize = 96
modelpath = './model/plant_disease_96_at_val_accuracy_0.80_epoch_3_tf260.tflite'
interpreter = tf.lite.Interpreter(model_path=modelpath)
interpreter.allocate_tensors()


input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)


labels = load(open('class.json','r'))


def preproces(imgfile):
    img = load_img(
        imgfile, target_size=(imgsize,imgsize), color_mode='rgb',
    )
    img_array = img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array



path = './test/'
imglist = os.listdir(path)

for img in imglist:
    img_array = preproces(path+img)
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(img, labels[str(np.argmax(output_data[0]))])
```

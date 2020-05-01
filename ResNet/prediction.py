import numpy as np
import cv2
import sys
import json
from keras.models import model_from_json
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import os
import glob
import math
import numpy as np
import imageio

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#model
model_path = './models'
json_file=open(model_path+'/mobileNet_json.json')
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)
#load weight
model.load_weights(model_path+'/mobileNet_checkpoints.h5')

#model1
json_file1=open(model_path+'/mobileNet1_json.json')
loaded_model_json1 = json_file1.read()
json_file1.close()

model1 = model_from_json(loaded_model_json1)
#load weight
model1.load_weights(model_path+'/mobileNet1_checkpoints.h5')

#model2
json_file2=open(model_path+'/mobileNet2_json.json')
loaded_model_json2 = json_file2.read()
json_file2.close()

model2 = model_from_json(loaded_model_json2)
#load weight
model2.load_weights(model_path+'/mobileNet2_checkpoints.h5')

def load_image(folder, mode):
    if (mode == 'train'):
        image_files = os.listdir(folder)
    elif (mode == 'test'):
        image_files = [str(x) + '.png' for x in range(10000)]
    dataset = np.ndarray(shape=(len(image_files),28,28,3), dtype=np.float32)
    print(folder)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        image_data = imageio.imread(image_file)
        image_data = cv2.resize(image_data,(28,28)).astype(np.float32)
        image_data = image_data/255.0
        image_data = (image_data-0.1307)/0.3081
        image_data = (np.repeat(image_data[..., np.newaxis], 3, -1))
        dataset[num_images, :, :] = image_data
        num_images += 1
            
    dataset = dataset[0:num_images, :, :]

    print('Full dataset tensor:', dataset.shape)

    return dataset

test_folder = './hw4_test'
test_data = load_image(test_folder,'test')

#test_result = model.predict(test_data)
#test_output = [np.argmax(x) for x in test_result]
test_datagen = ImageDataGenerator(data_format='channels_last')
test_data0 = test_datagen.flow(x=test_data, y=None, batch_size=128, shuffle=False)
result = model.predict_generator(test_data0, steps=10000/128,verbose=1)
result = [np.argmax(x) for x in result]

test_data1 = test_datagen.flow(x=test_data, y=None, batch_size=128, shuffle=False)
result1 = model1.predict_generator(test_data1, steps=10000/128,verbose=1)
result1 = [np.argmax(x) for x in result1]

test_data2 = test_datagen.flow(x=test_data, y=None, batch_size=128, shuffle=False)
result2 = model2.predict_generator(test_data2, steps=10000/128,verbose=1)
result2 = [np.argmax(x) for x in result2]

#dummy 3NN
final_result = []
for i in range(10000):
    if(result[i]==result1[i] and result1[i]==result2[i]):
        final_result.append(result[i])
    else:
        if(result[i]==result1[i]):
            final_result.append(result[i])
        elif(result[i]==result2[i]):
            final_result.append(result[i])
        elif(result1[i]==result2[i]):
            final_result.append(result1[i])
        else:
            print(i)
            final_result.append(result[i])

with open("prediction.txt", 'w+') as f:
    for x in final_result:
        f.write(str(x) + '\n')

f.close()

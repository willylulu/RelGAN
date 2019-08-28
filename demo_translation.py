import os
import sys
import random
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from PIL import Image
from keras import backend as K
from skimage import io, transform
from keras.models import load_model
from keras.backend.tensorflow_backend import set_session
from contrib.ops import SwitchNormalization
from module import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--device", type=str, default='0')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

def tileAttr(x):
    x = tf.expand_dims(x, axis = 1)
    x = tf.expand_dims(x, axis = 2)
    return tf.tile(x, [1, 256, 256, 1])

def tileAttr2(x):
        x = tf.expand_dims(x, axis = 1)
        x = tf.expand_dims(x, axis = 2)
        return tf.tile(x, [1, 4, 4, 1])
    
# train_path = '/share/diskB/willy/GanExample/FaceAttributeChange_StarGAN/model/generator499.h5'

def testPic(img, gender, bangs=-1, glasses=1):
    temp3 = io.imread(img)
    tempb = transform.resize(temp3, [256,256])
    tempb = tempb[:,:,:3]
    tempb = tempb*2 - 1
    
    # imgIndex = np.load("imgIndex.npy")
# imgAttr = np.load("anno_dic.npy").item()

    new_attrs = ['5_o_Clock_Shadow', 'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Male', 'Mustache', 'Pale_Skin', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Hat', 'Young']

    def att2vec(attt):
        temp_vec = np.expand_dims(attt, axis=0)
        return temp_vec

    attrs_pos, attrs_neg = [],[]
    attr = []
    
    temp = np.zeros([17])
    # temp[new_attrs.index('Brown_Hair')] = -1
    temp[new_attrs.index('Black_Hair')] = -1
    temp[new_attrs.index('Blond_Hair')] = 1
    attr.append(temp)

    temp = np.zeros([17])
    temp[new_attrs.index('Black_Hair')] = -1
    # temp[new_attrs.index('Brown_Hair')] = -1
    # temp[new_attrs.index('Black_Hair')] = 1
    temp[new_attrs.index('Brown_Hair')] = 1
    attr.append(temp)

    temp = np.zeros([17])
    if gender==1:
        temp[new_attrs.index('Male')] = -1
    else:
        temp[new_attrs.index('Male')] = 1
    attr.append(temp)

    temp = np.zeros([17])
    #if gender==0:
    temp[new_attrs.index('Male')] = 1 
    temp[new_attrs.index('Goatee')] = 1
    temp[new_attrs.index('Mustache')] = 1
    temp[new_attrs.index('5_o_Clock_Shadow')] = 1
    attr.append(temp)

    temp = np.zeros([17])
    temp[new_attrs.index('Pale_Skin')] = 1
    attr.append(temp)

    temp = np.zeros([17])
    temp[new_attrs.index('Smiling')] = 1
    attr.append(temp)

    temp = np.zeros([17])
    temp[new_attrs.index('Bangs')] = bangs
    attr.append(temp)

    temp = np.zeros([17])
    temp[new_attrs.index('Eyeglasses')] = glasses
    attr.append(temp)

    temp = np.zeros([17])
    temp[new_attrs.index('Gray_Hair')] = 1
    temp[new_attrs.index('Young')] = -1
    attr.append(temp)
    
    for at in attr:
        att_pos = att2vec(at)
        attrs_pos.append(att_pos)

    attrs_pos = np.concatenate(attrs_pos, axis=0)    
    
    output = np.expand_dims(tempb, axis=0)
    output2 = np.tile(output, [len(attr), 1, 1, 1])
    outputs_, _ = relGan.predict([output2, attrs_pos])
    images = np.concatenate([output, outputs_], axis = 0)
    width = 1
    height = len(attr) + 1
    new_im = Image.new('RGB', (256*height, 256*width))
    for ii in range(height):
        for jj in range(width):
            index=ii*width+jj
            image = (images[index]/2+0.5)*255
            image = image.astype(np.uint8)
            new_im.paste(Image.fromarray(image,"RGB"), (256*ii,256*jj))
    ans = np.array(new_im)
    return ans

# temp3 = io.imread('/share/data/celeba-hq/celeba-256/12345.jpg')
# version = int(sys.argv[2])
# if version==-1:
#     version = len(os.listdir('img'))-2
version = 519

print(version)

train_path = './generator'+str(version)+'.h5'
#train_path = 'model/generator1511.h5'

# train_path = 'good_v0513.h5'

def orthogonal(w):
    
    w_kw = K.int_shape(w)[0]
    w_kh = K.int_shape(w)[1]
    w_w = K.int_shape(w)[2]
    w_h = K.int_shape(w)[3]
    
    temp = 0
    for i in range(w_kw):
        for j in range(w_kh):
            wwt = tf.matmul(tf.transpose(w[i,j]), w[i,j])
            mi = K.ones_like(wwt) - K.identity(wwt)
            a = wwt * mi
            a = tf.matmul(tf.transpose(a), a)
            a = a * K.identity(a)
            temp += K.sum(a)
    return 1e-4 * temp

img_shape = (256, 256, 3)
vec_shape = (17,)

imgA_input = Input(shape=img_shape)
imgB_input = Input(shape=img_shape)
vec_input_pos = Input(shape=vec_shape)
vec_input_neg = Input(shape=vec_shape)

g_out = generator(imgA_input, vec_input_pos, 256)
relGan = Model(inputs=[imgA_input, vec_input_pos], outputs=g_out)
relGan.load_weights(train_path)
relGan.summary()

lengh = 10
temp = [None]*lengh
temp[0] = testPic('test_img/j.png',0)
temp[1] = testPic('test_img/c.2.jpg',0)
temp[2] = testPic('test_img/es.png',1)
temp[3] = testPic('test_img/e.2.png',1)
temp[4] = testPic('test_img/g.2.png',1)
temp[5] = testPic('test_img/y3.png',1)
temp[6] = testPic('test_img/f1.png',1,glasses=-1)
temp[7] = testPic('test_img/j1.png',0,glasses=-1)
temp[8] = testPic('test_img/c3.png',0)
temp[9] = testPic('test_img/g3.png',1,glasses=-1)

new_im = Image.new('RGB', (256*10, 256*lengh))
for jj in tqdm(range(lengh)):
    index = jj
    image = temp[index]
    new_im.paste(Image.fromarray(image,"RGB"), (0,256*jj))
new_im.save('test_v%04d.jpg'%version)

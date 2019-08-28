# Copyright (C) 2019 Willy Po-Wei Wu & Elvis Yu-Jing Lin <maya6282@gmail.com, elvisyjlin@gmail.com>
# 
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import sys
import random
import imageio
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
parser.add_argument("-f", "--file", type=str, default='test_img/y3.png')
parser.add_argument("-o", "--output", type=str, default='output.gif')

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
 
    for i in range(0,21):
        temp = np.zeros([17])
        
        temp[new_attrs.index('Smiling')] = 0.05 * i
        #temp[new_attrs.index('Gray_Hair')] = 0.07* i
        #temp[new_attrs.index('Young')] = -0.07* i
        attr.append(temp)
        
    for i in range(21,0,-1):
        temp = np.zeros([17])
        
        temp[new_attrs.index('Smiling')] = 0.05 * i
        #temp[new_attrs.index('Gray_Hair')] = 0.07* i
        #temp[new_attrs.index('Young')] = -0.07* i
        attr.append(temp)
        
    for at in attr:
        att_pos = att2vec(at)
        attrs_pos.append(att_pos)

    attrs_pos = np.concatenate(attrs_pos, axis=0)    
    
    output = np.expand_dims(tempb, axis=0)
    output2 = np.tile(output, [len(attr), 1, 1, 1])
    outputs_, _ = relGan.predict([output2, attrs_pos])
    outputs_ = np.ndarray.astype((outputs_/2+0.5)*255, np.uint8)
    imageio.mimsave('simple_'+args.output, outputs_[10:31])
    imageio.mimsave(args.output, outputs_)

# temp3 = io.imread('/share/data/celeba-hq/celeba-256/12345.jpg')
version = 519
#version = 461

train_path = './generator'+str(version)+'.h5'

print('version: ', version)
#train_path = 'model/generator1511.h5'

# train_path = 'good_v0513.h5'

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

lengh = 1
temp = [None]*lengh
temp[0] = testPic(args.file, 1, glasses=-1)

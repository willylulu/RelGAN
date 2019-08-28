import tensorflow as tf
import keras
from keras.layers import LeakyReLU, Conv2D, Add, ZeroPadding2D, Activation, Lambda, Dropout
from contrib.ops import SwitchNormalization
from keras import backend as K

def hard_tanh(x):
    return K.clip(x, -1, )

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
    return 2e-6 * temp

def residual_block(x, dim, ks, init_weight, name):
    y = Conv2D(dim, ks, strides=1, padding="same", kernel_initializer=init_weight, kernel_regularizer = orthogonal)(x)
    y = SwitchNormalization(axis=-1, name=name+'_0')(y)
    y = Activation('relu')(y)   
    y = Conv2D(dim, ks, strides=1, padding="same", kernel_initializer=init_weight, kernel_regularizer = orthogonal)(y)
    y = SwitchNormalization(axis=-1, name=name+'_1')(y)
    return Add()([x,y])
    
def glu(x):
    channel = K.int_shape(x)[-1]
    channel = channel//2
    a = x[..., :channel]
    b = x[..., channel:]
    return a * K.sigmoid(b)
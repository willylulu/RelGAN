import os
import random
import time
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.layers import Input
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras.utils import plot_model
from keras.backend.tensorflow_backend import set_session
from module import *
from ops import *
from skimage import io, transform
from tensorboardX import SummaryWriter
from keras.preprocessing.image import ImageDataGenerator

class Relgan():
    
    def __init__(self, args):
        
        self.path = args.path
        self.lr = args.lr
        self.b1 = args.beta1
        self.b2 = args.beta2
        self.batch = args.batch_size
        self.sample = args.sample_size
        self.epochs = args.epochs
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.lambda4 = args.lambda4
        self.lambda5 = args.lambda5
        self.gp_l = args.lambda_gp
        self.decay = self.lr/self.epochs
        self.imgSize = args.img_size
        self.sampleSize = args.img_size
        self.vecSize = args.vec_size
        self.step = args.step*200
        
        self.lr -= self.decay * self.step
        
        self.img_shape = (self.imgSize, self.imgSize, 3)
        self.vec_shape = (self.vecSize,)
        
        self.get_model()
        self.get_loss()
        self.get_optimizer()
        self.datagen = ImageDataGenerator(horizontal_flip=True)
        self.writer = SummaryWriter()
    def get_model(self):
        
        self.imgA_input = Input(shape=self.img_shape)
        self.imgB_input = Input(shape=self.img_shape)
        self.vec_input_pos = Input(shape=self.vec_shape)
        self.vec_input_neg = Input(shape=self.vec_shape)
            
        g_out = generator(self.imgA_input, self.vec_input_pos, self.imgSize)

        self.g_model = Model(inputs=[self.imgA_input, self.vec_input_pos], outputs=g_out)

        d_out = discriminator(self.imgA_input, self.imgB_input, self.vec_input_pos, self.imgSize, self.vecSize)

        self.d_model = Model(inputs=[self.imgA_input, self.imgB_input, self.vec_input_pos], \
                             outputs=d_out)
        
        print(self.g_model.summary())
        print(self.d_model.summary())
        
        plot_model(self.g_model, to_file='g_model.png')
        plot_model(self.d_model, to_file='d_model.png')
        
    def get_loss(self):
        
        def cal_df_gp():
            
            def cal_gp(gradients):
                
                gradients_sqr = K.square(gradients[0])
                gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
                gradient_l2_norm = K.sqrt(gradients_sqr_sum)
                gradient_penalty = K.mean(K.square(1 - gradient_l2_norm))
                return gradient_penalty
            
            alpha = K.random_uniform_variable(shape=(1,), low=0, high=1)
            
            mix_tar = alpha * self.img_a + (1 - alpha) * self.img_a2b
            
            mix_outputs_a2b = self.d_model([self.img_a, mix_tar, self.vec_ab_pos])
            mix_outputs_a2ab = self.d_model([self.img_a, self.img_a2ab, self.vec_ab_pos])
            
            
            gradients_a2b = K.gradients([mix_outputs_a2b[0]], [mix_tar])
            gradients_a2ab = K.gradients([mix_outputs_a2ab[2]], [self.img_a2ab])
            
            
            df_gp = cal_gp(gradients_a2b) + cal_gp(gradients_a2ab)
                
            return df_gp
        
        def lsgan(xs, ts):
            real = 0
            fake = 0
            for i in range(len(xs)):
                if ts[i]==1:
                    real += K.mean(K.square(K.ones_like(xs[i]) - xs[i]), axis=[-1])
                else:
                    fake += K.mean(K.square(K.zeros_like(xs[i]) - xs[i]), axis=[-1])
                    
            return real + fake
        
        self.img_a = Input(shape=self.img_shape)
        self.img_b = Input(shape=self.img_shape)
        self.img_c = Input(shape=self.img_shape)

        self.vec_ab_pos = Input(shape=self.vec_shape)
        self.vec_ac_pos = Input(shape=self.vec_shape)
        self.vec_cb_pos = Input(shape=self.vec_shape)
        
        self.img_a2b, self.enc_a2b = self.g_model([self.img_a, self.vec_ab_pos])
        self.img_a2a, self.enc_a2a = self.g_model([self.img_a, K.zeros_like(self.vec_ab_pos)])
        self.img_a2b2a, _ = self.g_model([self.img_a2b, -self.vec_ab_pos])
        
        inter_seed = K.random_uniform_variable(shape=([self.batch,]), low=0, high=1)
        inter_seed = K.reshape(inter_seed, [self.batch,1])
        self.img_a2ab, self.enc_a2ab = self.g_model([self.img_a, inter_seed * self.vec_ab_pos])
        
        input_real = [self.img_a, self.img_b, self.vec_ab_pos]
        input_fake = [self.img_a, self.img_a2b, self.vec_ab_pos]
        input_w_ori = [self.img_c, self.img_b, self.vec_ab_pos]
        input_w_tar = [self.img_a, self.img_c, self.vec_ab_pos]
        input_w_vec1 = [self.img_a, self.img_b, self.vec_ac_pos]
        input_w_vec2 = [self.img_a, self.img_b, self.vec_cb_pos]
        
        input_inter = [self.img_a, self.img_a2ab, inter_seed * self.vec_ab_pos]
        input_zero = [self.img_a, self.img_a2a, K.zeros_like(self.vec_ab_pos)]
        
        d_real, dc_real, _ = self.d_model(input_real)
        
        d_fake, dc_fake, di_fake = self.d_model(input_fake)
        
        d_w_ori, dc_w_ori, _ = self.d_model(input_w_ori)
        d_w_tar, dc_w_tar, _ = self.d_model(input_w_tar)
        d_w_vec1, dc_w_vec1, _ = self.d_model(input_w_vec1)
        d_w_vec2, dc_w_vec2, _ = self.d_model(input_w_vec2)
        
        _, _, di_inter = self.d_model(input_inter)
        _, _, di_zero = self.d_model(input_zero)
    
        self.df_loss = lsgan([d_real, d_fake], [1, 0])
        self.dc_loss = lsgan([dc_real, dc_fake, dc_w_ori, dc_w_tar, dc_w_vec1, dc_w_vec2], [1, 0, 0, 0, 0, 0])
        inter_seed_rep = K.flatten(inter_seed)
        
        di_temp = K.switch(K.less(inter_seed_rep, 0.5 * K.ones_like(inter_seed_rep)), di_zero, di_fake)
        
        self.di_loss = K.square(K.minimum(inter_seed_rep, K.ones_like(inter_seed_rep) - inter_seed_rep) * K.ones_like(di_inter) - di_inter) + K.square(di_temp)
        print('self.df_loss', K.int_shape(self.df_loss))
        print('self.dc_loss', K.int_shape(self.dc_loss))
        print('self.di_loss', K.int_shape(self.di_loss))
        
        
        self.df_gp = cal_df_gp()
        
        self.d_loss = self.df_loss + self.dc_loss + self.gp_l * self.df_gp + self.lambda5 * self.di_loss
        
        self.gf_loss = lsgan([d_real, d_fake], [0, 1])
        self.gc_loss = lsgan([dc_real, dc_fake], [0, 1])
        self.gi_loss = K.square(di_inter)
        
        dist_a2b = self.enc_a2b - self.enc_a2a
        dist_a2ab = self.enc_a2ab - self.enc_a2a
        
        inter_seed = K.reshape(inter_seed, [self.batch,1,1,1])
        self.g_inter_loss = K.mean(K.abs(inter_seed * dist_a2b - dist_a2ab))
        
        g_loss_rec1 = K.mean(K.abs(self.img_a - self.img_a2b2a))
        g_loss_rec2 = K.mean(K.abs(self.img_a - self.img_a2a))
        
        print('self.gf_loss', K.int_shape(self.gf_loss))
        print('self.gc_loss', K.int_shape(self.gc_loss))
        print('self.gi_loss', K.int_shape(self.gi_loss))
        print('self.g_loss_rec1', K.int_shape(g_loss_rec1))
        print('self.g_loss_rec2', K.int_shape(g_loss_rec2))
        
        self.gr_loss = self.lambda1 * g_loss_rec1 + self.lambda2 * g_loss_rec2
        self.g_loss = self.gf_loss + self.gc_loss + self.gr_loss + self.lambda5 * self.gi_loss
        
    def get_optimizer(self):
        
        g_opt = Adam(lr=self.lr, decay = self.decay, beta_1=self.b1, beta_2=self.b2)
        g_weights = self.g_model.trainable_weights
        g_inputs = [self.img_a, self.img_b, self.vec_ab_pos]
        
        self.g_training_updates = g_opt.get_updates(g_weights, [], self.g_loss)
        self.g_train = K.function(g_inputs, 
                                  [K.mean(self.g_loss), 
                                   K.mean(self.gf_loss), 
                                   K.mean(self.gc_loss), 
                                   K.mean(self.gr_loss), 
                                   K.mean(self.g_inter_loss),
                                   K.mean(self.gi_loss)], 
                                  self.g_training_updates)
        
        d_opt = Adam(lr=self.lr, decay = self.decay, beta_1=self.b1, beta_2=self.b2)
        d_weights = self.d_model.trainable_weights
        d_inputs = [self.img_a, self.img_b, self.img_c, self.vec_ab_pos, self.vec_ac_pos, self.vec_cb_pos]
        
        self.d_training_updates = d_opt.get_updates(d_weights, [], self.d_loss)
        self.d_train = K.function(d_inputs, 
                                  [K.mean(self.d_loss), 
                                   K.mean(self.df_loss), 
                                   K.mean(self.dc_loss), 
                                   K.mean(self.gp_l * self.df_gp), 
                                   K.mean(self.di_loss)], 
                                  self.d_training_updates)
        
    def get_imgs_tags(self, indexserX, imgIndex, imgAttr):
        imgs = [None]*self.batch
        atts = [None]*self.batch
        
        for i in range(self.batch):
            temp_index = indexserX[i]
            img_fa = imgIndex[temp_index]

            while img_fa == None:
                temp_index = np.random.choice(len(imgIndex), 1)[0]
                img_fa = imgIndex[temp_index]
            atts[i] = imgAttr[img_fa]
            
            img = io.imread(os.path.join(self.path, str(temp_index)+".jpg"))
            imgs[i] = img/127.5-1
            
        imgs = np.array(imgs)
        atts = np.array(atts)
        
        self.datagen.fit(imgs)
        
        imgs = self.datagen.flow(imgs, batch_size=self.batch, shuffle=False).next()
        
        return imgs, atts
                          
    def train(self):
        
        print("load index")
        imgIndex = np.load("imgIndex.npy")
        imgAttr = np.load("anno_dic.npy").item()
        print("training")
        
        ite = self.step
        
        def getIndex():
            while True:
                count = 0
                index_permutation = np.random.permutation(len(imgIndex))
                while count + self.batch*3 < len(imgIndex):
                    yield index_permutation[count:(count+self.batch*3)]
                    count = count + self.batch*3
        
        index_gen = getIndex()
        
        def get_training_data(wrong=False):
            indexser = next(index_gen)
            indexser1 = indexser[self.batch*0:self.batch*1] 
            indexser2 = indexser[self.batch*1:self.batch*2]
            indexser3 = indexser[self.batch*2:self.batch*3] 

            img_as, att_as = self.get_imgs_tags(indexser1, imgIndex, imgAttr)
            img_bs, att_bs = self.get_imgs_tags(indexser2, imgIndex, imgAttr)
            vec_ab_pos = att_bs - att_as
            
            if wrong==False:
                return img_as, img_bs, vec_ab_pos
            
            img_cs, att_cs = self.get_imgs_tags(indexser3, imgIndex, imgAttr)

            vec_ac_pos = att_cs - att_as
            vec_cb_pos = att_bs - att_cs
            
            return img_as, img_bs, img_cs, vec_ab_pos, vec_ac_pos, vec_cb_pos
        
        for ep in range(int(self.epochs)):
            
            t_start = time.time()
            
            img_as, img_bs, img_cs, vec_ab_pos, vec_ac_pos, vec_cb_pos = get_training_data(wrong=True)
            
            for i in range(1):
                errD = self.d_train([img_as, img_bs, img_cs, vec_ab_pos, vec_ac_pos, vec_cb_pos])
                
            for i in range(1):
                errG = self.g_train([img_as, img_bs, vec_ab_pos])
            
            t_end = time.time()
            
            print("%9.6f %9.6f | real: %7.4f wrong: %7.4f gp: %7.4f| fake: %7.4f wrong: %7.4f recs: %7.4f enc: %7.4f| time: %.4f"%(errD[0], errG[0],errD[1], errD[2], errD[3], errG[1], errG[2], errG[3], errG[4], t_end - t_start)) 
            
            self.writer.add_scalar('d_loss', errD[0], ite)
            self.writer.add_scalar('g_loss', errG[0], ite)
            self.writer.add_scalar('df_loss', errD[1], ite)
            self.writer.add_scalar('gf_loss', errG[1], ite)
            self.writer.add_scalar('dc_loss', errD[2], ite)
            self.writer.add_scalar('gc_loss', errG[2], ite)
            self.writer.add_scalar('gr_loss', errG[3], ite)
            self.writer.add_scalar('inter_loss', errG[4], ite)
            self.writer.add_scalar('gp_loss', errD[3], ite)
            self.writer.add_scalar('gi_loss', errG[5], ite)
            self.writer.add_scalar('di_loss', errD[4], ite)
            
            if ite%50==0 and ite>0:
                
                img_as, img_bs, vec_ab_pos = get_training_data(wrong=False)
                
                g_a2b = [img_as[:self.sample], vec_ab_pos[:self.sample]]
                fakea2b,_ = self.g_model.predict(g_a2b)
                
                g_a2a = [img_as[:self.sample], np.zeros([self.sample, self.vecSize])]
                fakea2a,_ = self.g_model.predict(g_a2a)
                
                g_a2b2a = [fakea2b[:self.sample], -vec_ab_pos[:self.sample]]
                fakea2b2a,_ = self.g_model.predict(g_a2b2a)
                
                images = np.concatenate([img_as[:self.sample], fakea2b, fakea2b2a, fakea2a], axis = 0)
                
                width = self.sample
                height = 4
                new_im = Image.new('RGB', (self.sampleSize*height, self.sampleSize*width))
                for ii in range(height):
                    for jj in range(width):
                        index=ii*width+jj
                        image = (images[index]/2+0.5)*255
                        image = transform.resize(image, (self.sampleSize, self.sampleSize), preserve_range = True)
#                         image = image*255
                        image = image.astype(np.uint8)
                        new_im.paste(Image.fromarray(image,"RGB"), (self.sampleSize*ii,self.sampleSize*jj))
                filename = "img/fakeFace%d.jpg"%(ite//200)
                new_im.save(filename)
                
                try:
                    self.g_model.save("model/generator%d.h5"%(ite//200))
                    self.d_model.save("model/discriminator.h5")
                except:
                    print('Pass save')
            ite = ite + 1




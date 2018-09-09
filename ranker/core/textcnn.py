import tensorflow as tf
import numpy as np


class TextCnn:

    def __init__(self, filter_num, filter_sizes, drop_rate, embed_dim, sen_len):
        
        self.filter_num = filter_num
        self.filter_sizes = filter_sizes
        self.drop_rate = drop_rate
        self.embed_dim = embed_dim
        self.sen_len = sen_len
        pass

    def __call__(self,input_embed,reuse=True):
        
        input_pool = []
        for idx,filter_size in enumerate(self.filter_sizes):
            with tf.name_scope('conv_'+str(idx)):
                input_convout = self.conv_pool_dense(input_embed,self.sen_len,filter_size,self.embed_dim, self.filter_num)
                input_pool.append(input_convout)

        input_flat = tf.concat(input_pool,axis=3, name='concater')
        input_reshape = tf.reshape(input_flat,[-1,self.filter_num*len(self.filter_sizes)])
        return input_reshape




    def conv_pool_dense(self,input_embed, sen_len, filter_size,embed_dim,filter_num):
        input_4d = tf.expand_dims(input_embed,axis=-1)
        W = tf.Variable(tf.truncated_normal([filter_size,embed_dim,1,filter_num], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[filter_num]), name="b")
        input_conv2d = tf.nn.conv2d(input_4d, W, [1,1,1,1], 'VALID', name='conv2d')
        input_conv2d_bias = tf.nn.relu(tf.nn.bias_add(input_conv2d, b), name="relu")
        input_pooling = tf.nn.max_pool(input_conv2d_bias, [1,sen_len-filter_size+1,1,1], [1,1,1,1], 'VALID', name='max_pooling')
        return input_pooling

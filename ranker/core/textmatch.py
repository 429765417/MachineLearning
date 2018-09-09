import tensorflow as tf
import numpy as np

class TextMatch:

    def __init__(self, filter_num, filter_sizes, drop_rate, embed_dim, sen_len):
        
        self.filter_num = filter_num
        self.filter_sizes = filter_sizes
        self.drop_rate = drop_rate
        self.embed_dim = embed_dim
        self.sen_len = sen_len
        self.loss = 0.0
        pass

    def __call__(self,input_embed1, input_embed2, reuse=True):
        self.loss = 0.0
        input_embed1 = tf.layers.dense(input_embed1, self.embed_dim, activation=tf.nn.relu)
        input_embed2 = tf.layers.dense(input_embed2, self.embed_dim, activation=tf.nn.relu )
        input_embed2_trans = tf.transpose(input_embed2, perm=[0,2,1])

        input_intersec1 = tf.matmul(input_embed1,input_embed2_trans)/self.embed_dim
        
        input_embed1_trans = tf.transpose(input_embed1, perm=[0,2,1])
        input_intersec2 = tf.matmul(input_embed2,input_embed1_trans)/self.embed_dim


        
        input_pool1 = []
        input_pool2 = []
        with tf.variable_scope("xxx") as var:
            for idx,filter_size in enumerate(self.filter_sizes):
                with tf.name_scope('conv_'+str(idx)):
                    input_convout1 = self.conv_pool_dense(input_intersec1,self.sen_len,filter_size,self.sen_len, self.filter_num)
                    input_pool1.append(input_convout1)
                    var.reuse_variables()
                    input_convout2 = self.conv_pool_dense(input_intersec2,self.sen_len,filter_size,self.sen_len, self.filter_num)
                    input_pool2.append(input_convout2)

        input_flat1 = tf.concat(input_pool1,axis=3, name='concater')
        input_reshape1 = tf.reshape(input_flat1,[-1,self.filter_num*len(self.filter_sizes)])
        input_flat2 = tf.concat(input_pool2,axis=3, name='concater')
        input_reshape2 = tf.reshape(input_flat2,[-1,self.filter_num*len(self.filter_sizes)])
        
        input_flat = tf.concat([input_reshape1,input_reshape2],axis=-1, name='concater')
        input_drop = tf.nn.dropout(input_flat, 0.7)
        return input_drop




    def conv_pool_dense(self,input_embed, sen_len, filter_size,embed_dim,filter_num):
        input_4d = tf.expand_dims(input_embed,axis=-1)
        W = tf.Variable(tf.truncated_normal([filter_size,embed_dim,1,filter_num], stddev=0.01), name="W")
        b = tf.Variable(tf.constant(0.01, shape=[filter_num]), name="b")
        input_conv2d = tf.nn.conv2d(input_4d, W, [1,1,1,1], 'VALID', name='conv2d')
        input_conv2d_bias = tf.nn.relu(tf.nn.bias_add(input_conv2d, b), name="relu")
        input_pooling = tf.nn.max_pool(input_conv2d_bias, [1,sen_len-filter_size+1,1,1], [1,1,1,1], 'VALID', name='max_pooling')
        return input_pooling

import tensorflow as tf
import numpy as np
from embedding import Embedder
from textcnn import TextCnn
from textmatch import TextMatch

class Net:
    
    def __init__(self, vocab_file, embed_dim,filter_num, filter_sizes, drop_rate,  sen_len):
        self.embedding_ = Embedder(vocab_file, embed_dim)
        self.textcnn_ = TextCnn(filter_num, filter_sizes, drop_rate, embed_dim, sen_len)
        self.textmatch_ = TextMatch(filter_num, filter_sizes, drop_rate, embed_dim, sen_len)
        self.sen_len = sen_len
        self.__call__()
        pass

    def __call__(self):
        
        self.input_x1 = tf.placeholder(dtype=tf.int32, shape=(None,self.sen_len), name="input_x1")
        self.input_x2 = tf.placeholder(dtype=tf.int32, shape = (None, self.sen_len), name="input_x2")
        self.input_y = tf.placeholder(dtype=tf.int32, name="input_y")

        'embedding lookup'
        input_embed1 = self.embedding_(self.input_x1)
        input_embed2 = self.embedding_(self.input_x2)
        
        input_out1= self.textcnn_(input_embed1)
        input_out2= self.textcnn_(input_embed2)

        input_intersec = self.textmatch_(input_embed1,input_embed2)
        
        'encoding jointly'
        input_sub = tf.subtract(input_out1, input_out2)
        input_mul = tf.multiply(input_out1, input_out2)
        input_encoding = tf.concat([input_out1,input_out2], axis=1, name='encoder')
        ''' 
        input_encoding = tf.reshape(input_encoding,[-1,100])
        input_relu = tf.layers.dense(input_encoding, 5, tf.nn.relu)
        input_relu = tf.reshape(input_relu,[-1,200])

        input_x = tf.layers.dense(input_relu, 100, tf.nn.relu)
        input_relu = tf.layers.Dropout(input_relu, 0.2)
        '''
        #input_x = tf.concat([input_encoding,input_intersec], axis=1, name='jointly')
        #input_x = tf.nn.dropout(input_x,0.8)
        input_x = input_intersec

        input_x = tf.layers.dense(input_x, 2, tf.nn.relu)

        losses = tf.nn.softmax_cross_entropy_with_logits(logits=input_x, labels=self.input_y, name='losses')
        self.loss = tf.reduce_sum(losses, axis=0)+tf.losses.get_regularization_loss()
        'accuracy'
        self.pred_indexs = tf.argmax(input_x,axis=1)
        label_indexs = tf.argmax(self.input_y, axis=1)
        self.accuracy=tf.reduce_mean(tf.cast(tf.equal(self.pred_indexs, label_indexs),dtype=tf.float32))
        








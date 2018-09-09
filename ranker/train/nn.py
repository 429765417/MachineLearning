#!--*--coding:utf8--*--

import tensorflow as tf
import numpy as np
import os
import datetime
from sklearn import metrics
import pdb
import sys
sys.path.append('../')
from setting import *
from core.net import Net
from prepare.data_loader import DataLoader

tf.app.flags.DEFINE_string('version', 'test',"model version")
tf.app.flags.DEFINE_integer('filter_num', 50,"filter num")
tf.app.flags.DEFINE_string('filter_sizes', '2,3',"filter sizes")
tf.app.flags.DEFINE_integer('total_step', 20000,"total_step")
tf.app.flags.DEFINE_integer('every_step', 2000,"every_step")
tf.app.flags.DEFINE_integer('batch_size', 64,"batch_size")
tf.app.flags.DEFINE_float('learning_rate', 5e-4,"learning_rate")
tf.app.flags.DEFINE_integer('embed_dim', 100,"embed_dim")
tf.app.flags.DEFINE_float('drop_rate', 0.1,"drop_rate")
tf.app.flags.DEFINE_integer('sen_len', 10,"sen_eln")

FLAGS = tf.app.flags.FLAGS

print "Params as follows:"
for k in FLAGS:
    print "{} = {}".format(k, eval('FLAGS.'+k))
    

class NN:
    def __init__(self):
        self.vocab_file = os.path.join(DATA_DIR,FLAGS.version,'word_vocab')
        self.neter = Net(self.vocab_file,FLAGS.embed_dim, FLAGS.filter_num, map(int,FLAGS.filter_sizes.split(',')), FLAGS.drop_rate, FLAGS.sen_len)
        self.data_loader_train = DataLoader(FLAGS.batch_size)
        self.data_loader_test = DataLoader(FLAGS.batch_size)
        self.train_path=os.path.join(DATA_DIR,FLAGS.version,"train_id")
        self.test_path=os.path.join(DATA_DIR,FLAGS.version,"validate_id")
        self.load_res()
        self.init_env()
        pass

    def load_res(self):
        self.data_loader_train.create_batches(self.train_path)
        self.data_loader_test.create_batches(self.test_path)
    
    def init_env(self):
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.neter.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        

    def train_step(self):
        feed_dict={}
        x1,x2,y = self.data_loader_train.get_rand_batch()
        if np.random.randint(2) == 0:
            feed_dict[self.neter.input_x1] = x1
            feed_dict[self.neter.input_x2] = x2
        else:
            feed_dict[self.neter.input_x1] = x2
            feed_dict[self.neter.input_x2] = x1

        feed_dict[self.neter.input_y]  = y
        _,step,loss,acc = self.sess.run([self.train_op,self.global_step,self.neter.loss, self.neter.accuracy], feed_dict)

        time_str = datetime.datetime.now().isoformat()
        print "{}: step {},loss {:g}, acc {:g}".format(time_str, step, loss, acc)

    def dev_step(self):
        argmax_indices=[]
        labels = []
        self.data_loader_test.reset()
        while not self.data_loader_test.is_end():
            x1,x2,y = self.data_loader_test.next_batch()
            feed_dict={}
            feed_dict[self.neter.input_x1] = x1
            feed_dict[self.neter.input_x2] = x2
            batch_argmax_indices = self.sess.run(self.neter.pred_indexs, feed_dict)
            argmax_indices.append(batch_argmax_indices)
            labels.append(np.argmax(y,axis=-1))

        pred_gl =   np.reshape(np.stack(argmax_indices,axis=1),(-1,1))
        label_gl =  np.reshape(np.stack(labels,axis=1),(-1,1))

        print(metrics.classification_report(label_gl, pred_gl))

    def run(self):
        for step_idx in range(FLAGS.total_step):
            self.train_step()
            if step_idx % FLAGS.every_step == 0:
                self.dev_step()



def main(argv):
    nn_obj = NN()
    nn_obj.run()

if __name__=='__main__':
    tf.app.run()

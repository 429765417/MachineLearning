import tensorflow as tf
import numpy as np
import os
import sys
sys.path.append("../")
from setting import *

import pdb

class Embedder:
    
    def __init__(self, vocab,embed_dim):
        self.pretrain_file = os.path.join(DATA_DIR,"pretrain_embed") 
        self.embeddings = self.load(vocab, embed_dim)

        pass

    def __call__(self, input_x):
        return tf.nn.embedding_lookup(self.embeddings, input_x, name="loced_embedding")

    def load(self, vocab_file, embed_dim):
        pretrain_dic={}
        for line in open(self.pretrain_file):
            items = line.split()
            if len(items) < 2:continue
            key = items[0]
            float_arr = list(map(float,items[1:]))
            pretrain_dic[key]=float_arr

        new_vocab = {}
        for line in open(vocab_file):
            items = line.strip().split('\t')
            if len(items) != 2:continue
            new_vocab[items[0]]=int(items[1])
        vocab_sorted = sorted(new_vocab.items(), key=lambda x:x[1])
        out_embedding=[]
        for word,id in vocab_sorted:
            if word == 'PAD':
                out_embedding.append(np.zeros(embed_dim))
            elif word == 'UNKNOWN':
                out_embedding.append(np.random.uniform(-1.0,1.0, embed_dim))
            elif word in pretrain_dic:
                out_embedding.append(pretrain_dic[word])
            else:
                out_embedding.append(np.random.uniform(-1.0,1.0, embed_dim))

        embed_tensor = tf.convert_to_tensor(np.array(out_embedding),dtype=tf.float32)
        embeddings = tf.Variable(embed_tensor,trainable=True, name='embeddings')
        return embeddings


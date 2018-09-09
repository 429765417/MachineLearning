#!--*--coding:utf8--*--

import  pandas as pd
import numpy as np
import jieba

import sys
import pdb
import os
import random

sys.path.append("../")
from setting import *

jieba.load_userdict(DATA_DIR+"user_dict")

class Preparor:
    
    def __init__(self, ver, file):
        self.version = ver
        self.raw_file = file
        self.cv = 0.1

    def __call__(self):
        tar_dir = os.path.join(DATA_DIR,self.version)
        os.system("mkdir -p "+tar_dir)
        seg_file = os.path.join(tar_dir,"corpus")
        corpus = pd.read_csv(self.raw_file, header=None, sep='\t')
        corpus_lst = []
        fw = open(seg_file, 'w')
        for index, row in corpus.iterrows():
            row1=row[1].replace("***","num")
            row2=row[2].replace("***","num")
            seg1 = " ".join(jieba.cut(row1.decode('utf8'))).encode('utf8')
            seg2 = " ".join(jieba.cut(row2.decode('utf8'))).encode('utf8')
            corpus_lst.append((row[0],seg1,seg2,row[3]))
            fw.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(row[0], seg1, seg2, row[3], row[1],row[2]))
        fw.close()
        
        random.shuffle(corpus_lst)

        train_file = os.path.join(tar_dir,"train_raw")
        seq_idx = int(self.cv*len(corpus_lst))
        fw = open(train_file, 'w')
        for id,seg1,seg2,label in corpus_lst[seq_idx:]:
            fw.write("{}\t{}\t{}\t{}\n".format(id, seg1, seg2, label))
        fw.close()

        validate_file = os.path.join(tar_dir,"validate_raw")
        fw = open(validate_file, 'w')
        for id,seg1,seg2,label in corpus_lst[:seq_idx]:
            fw.write("{}\t{}\t{}\t{}\n".format(id, seg1, seg2, label))
        fw.close()

        word_vocab = self.gen_vocab()
        self.format_files(word_vocab)

    def iding(self,tokens,word_vocab):
        ids = []
        for idx in range(SEN_LEN):
            if idx < len(tokens):
                if tokens[idx] in word_vocab:
                    ids.append(word_vocab[tokens[idx]])
                else:
                    ids.append(word_vocab['UNKNOWN'])

            else:
                    ids.append(word_vocab['PAD'])
        return " ".join(map(str,ids))

    '训练和测试的问句的id化'
    def format_files(self,word_vocab):
        train_file_path = os.path.join(DATA_DIR,self.version,"train_raw")
        test_file_path = os.path.join(DATA_DIR,self.version,"validate_raw")
        train_id_path = os.path.join(DATA_DIR,self.version,"train_id")
        test_id_path = os.path.join(DATA_DIR,self.version,"validate_id")

        fw = open(train_id_path,'w')
        for line in open(train_file_path):
            items = line.strip().split('\t')
            if len(items) < 4:continue
            sen1_tokens = items[1].split()
            sen2_tokens = items[2].split()
            if len(sen1_tokens)==0 or len(sen2_tokens)==0:
                continue
            id1 = self.iding(sen1_tokens,word_vocab)
            id2 = self.iding(sen2_tokens,word_vocab)
            fw.write("{}\t{}\t{}\t{}\n".format(items[0],id1,id2,items[3]))
        fw.close()

        fw = open(test_id_path,'w')
        for line in open(test_file_path):
            items = line.strip().split('\t')
            if len(items) < 4:continue
            sen1_tokens = items[1].split()
            sen2_tokens = items[2].split()
            if len(sen1_tokens)==0 or len(sen2_tokens)==0:
                continue
            id1 = self.iding(sen1_tokens,word_vocab)
            id2 = self.iding(sen2_tokens,word_vocab)
            fw.write("{}\t{}\t{}\t{}\n".format(items[0],id1,id2,items[3]))
        fw.close()


    def gen_vocab(self):
        train_file_path = os.path.join(DATA_DIR,self.version,"train_raw")
        word_set=set()
        for line in open(train_file_path,'r'):
            items = line.strip().split('\t')
            if len(items) < 3:continue 
            word_set |= set((items[1]+' '+items[2]).split(' '))

        id = 0
        word_vocab={}
        for k,v in DEF_VOCAB.items():
            word_vocab[k]=v
            id += 1

        for item in  word_set:
            if item == "":continue
            word_vocab[item]=id
            id += 1

        vocab_file_path = os.path.join(DATA_DIR, self.version, 'word_vocab')
        fw = open(vocab_file_path, 'w')
        vocab_sorted = sorted(word_vocab.items(), key=lambda x:x[1])
        for word,id in vocab_sorted:
            fw.write("{}\t{}\n".format(word,id))
        fw.close()
        return word_vocab


if __name__=="__main__":
    if len(sys.argv) != 3:
        print "params is invalid"
        exit(-1)
    
    obj = Preparor(sys.argv[1], sys.argv[2])
    obj()

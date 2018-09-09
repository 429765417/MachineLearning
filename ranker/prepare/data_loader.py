#!--*--coding:utf8--*--

import numpy as np
import random
import pdb

class DataLoader:
    def __init__(self,batch_size):
        self.samples=[]
        self.pos_samples=[]
        self.neg_samples=[]
        self.batch_size=batch_size
        self.idx = 0
        pass
    

    def create_batches(self, file_path):
        self.idx = 0
        for line in open(file_path):
                items = line.strip().split('\t')
                if len(items) <3:continue
                sen1= np.array(list(map(int,items[1].split())))
                sen2= np.array(list(map(int,items[2].split())))
                label = int(items[3])
                label_lst = np.zeros(2)
                label_lst[label]=1
                self.samples.append((sen1,sen2,np.array(label_lst)))
                if label == 0:
                    self.neg_samples.append((sen1,sen2,np.array(label_lst)))
                else:
                    self.pos_samples.append((sen1,sen2,np.array(label_lst)))


    def get_rand_batch(self):
        samples=[]
        batch_pos_samples = random.sample(self.pos_samples, self.batch_size/2)
        batch_neg_samples = random.sample(self.neg_samples, self.batch_size/2)
        samples += batch_pos_samples
        samples += batch_neg_samples
        batch_samples = np.array(samples)
        return np.stack(batch_samples[:,0],axis=0), np.stack(batch_samples[:,1],axis=0),np.stack( batch_samples[:,2],axis=0)
    
    def is_end(self):
        if self.idx+self.batch_size >= len(self.samples):
            return True
        return False
    def reset(self):
        self.idx = 0
    def next_batch(self):
        if self.idx+self.batch_size >= len(self.samples):
            return None
        batch_samples = np.array(self.samples[self.idx:self.idx+self.batch_size])
        self.idx += self.batch_size
        return np.stack(batch_samples[:,0],axis=0),np.stack(batch_samples[:,1],axis=0),np.stack(batch_samples[:,2],axis=0)






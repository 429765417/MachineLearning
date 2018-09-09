#!--*--coding:utf8--*--

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVR,SVC
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn import metrics
import argparse
import sys
sys.path.append("../")
from setting import *

import os
import pdb
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-version',default='test', type=str,help='model version')

FLAGS=parser.parse_args()

class Featurer:
    def __init__(self):
        self.raw_test=[]
        pass

    'load train test file'
    def load_corpus(self, file_name):
        cand_corpus = []
        cand_label = []
        cand_line=[]
        for line in open(file_name):
            items = line.strip().split('\t')
            if len(items) != 4:continue
            cand_corpus.append(items[1])
            cand_corpus.append(items[2])
            cand_label.append(int(items[3]))
            cand_line.append(line.strip())
        return cand_corpus,cand_label,cand_line

    def digitalizing(self, cand_corpus, cand_label, vectorizer=None, tfidf_transformer=None, pcaor=None, is_train=True):
        if is_train:
            vectorizer = CountVectorizer(min_df=3, ngram_range=(1,1))
            tfidf_transformer = TfidfTransformer()
            ids = vectorizer.fit_transform(cand_corpus)
            doc_tfidf = tfidf_transformer.fit_transform(ids)
            pcaor = TruncatedSVD(n_components=1000)
            doc_tfidf = pcaor.fit_transform(doc_tfidf)
        else:
            ids = vectorizer.transform(cand_corpus)
            doc_tfidf = tfidf_transformer.transform(ids)
            doc_tfidf = pcaor.transform(doc_tfidf)


        'organlize in pair'
        first = doc_tfidf[0::2,:]
        second = doc_tfidf[1::2,:]
        cosine_val = 0.5+0.5*(np.sum(first*second, axis=1)/(np.linalg.norm(first,axis=1)*np.linalg.norm(second,axis=1)+0.0000000001))
        mul_val = np.multiply(first,second)
        sub_val = np.abs(np.subtract(first,second))
        cosine_val = np.expand_dims(cosine_val,axis=-1)
        x = np.c_[first,second,sub_val,mul_val,cosine_val]
        
        return x,cand_label,vectorizer,tfidf_transformer,pcaor


    def __call__(self,train_file,test_file):
        cand_train_cs,cand_train_label,_ = self.load_corpus(train_file) 
        x_train,y_train,vecer,tfidformer,pca_obj = self.digitalizing(cand_train_cs, cand_train_label)
        cand_test_cs,cand_test_label,cand_test_line = self.load_corpus(test_file)
        x_test,y_test,_,_,_ = self.digitalizing(cand_test_cs, cand_test_label, vectorizer=vecer, tfidf_transformer=tfidformer, pcaor=pca_obj,is_train=False)

        return x_train,y_train,x_test,y_test,cand_test_line





class SVMMethod:
    def __init__(self,version):
        self.version=version
        self.train_file=os.path.join(DATA_DIR,version,"train_raw")
        self.test_file=os.path.join(DATA_DIR,version,"validate_raw")
        self.featurer = Featurer()
        self.svmm = LogisticRegression(class_weight='balanced', C=1.0)


    def process(self):
        x_train,y_train,x_test,y_test,test_content = self.featurer(self.train_file, self.test_file)
        self.svmm.fit(x_train,y_train)
        y_pred = self.svmm.predict(x_test)
        for one_pred, one_test, one_content in zip(y_pred, y_test, test_content):
            print one_content, one_pred
        print(metrics.classification_report(y_test, y_pred))
        



if __name__=="__main__":
    svm_obj = SVMMethod(FLAGS.version)
    svm_obj.process()

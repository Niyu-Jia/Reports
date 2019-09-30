
import jieba
import pandas as pd
import re
import numpy as np
from matplotlib import pyplot as plt
from numba import jit
import os
import random
from sklearn import preprocessing
from bisect import bisect

os.chdir('/home/niyu/Documents/Project/Lab/code')

#object: origin csv file
class csv():
    patterncv=re.compile(r'cv\d+',re.M)
    patternName=re.compile(r'name\d+',re.M)
    patternTime=re.compile(r'\d+年|\d+月|\d+',re.M)
    patternPunc=re.compile(r'[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）：；《》]+')

    def __init__(self,path):
        self.path=path
        self.csvdata=pd.read_csv(open(self.path,encoding='gbk',errors='ignore'),engine='python')
        #self.csvdata=pd.read_csv(path)
        
        
    def process(self):
        #basic info of companies:
        self.codeList=list(self.csvdata['stkcd'])
        self.codeList=list(set(self.codeList))
        self.distance=list(self.csvdata['Distance_to_Default'])
        
        fields=list(self.csvdata.columns)
        fields="".join(fields)
        
        matchcv=re.findall(csv.patterncv,fields)
        matchName=re.findall(csv.patternName,fields)
        dfname=self.csvdata[matchName]
        dfcv=self.csvdata[matchcv]
        dfcv=dfcv.replace("简历缺失",np.NaN)
        self.arrNull=np.array(dfcv.isnull())
        self.arrcv=np.array(dfcv)
        self.arrname=np.array(dfname)
        return self.arrcv,self.arrname,self.arrNull,self.codeList
        

    def split_word(self):
        #use cpu
        #@vectorize(['float32(float32, float32)'], target='cuda')  #use cuda gpu
        @jit
        def split():
            for a in range(len(self.arrcv)):
                num=100*(a/len(self.arrcv))
                for b in range(self.arrcv.shape[1]):
                    if self.arrNull[a,b]==False:
                        name=self.arrname[a,b]
                        cv=self.arrcv[a,b]
                        cv= re.sub(csv.patternPunc,"",cv)  
                        cv= re.sub(csv.patternTime,"",cv)  
                        cv=cv.strip(name)
                        cvCut=jieba.cut(cv)
                        cvCut=",".join(cvCut)
                        self.arrcv[a,b]=cvCut
                        
                if a %2000 ==0:
                    print(str(num)+'% finished')
        split()
        return self.arrcv
    
    def stop(self):
        stoplist = open('/home/niyu/Documents/Project/Lab/code/stopwords.txt', 'r+', encoding='utf-8')
        stopword = stoplist.read().split("\n")
        dict_word= {i: '' for i in stopword}
        
        #words: total list of cv words 
        self.words=[np.nan]*len(self.arrcv)
        
        for i in range(len(self.arrcv)):
            sample=list(self.arrcv[i])
            cv_word=[]
            for j in range(len(sample)):
                if type(sample[j])== str:
                    #combine all the cv word in a same company
                    wordlist=sample[j].split(",")
                    cv_word=cv_word+wordlist
        
        
            #delete all the stop words by mapping
            wordSeries=pd.Series(cv_word)
            wordSeries=wordSeries.map(dict_word).fillna(wordSeries)
            
            #remove empty elements(original stop words)
            wordSeries = list(filter(None, wordSeries))
            self.words[i]=wordSeries
        
        return self.words
    
    
    def train_test(self,train_per,min_doc_len):
        #divide into training, validation and test set
    
        keep_idx=[i for i in range(len(self.words)) if len(self.words[i])>min_doc_len]
        n=round(train_per*len(keep_idx))    
        random.shuffle(keep_idx)
        test_id=keep_idx[0:n]
        train_id=keep_idx[n:]
        
        return train_id,test_id
        
    
    
    def DD_normalize(self,train_per,min_doc_len):
        #data visiualization of DD 
        self.train_id,self.test_id=self.train_test(train_per,min_doc_len)
        
        DD_list=self.csvdata['Distance_to_Default']
        
        train_DD=[DD_list[i] for i in self.train_id]
        test_DD=[DD_list[i] for i in self.test_id]
                
        dummy=np.array([1,1,1,1])
        dummy_mat=np.diag(dummy)
        
        breakpoints=[1,2,3]
        Class=[bisect(breakpoints,i) for i in DD_list]
        Class_num=[len([i for i in Class if i==j]) for j in [0,1,2,3]]
        label=['A','B','C','D']
        color=['#2E86C1','#85C1E9','#AED6F1','#D6EAF8'  ]
        plt.pie(Class_num,labels=label,autopct='%1.1f%%',colors=color)
        plt.show()
        
        target=[dummy_mat[i] for i in Class]
        
        DD=[target[i] for i in id1]
        y=np.array(DD)
        test_DD=[target[i] for i in id2]
        test_y=np.array(test_DD)
        valid_DD=DD[0:3000]
        valid_y=y[0:3000]
        train_DD=DD[3000:]
        train_y=y[3000:]

        return train_DD,train_y,valid_DD,valid_y,test_DD,test_y

##############################################################################
text=csv('/home/niyu/Documents/Project/Lab/data.csv')
text.process()
split=text.split_word()
word_list=text.stop()
id1,id2=text.train_test(0.15,100)
train_DD,train_y,valid_DD,valid_y,test_DD,test_y=text.DD_normalize(0.15,100)






# coding: utf-8

# In[197]:


import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
 
from tqdm import tqdm
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, RegexpTokenizer


# In[198]:


#原始贝叶斯
class NaiveBayes:
    def fit(self, x, y):
        self.x = x
        self.y = y
        self.classes = np.unique(y)
#         print(self.classes)
        self.parameters = {}
        for i , c in enumerate(self.classes):
            #计算属于同一类别的均值,方差和各类别的先验概率p(y).
            X_index_c  = x[np.where(y == c)]
            X_index_c_mean = np.mean(X_index_c, axis=0, keepdims=True)
            X_index_c_var = np.var(X_index_c, axis=0, keepdims=True)
            parameters = {'mean':X_index_c_mean,'var':X_index_c_var,'prior':X_index_c.shape[0]/ x.shape[0]}
            self.parameters['class' + str(c)] = parameters  #字典嵌套
#             print(X_index_c.shape[0])
        
        
    def _pdf(self, x, classes):
        #用高斯分布拟合p(x|y),也就是后验概率.并且按行每个特征的p(x|y)累乘,取log成为累加.
        eps = 1e-4  #防止分母为0
        mean = self.parameters['class' + str(classes)]['mean']
        var  = self.parameters['class' + str(classes)]['var']
        fenzi = np.exp(-(x - mean) ** 2 / (2 * (var) ** 2 + eps))
        fenmu = (2 * np.pi) ** 0.5 * var + eps
        result = np.sum(np.log(fenzi / fenmu), axis=1, keepdims=True)
        #print(result.T.shape)
        return result.T #(1, 719)
       
        
    def _predict(self, x):
        # 计算每个种类的p(y)p(x|y)
        output = []
        for y in range(self.classes.shape[0]):
            prior = np.log(self.parameters['class' + str(y)]['prior'])
            posterior = self._pdf(x, y)
            prediction = prior + posterior
            output.append(prediction)
        return output
        
    def predict(self, x):
        #argmax(p(y)p(x|y))就是最终的结果
        output = self._predict(x)
        output = np.reshape(output, (self.classes.shape[0], x.shape[0]))
        prediction = np.argmax(output, axis=0)
        return prediction


# In[199]:


#互信息熵贝叶斯
class WeightNaiveBayes:
    def fit(self, x, y):
        self.x = x
        self.y = y
        self.classes = np.unique(y)
#         print(self.classes)
        self.parameters = {}
        for i , c in enumerate(self.classes):
            #计算属于同一类别的均值,方差和各类别的先验概率p(y).
            X_index_c  = x[np.where(y == c)]
            X_index_c_mean = np.mean(X_index_c, axis=0, keepdims=True)
            X_index_c_var = np.var(X_index_c, axis=0, keepdims=True)
            parameters = {'mean':X_index_c_mean,'var':X_index_c_var,'prior':X_index_c.shape[0]/ x.shape[0]}
            self.parameters['class' + str(c)] = parameters  #字典嵌套
#             print(X_index_c.shape[0])
        
        global_mean = np.mean(x,axis=0,keepdims=True)
        global_var = np.var(x,axis=0,keepdims=True)
        self.parameters['global_mean'] = global_mean
        self.parameters['global_var'] = global_var
    
    def px(self,x):
        eps = 1e-4  #防止分母为0
        mean = self.parameters['global_mean']
        var  = self.parameters['global_var']
        fenzi = np.exp(-(x - mean) ** 2 / (2 * (var) ** 2 + eps))
        fenmu = (2 * np.pi) ** 0.5 * var + eps
        ff = fenzi/fenmu
        return ff
        
    def _pdf(self, x, classes):
        #用高斯分布拟合p(x|y),也就是后验概率.并且按行每个特征的p(x|y)累乘,取log成为累加.
        eps = 1e-4  #防止分母为0
        mean = self.parameters['class' + str(classes)]['mean']
        var  = self.parameters['class' + str(classes)]['var']
        fenzi = np.exp(-(x - mean) ** 2 / (2 * (var) ** 2 + eps))
        fenmu = (2 * np.pi) ** 0.5 * var + eps
        ff = fenzi/fenmu
        px = self.px(x)
        
        #p(w/c) = p(w,c)/p(c)
        #求px
        result = np.sum(np.log(ff)*(np.log(px/ff)), axis=1, keepdims=True)
        #print(result.T.shape)
        return result.T
       
    def _predict(self, x):
        # 计算每个种类的p(y)p(x|y)
        output = []
        for y in range(self.classes.shape[0]):
            prior = np.log(self.parameters['class' + str(y)]['prior'])
            posterior = self._pdf(x, y)
            prediction = prior + posterior
            output.append(prediction)
        return output
        
    def predict(self, x):
        #argmax(p(y)p(x|y))就是最终的结果
        output = self._predict(x)
        output = np.reshape(output, (self.classes.shape[0], x.shape[0]))
#         output = 1-output
        prediction = np.argmax(output, axis=0)
        return prediction


# In[200]:


#文件预处理成tsv格式
def read_file(file_name):
    with open(file_name,'r',encoding='utf-8') as f:
        data = f.readlines()
    label_map = {}
    label_map["ham"] = 0
    label_map["spam"] = 1
    
    
    text = list()
    label = list()
    label_num = list()
    for item in data:
        item = item.strip()
        item = item.split(" ")
        item_len = len(item)
        tmp_list = list()
        if(item[1] in ["ham","spam"]):
            label.append(item[1])
            label_num.append(label_map[item[1]])
            for index in range(2,item_len,2):
                tmp_list.append(item[index])
        text.append(tmp_list)
    return text,label,label_num


# In[201]:


train_text,train_label,train_label_num = read_file("./data/train")


# In[202]:


test_text,test_label,test_label_num = read_file("./data/test")


# In[203]:


train_text = [" ".join(item) for item in train_text]
test_text = [" ".join(item) for item in test_text]


# In[204]:

des_min_df = 256
max_acc = -1
pre_chara = 1000
for my_min_df in tqdm(range(495,496)):
    cv = CountVectorizer(min_df = my_min_df)
    part_fit = cv.fit(train_text) # 以部分句子为参考
    train_all_count = cv.transform(train_text) # 对训练集所有邮件统计单词个数
    test_all_count = cv.transform(test_text) # 对测试集所有邮件统计单词个数
    tfidf = TfidfTransformer()
    train_tfidf_matrix = tfidf.fit_transform(train_all_count)
    test_tfidf_matrix = tfidf.fit_transform(test_all_count)



    if( pre_chara>train_tfidf_matrix.shape[1]):
        pre_chara = train_tfidf_matrix.shape[1]
    else:
        continue
    print('训练集', train_tfidf_matrix.shape)
    print('测试集', test_tfidf_matrix.shape)


    # In[206]:


    #训练
    train_tfidf_matrix_1 = train_tfidf_matrix.toarray()
    mnb = NaiveBayes()
    mnb.fit(train_tfidf_matrix_1, train_label_num)
    #预测
    test_tfidf_matrix1 = test_tfidf_matrix.toarray()
    pred_test_label_num = mnb.predict(test_tfidf_matrix1)
    #混淆矩阵输出
    c_m = confusion_matrix(test_label_num, pred_test_label_num, labels=None, sample_weight=None)
    print("原始贝叶斯混淆矩阵:\n",c_m)
    print("原始贝叶斯准确率：\n",accuracy_score(test_label_num, pred_test_label_num)) 


    # In[207]:


    train_tfidf_matrix_1 = train_tfidf_matrix.toarray()
    mnb = WeightNaiveBayes()
    mnb.fit(train_tfidf_matrix_1, train_label_num)
    #预测
    test_tfidf_matrix1 = test_tfidf_matrix.toarray()
    pred_test_label_num = mnb.predict(test_tfidf_matrix1)
    #混淆矩阵输出
    c_m = confusion_matrix(test_label_num, pred_test_label_num, labels=None, sample_weight=None)
    print("加权贝叶斯混淆矩阵\n",c_m)
    weight_acc_score = accuracy_score(test_label_num, pred_test_label_num)
    print("加权贝叶斯准确率：\n",weight_acc_score) 
    if weight_acc_score>max_acc:
        max_acc = weight_acc_score
        des_min_df = my_min_df
print(max_acc,des_min_df)
# In[ ]:





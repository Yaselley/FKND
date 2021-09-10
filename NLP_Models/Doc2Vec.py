# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 18:13:32 2020

@author: Yassine ELKHEIR
"""
import numpy as np
import re
import string
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from gensim import utils
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier



class doc2vec_method :
    extra_words = set(stopwords.words("english"))
    
    #direction to data files
    def __init__(self, way) :
        self.way = way
    
    #clean_data__________________________________________________________________________ 
    def clean(self,data) : 
        #replace strange ponctuations with a simple space
        data = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", str(data))
        #lower ==> same size, split=> split sentence to a list of words 
        data = data.lower().split()
        
        #________________________________________________________________________________
        data_prov = []
        for word in data :
            if not word in self.extra_words :    # words like and a ... preposition in general 
                if len(word) > 2 :               # let just words with len > 1 
                    data_prov.append(word)
       
        data = data_prov  
        #________________________________________________________________________________
        
        # change the list to a string with spaces 
        data = " ".join(data)                                         
        # another method in order to eliminate punctuation eliminate 
        data = data.translate(str.maketrans("","",string.punctuation))
        
        
        return data
    
    #Construction_of_sentences_____________________________________________________________________________
    #"Hi I am Yassine" => words=['Hi', 'I', 'am', 'yassine'], tags=['Text_index']
    #a gensim construction of datas 
    
    def constructSentences(self,data):
        sentences = []
        for index, row in data.iteritems():
            sentences.append(LabeledSentence(utils.to_unicode(row).split(), ['Text' + '_%s' % str(index)]))
        return sentences
    
    #______________________________________________________________________________________________________
    
    def getdata(self) :
        
        vector_dimension=300
        #import
        train_data = pd.read_csv(self.way+"/train.csv")
        test_data = pd.read_csv(self.way+"/test.csv")
        submit_data = pd.read_csv(self.way+"/submit.csv")
        
        #clean_texts 
        #__________________________________________________
        for i in range(len(train_data)):
            #loc is like data[i]==> text of row indexed i 
            text = train_data.loc[i,'text']
            #clean text row_i
            train_data.loc[i,'text'] = self.clean(text)
        
        for i in range(len(test_data)):
            text = test_data.loc[i,'text']
            test_data.loc[i,'text'] = self.clean(text)
        #__________________________________________________
        
        
        
        train_data_text_doc2vec = self.constructSentences(train_data['text']) 
        train_data_label_doc2vec = train_data['label'].values
        
        test_data_text_doc2vec = self.constructSentences(test_data['text'])
        test_data_label_doc2vec = submit_data['label'].values
        
        
        #initialized our model 
        text_model = Doc2Vec(window=5, vector_size=vector_dimension, sample=1e-4, negative=5, workers=7, epochs=10,seed=1)
        text_model.build_vocab(train_data_text_doc2vec)
        
        # train our model with x (train.csv)
        # default epochs = 10 
        text_model.train(train_data_text_doc2vec, total_examples=text_model.corpus_count, epochs=text_model.iter)
        
        train_size = len(train_data_text_doc2vec)
        test_size = len(test_data_text_doc2vec)
        
        text_train_arrays = np.zeros((train_size, vector_dimension))
        text_test_arrays = np.zeros((test_size, vector_dimension))
        

        for i in range(train_size):
            text_train_arrays[i] = text_model.docvecs['Text_' + str(i)]
        
        for i in range(test_size) : 
            text_test_arrays[i] = text_model.docvecs['Text_' + str(i)]
        
        return text_train_arrays, text_test_arrays, train_data_label_doc2vec, test_data_label_doc2vec

#___________________________________________Start_our_Models___________________________________________________

start_process = doc2vec_method('C:/Users/Yassine ELKHEIR/Desktop/EURECOM COURSES/MALIS/Project/fake-news-data')

x_train,x_test,y_train,y_test = start_process.getdata()

#___________________________________________Naive_Bayes_________________________________________________________
naive_bayes_gaussian = GaussianNB()
naive_bayes_gaussian.fit(x_train,y_train)
y_pred_doc2vec_gnb = naive_bayes_gaussian.predict(x_test)
accurancy_doc2vec_gnb = np.mean(y_test == y_pred_doc2vec_gnb)

#__________________________________________Logistic_Regression__________________________________________________
       
logistic_regression = LogisticRegression(
    penalty = 'l2', 
    tol = 0.0001, 
    C=1.0, 
    dual = False, 
    class_weight = None, 
    multi_class = 'multinomial', 
    solver = 'saga', 
    max_iter = 100, 
    n_jobs = 1).fit(x_train, y_train)

y_pred_doc2vec_lr = logistic_regression.predict(x_test)
accurancy_doc2vec_lr = np.mean(y_test == y_pred_doc2vec_lr)

#________________________________________linear SVM classifier___________________________________________________

svm = svm.LinearSVC().fit(x_train, y_train)
predicted_doc2vec_svm = svm.predict(x_test)
accuracy_doc2vec_svm = np.mean(y_test == predicted_doc2vec_svm) 

#_______________________________________Random_Forest_Classifier_________________________________________________   

rd = RandomForestClassifier(n_estimators=300,n_jobs=3).fit(x_train, y_train)

predicted_doc2vec_rf = rd.predict(x_test)
accuracy_doc2vec_rf = np.mean(y_test == predicted_doc2vec_rf) 

#__________________________________________sgd classifier________________________________________________________

sgd = SGDClassifier(penalty='l2', alpha=1e-3).fit(x_train, y_train)

predicted_doc2vec_sgd = sgd.predict(x_test)
accuracy_doc2vec_sgd = np.mean(y_test == predicted_doc2vec_sgd)

#________________________________________K_neighbor classifier___________________________________________________

kn = KNeighborsClassifier(n_neighbors=3).fit(x_train, y_train)

predicted_doc2vec_kn = kn.predict(x_test)
accuracy_doc2vec_kn = np.mean(y_test == predicted_doc2vec_kn) 

#_______________________________________Multi_Layer_Perceptron___________________________________________________

mlpclassifier = MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(30,),random_state=1)

mlpclassifier.fit(x_train, y_train)
predicted_doc2vec_mlp = mlpclassifier.predict(x_test)
accuracy_doc2vec_mlp = np.mean(predicted_doc2vec_mlp == y_test) 



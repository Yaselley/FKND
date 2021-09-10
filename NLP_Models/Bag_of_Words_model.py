# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 18:55:18 2020

@author: Yassine ELKHEIR
"""

"""
________________________________________________________________________________________________
This is one of the traditional models used to convert text to numbers. 
In a given set of records, certain words are identified as significant words.
The models generally have these prebuilt set of words. 
Consider the count of such words equal to ‘n’. 
So the vector generated for the word will be of length n (number of components). 
If the ith word is present in the document (or text), 
then the ith component in the corresponding vector will 1, 0 otherwise. 
This way the vectors are generated for all the documents (or articles). 
This is the simplest vectorization technique. 
The main challenge in this technique is that all the words (among the ones you chose) 
are weighted uniformly which is not true in all scenarios as the importance of the word
differs with respect to context.
________________________________________________________________________________________________ 
"""
import Prep_data
import pandas as pd
import csv
import re
import numpy as np
import nltk
from nltk.stem import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import seaborn as sb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import  LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

eng_stemmer = SnowballStemmer('english')
stopwords = set(nltk.corpus.stopwords.words('english'))

#______________________________________________________________________________
#clean some words, 
#for example : word. => word
   
list_clean = ["&","'","-","_","@","/",":",";",",","."]
def clean_string(s):
    new_w =str()
    for i in range(len(s)) : 
        if s[i] not in list_clean : 
            new_w+=s[i]
    return new_w
#______________________________________________________________________________

#process the data
#get ride of some inutile words 
def process_data(data): #a simple list or tuple  
    # même_taille
    
    # adapt so that this function can works for tuple and list
    #__________________________________________________________________________
    L = [data[0]]
    if data[0] is not list : 
        L[0] = data
    data = L 
    #__________________________________________________________________________
    
    
    #__________________________________________________________________________
    #w.lower() ==> ["HI"=>"hi"]
    #w.split() ==> ['Hi we are the best group'] => ['Hi','we','are','the','best','group']    
    tokens = [w.lower().split() for w in data]
    #__________________________________________________________________________

    #__________________________________________________________________________
    data_fine=[]
    for w in tokens[0] :
        ww = clean_string(w)
        if not ww in stopwords and ww!="":
            if not ww[0].isdigit() :
                data_fine.append(ww)
    #__________________________________________________________________________
    
    return data_fine


#______________________________________________________________________________
#[process_data(data) for data in datas]
#______________________________________________________________________________    
def process_datas(datas):
    process_datas_L = []
    for data in datas : 
        process_datas_L.append(" ".join(process_data(data)))
    return process_datas_L

#______________________________________________________________________________
 #countVectorize
train_data = process_datas(Prep_data.train_set_text)
test_data = process_datas(Prep_data.test_set_text)
countV = CountVectorizer()
train_countV = countV.fit_transform(train_data)
#tfidfV = TfidfTransformer()
#train_tfidf = tfidfV.fit_transform(train_countV)
#tfidf_ngram = TfidfVectorizer(stop_words='english',ngram_range=(1,4),use_idf=True,smooth_idf=True)
#train_countV = train_countV.toarray()

#______________________________________________________________________________
#present our models that we are going tu use 

#__________________________Bayes_Classifier____________________________________    
bayes_pipeline_BoW = Pipeline([('countV_bayes',countV),
        ('bayes_classifier',MultinomialNB())])

bayes_pipeline_BoW.fit(train_data,Prep_data.train_set_label)
#predicted_bayes = bayes_pipeline.predict(test_set_text)
predicted_Bow_bayes = bayes_pipeline_BoW.predict(test_data)
accuracy_BoW_bayes = np.mean(predicted_Bow_bayes == Prep_data.test_set_label)
  
#__________________________Logestic_Regression_________________________________
logR_pipeline_BoW = Pipeline([('countV_logR',countV),
        ('logR_classifier',LogisticRegression(penalty="l2",C=1))])

logR_pipeline_BoW.fit(train_data,Prep_data.train_set_label)
predicted_Bow_logR = logR_pipeline_BoW.predict(test_data)
accuracy_BoW_lr = np.mean(predicted_Bow_logR == Prep_data.test_set_label)     
    
#_______________________linear SVM classifier__________________________________
svm_pipeline_BoW = Pipeline([('countV_svm',countV),
        ('svm_classifier',svm.LinearSVC())])

svm_pipeline_BoW.fit(train_data,Prep_data.train_set_label)
predicted_Bow_svm = svm_pipeline_BoW.predict(test_data)
accuracy_BoW_svm = np.mean(predicted_Bow_svm == Prep_data.test_set_label) 

#_____________________Random_Forest_Classifier_________________________________
randomF_pipeline_BoW = Pipeline([('countV_rf',countV),
        ('rf_classifier',RandomForestClassifier(n_estimators=300,n_jobs=3))])    

randomF_pipeline_BoW.fit(train_data,Prep_data.train_set_label)
predicted_Bow_rf = randomF_pipeline_BoW.predict(test_data)
accuracy_BoW_rf = np.mean(predicted_Bow_rf == Prep_data.test_set_label) 

#___________________________sgd classifier_____________________________________
sgd_pipeline_BoW = Pipeline([
         ('countV_sgd',countV),
         ('sgd_classifier',SGDClassifier(penalty='l2', alpha=1e-3))])

sgd_pipeline_BoW.fit(train_data,Prep_data.train_set_label)
predicted_Bow_sgd = sgd_pipeline_BoW.predict(test_data)
accuracy_BoW_sgd = np.mean(predicted_Bow_sgd == Prep_data.test_set_label)

#________________________K_neighbor classifier_________________________________
kneighbors_pipeline_BoW = Pipeline([('countV_rf',countV),
        ('kneighbor_classifier', KNeighborsClassifier(n_neighbors=3))])

kneighbors_pipeline_BoW.fit(train_data,Prep_data.train_set_label)
predicted_Bow_kn = kneighbors_pipeline_BoW.predict(test_data)
accuracy_BoW_kn = np.mean(predicted_Bow_kn == Prep_data.test_set_label) 

#________________________Multi_Layer_Perceptron________________________________
mlp_pipeline_BoW = Pipeline([('countV_rf',countV),
        ('mlp_classifier', MLPClassifier(solver='lbfgs',
                                         alpha=1e-5,
                                         hidden_layer_sizes=(30,),
                                         random_state=1))])


mlp_pipeline_BoW.fit(train_data,Prep_data.train_set_label)
predicted_Bow_mlp = mlp_pipeline_BoW.predict(test_data)
accuracy_BoW_mlp = np.mean(predicted_Bow_mlp == Prep_data.test_set_label) 



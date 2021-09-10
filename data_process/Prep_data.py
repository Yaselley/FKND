# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 19:40:34 2020

@author: Yassine ELKHEIR
"""
import pandas as pd
from sklearn.neural_network import MLPClassifier

#Prep_data is used to clean data from non defined "text" or "label"
#so that we can well fit our model  
#charge files
way = 'C:/Users/Yassine ELKHEIR/Desktop/EURECOM COURSES/MALIS/Project/fake-news-data'

test_filename = way+'/test.csv'
train_filename = way+'/train.csv'
sumbit_filename = way+'/submit.csv'

#pandas read csv
train_news = pd.read_csv(train_filename,error_bad_lines = False)
test_news = pd.read_csv(test_filename,error_bad_lines = False)
submit_news = pd.read_csv(sumbit_filename,error_bad_lines = False)

#to avoid problem 
train_news['text'].astype(str)
test_news['text'].astype(str)


#getride of raws without text 'train' 'test'
train_set_text = train_news['text'].to_numpy()
train_set_label = train_news['label'].to_numpy()
test_set_text = test_news['text'].to_numpy()
test_set_label = submit_news['label'].to_numpy()

def clean_nan(list_text, list_label) : 
    L,S=[],[]
    for elem in range(len(list_text)) :
        if str(list_text[elem]) != "nan":
            if str(list_text[elem])!= "":
                L.append(list_text[elem])
                S.append(list_label[elem])           
    return L,S

#clean data ==W what'we are going to use 

train_set_text,train_set_label = clean_nan(train_set_text,train_set_label)
test_set_text,test_set_label = clean_nan(test_set_text,test_set_label)


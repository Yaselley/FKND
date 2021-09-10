# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 15:05:18 2020

@author: Yassine ELKHEIR
"""

import Prep_data
import Bag_of_Words_model 
import TF_IDF_Model 
import Doc2Vec 
import matplotlib.pyplot as plt

def plot_cmat(yte, ypred):
    '''Plotting confusion matrix'''
    skplt.plot_confusion_matrix(yte,ypred)
    plt.show()

#redefine 
def build_confusion_matrix(classifier):
    
    k_fold = KFold(n_splits=5)
    scores = []
    confusion = np.array([[0,0],[0,0]])

    for train_ind, test_ind in k_fold.split(sanstitre8.train_news):
        train_text = sanstitre8.train_news.iloc[train_ind]['text'] 
        train_y = sanstitre8.train_news.iloc[train_ind]['label']
    
        test_text = sanstitre8.train_news.iloc[test_ind]['text']
        test_y = sanstitre8.train_news.iloc[test_ind]['label']
        
        classifier.fit(train_text,train_y)
        predictions = classifier.predict(test_text)
        
        confusion += confusion_matrix(test_y,predictions)
        score = f1_score(test_y,predictions)
        scores.append(score)
    
    return (print('Total statements classified:', len(sanstitre8.train_news)),
    print('Score:', sum(scores)/len(scores)),
    print('score length', len(scores)),
    print('Confusion matrix:'),
    print(confusion))
    
names = ['NB', 'LR', 'SVM', 'RandomF', 'sgd', 'K_nbr', 'MLP']
values_BoW = [Bag_of_Words_model.accuracy_BoW_bayes,
              Bag_of_Words_model.accuracy_BoW_lr,
              Bag_of_Words_model.accuracy_BoW_svm,
              Bag_of_Words_model.accuracy_BoW_rf,
              Bag_of_Words_model.accuracy_BoW_sgd,
              Bag_of_Words_model.accuracy_BoW_kn,
              Bag_of_Words_model.accuracy_BoW_mlp
             ]

values_TFIDF = [TF_IDF_Model.accuracy_TFIDF_bayes,
                TF_IDF_Model.accuracy_TFIDF_logR,
                TF_IDF_Model.accuracy_TFIDF_svm,
                TF_IDF_Model.accuracy_TFIDF_rf,
                TF_IDF_Model.accuracy_TFIDF_sgd,
                TF_IDF_Model.accuracy_TFIDF_kn,
                TF_IDF_Model.accuracy_TFIDF_mlp
        ]

values_Doc2Vec = [Doc2Vec.accurancy_doc2vec_gnb,
                  Doc2Vec.accurancy_doc2vec_lr,
                  Doc2Vec.accuracy_doc2vec_svm,
                  Doc2Vec.accuracy_doc2vec_rf,
                  Doc2Vec.accuracy_doc2vec_sgd,
                  Doc2Vec.accuracy_doc2vec_kn,
                  Doc2Vec.accuracy_doc2vec_mlp
        ]

plt.bar(names, values_BoW ,color="orange",title="Bag of Words")
plt.bar(names, values_TFIDF ,color="red",title="TF_IDF Model")
plt.bar(names, values_Doc2Vec ,color="blue",title="Doc2Vec Model")
plt.bar(names, values_BoW,values_TFIDF, values_Doc2Vec , color = "#EDFF91")
plt.show()        


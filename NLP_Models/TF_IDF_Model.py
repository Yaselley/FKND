# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 13:37:48 2020

@author: Yassine ELKHEIR
"""

"""
TF-IDF stands for Term Frequency-Inverse Document Frequency. 
In this model, the words are assigned a weight based on the frequency 
of appearance. The model has 2 parameters as mentioned in the name. 
The term frequency component adjusts the weight proportionally with 
the number of times the word appears in the document with respect to 
the total number of words in that document. 
Inverse document frequency component identifies unique words in the set
of documents and increases weight accordingly.
If a particular word is appearing in most of the documents, 
then its weight is reduced as it will not help anyway in distinguishing 
the documents. Though this model weights the words based on the frequency
and unique factors, it is not able to capture the meaning of the word.
"""
#%%
import Bag_of_Words_model
from sklearn.neural_network import MLPClassifier

tfidfV = TfidfTransformer()
train_tfidf = tfidfV.fit_transform(Bag_of_Words_model.train_countV)
tfidf_ngram = TfidfVectorizer(stop_words='english',ngram_range=(1,4),use_idf=True,smooth_idf=True)
#tfidf_ngram = TfidfVectorizer(stop_words='english',ngram_range=(1,4),use_idf=True,smooth_idf=True)
#train_countV = train_countV.toarray()
#%%
#__________________________Bayes_Classifier____________________________________    
bayes_pipeline_TFIDF = Pipeline([('tfidfv_bayes',tfidf_ngram),
        ('bayes_classifier',MultinomialNB())])

bayes_pipeline_TFIDF.fit(Bag_of_Words_model.train_data,Prep_data.train_set_label)
#predicted_bayes = bayes_pipeline.predict(test_set_text)
predicted_TFIDF_bayes = bayes_pipeline_TFIDF.predict(Bag_of_Words_model.test_data)
accuracy_TFIDF_bayes = np.mean(predicted_TFIDF_bayes == Prep_data.test_set_label)

#%%
#__________________________Logestic_Regression_________________________________
logR_pipeline_TFIDF = Pipeline([('tfidfv_logR',tfidf_ngram),
        ('logR_classifier',LogisticRegression(penalty="l2",C=1))])

logR_pipeline_TFIDF.fit(train_data,train_set_label)
predicted_TFIDF_logR = logR_pipeline_TFIDF.predict(test_data)
accuracy_TFIDF_logR = np.mean(predicted_TFIDF_logR == test_set_label)     
#%%    
#_______________________linear SVM classifier__________________________________
svm_pipeline_TFIDF = Pipeline([('tfidfv_logR',tfidf_ngram),
        ('svm_classifier',svm.LinearSVC())])

svm_pipeline_TFIDF.fit(Bag_of_Words_model.train_data,Prep_data.train_set_label)
predicted_TFIDF_svm = svm_pipeline_TFIDF.predict(Bag_of_Words_model.test_data)
accuracy_TFIDF_svm = np.mean(predicted_TFIDF_svm == Prep_data.test_set_label) 
#%%
#_____________________Random_Forest_Classifier_________________________________
randomF_pipeline_TFIDF = Pipeline([('tfidfv_rf',tfidf_ngram),
        ('rf_classifier',RandomForestClassifier(n_estimators=300,n_jobs=3))])    

randomF_pipeline_TFIDF.fit(Bag_of_Words_model.train_data,Prep_data.train_set_label)
predicted_TFIDF_rf = randomF_pipeline_TFIDF.predict(Bag_of_Words_model.test_data)
accuracy_TFIDF_rf = np.mean(predicted_TFIDF_rf == Prep_data.test_set_label) 
#%%
#___________________________sgd classifier_____________________________________
sgd_pipeline_TFIDF = Pipeline([('tfidfv_sgd',tfidf_ngram),
         ('sgd_classifier',SGDClassifier(penalty='l2', alpha=1e-3))])

sgd_pipeline_TFIDF.fit(Bag_of_Words_model.train_data,Prep_data.train_set_label)
predicted_TFIDF_sgd = sgd_pipeline_TFIDF.predict(Bag_of_Words_model.test_data)
accuracy_TFIDF_sgd = np.mean(predicted_TFIDF_sgd == Prep_data.test_set_label)
#%%
#________________________K_neighbor classifier_________________________________
kneighbors_pipeline_TFIDF = Pipeline([('tfidfv_kn',tfidf_ngram),
        ('kneighbor_clf', KNeighborsClassifier(n_neighbors=3))])

kneighbors_pipeline_TFIDF.fit(Bag_of_Words_model.train_data,Prep_data.train_set_label)
predicted_TFIDF_kn = kneighbors_pipeline_TFIDF.predict(Bag_of_Words_model.test_data)
accuracy_TFIDF_kn = np.mean(predicted_TFIDF_kn == Prep_data.test_set_label) 

#%%
#________________________Multi_Layer_Perceptron________________________________
mlp_pipeline_TFIDF = Pipeline([('tfidfv_kn',tfidf_ngram),
        ('mlp_classifier', MLPClassifier(solver='lbfgs',
                                         alpha=1e-5,
                                         hidden_layer_sizes=(30,),
                                         random_state=1))])


mlp_pipeline_TFIDF.fit(Bag_of_Words_model.train_data,Prep_data.train_set_label)
predicted_TFIDF_mlp = mlp_pipeline_TFIDF.predict(Bag_of_Words_model.test_data)
accuracy_TFIDF_mlp = np.mean(predicted_TFIDF_mlp == Prep_data.test_set_label) 

#%%
  

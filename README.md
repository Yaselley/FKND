# Machine Learning and Intelligent System 
## Abstract
Our modern society is struggling with an unprecedented amount of online misinformation, which does
harm to democracy, economics, and national security. Since 2016, Fake news has been well tuned and
aligned with United states elections, and that came back to the media recently with the approach of
elections, so we tried though that project, based on the highly motivation nowadays “US Election” to
develop a reliable model that classifies a given news article as either fake or true. We tried to simulate
the different programs made by large big companies, like google, on sources and web sites as well. Our
simple models give us accuracy up to 76%.

__Key Words__ : Bag of words, TF-IDF, Doc Vec, Naive Bayes, Logistic Regression, SVM, Random forest, MLP.

![alt text](https://raw.githubusercontent.com/Yaselley/FKND/main/images/fake_true_1.PNG)

## Bag of Words Model :
Using the Bag of words model to extract
features, we observed an accuracy of
74% percent on our testing set, using
k_near neighbors algorithm model.


![alt text](https://raw.githubusercontent.com/Yaselley/FKND/main/images/fake_true_Bow.PNG)

## TF-IDF :
The increase in accuracy for the models
made previously was foreseen, since the
TF-IDF method does not equalize all the
inputs (as in the bag of words method),
but weights them according to their
appearance in the inputs. As last time, the
highest accuracy was reached by k Near
Neighbors for an accuracy of 74%

![alt text](https://raw.githubusercontent.com/Yaselley/FKND/main/images/fake_true_TFIDF.PNG)

## DOC2VEC :
We learned that the most frequent
tokens are not the most casual ones,
so the context of the word that
Doc2Vec takes into consideration in
fitting doesn’t affect a lot,
unfortunately that drops accuracy.

![alt text](https://raw.githubusercontent.com/Yaselley/FKND/main/images/fake_true_doc2vec.PNG)

## Notes : 
__Preparation of Data__ : ==> _data_process_ Folder

__Test_NLP_Models__ : ==> _NLP_Models_ Folder 

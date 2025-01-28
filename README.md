# Hate Speech Detection in Tweets
This repository contains a project focused on detecting hate speech, offensive language, and neutral content in tweets using machine learning models and NLP techniques.

Detecting hate speech has become crucial for maintaining safe online spaces. This project implements and evaluates two machine learning models (XGBoost and SVM) to classify tweets into three categories:
* hate speech
* offensive language
* neutral

we used <a href="https://github.com/t-davidson/hate-speech-and-offensive-language">Thomas Davidson’s Hate Speech and Offensive Language dataset</a>, with preprocessing including text cleaning, normalization, tokenization, and word embeddings. Models are evaluated based on accuracy, precision, recall, F1 score, and confusion matrix analysis.

In the result, XGBoost showed superior performance compared to SVM, leveraging Word2Vec embeddings to enhance classification accuracy.
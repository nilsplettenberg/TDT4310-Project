# TDT4310-Project
This repository contains all the code for the student project in TDT4310 Intelligent Text Analytics and Language Processing.
The goal of this project is to develop a classifier that is able to detect bots and gender based on 100 tweets and also classifies the gender of human users.
## Data
This project is using the Bots and Gender Profiling 2019 dataset. For copyright reasons, the dataset is not included in this public repository and can be found here: https://pan.webis.de/clef19/pan19-web/author-profiling.html
## Pre processing
the words are tokenize and embedded via Global Vector word representation using the matrix pretrained on 2Bn tweets from [standford](https://nlp.stanford.edu/projects/glove/). The pretrained data is automatically downloaded when training is ran for the first time. Stopwords and punctuation are not removed as they are part of the glove dataset. Some tokens like users, number, hashtags and links are generalized to labels present in the glove matrix. There are 25-200 dimensions in the glove files available, but the file for the 200d matrix is so big that the code is likely to crash due to insufficient memory.
## Model 
Following the approach of [this paper by Wei and Nguyen](https://arxiv.org/abs/2002.01336v1), a bidirectional LSTM with 3 layers and 100 units is used to classify the input (all tweets of one user) to bot, female or male.
## Training
The model paremeters and number of dimensions can be specified in the main script that also starts training and plots the lost.
import xml.etree.ElementTree as ET
import os
import re

def loadData(path = 'data/pan19-author-profiling-20200229/training/en/'):
    '''
    Reads tweets and truth from the given path. Returns a list of lists with tweets of a users 
    and a list of the gender of the corresponding user (labels)
    
    Parameters:
    path (str): path of the dataset

    Returns:
    (list, list): tuple of texts and genders
    '''
    datasets = []
    for file in os.listdir(path):
        if file.endswith(".xml"):
            datasets.append(file)

    tweets = {}
    for dataset in datasets:
        root = ET.parse(os.path.join(path,dataset)).getroot()
        tweet_texts = []
        # get text from tweets
        for type_tag in root.findall('documents/document'):
            text = type_tag.text
            tweet_texts.append(text)
        user_id = re.findall(r"(.*)\.xml", dataset)[0]
        tweets[user_id] = tweet_texts

    labels = {}
    # get truth
    with open(os.path.join(path, 'truth.txt')) as f:
        for line in f:
            user_id, kind, gender = re.findall(r'([A-Za-z0-9]*):::(human|bot):::([a-z]*)', line)[0]
            labels[user_id] = gender

    # create lists for input and output
    x, y =([] for i in range(2))

    for key, value in tweets.items():
        x.append(value)
        y.append(labels[key])

    return x,y

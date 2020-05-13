import re
import requests
import zipfile
import pathlib
import nltk
import numpy as np
from nltk.corpus import stopwords
# from gensim.models import Word2Vec



def download_glove():
    ''''
        Downloads and unzips the Glove Twitter word embeddings
    '''
    print('Beginning file download with requests')
    url = 'http://nlp.stanford.edu/data/glove.twitter.27B.zip'
    r = requests.get(url)
    with open('data/glove.twitter.27B.zip', 'wb') as f:
        f.write(r.content)
    
    #unzip
    pathlib.Path("/data/glove").mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile('data/glove.twitter.27B.zip', 'r') as zip_ref:
        zip_ref.extractall('data/glove')
    


def tweet_tokenization(tweet):
    # tag special expressions
    tagged = re.sub(r'(?:@[\w_]+)', "<USER>", tweet) # @-mentions
    tagged = re.sub(r'(?:\#+[\w_]+[\w\'_\-]*[\w_]+)', "<HASHTAG>", tagged) # hash-tags
    tagged = re.sub(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|![*\(\),]|(?:%[0-9a-f][0-9a-f]))+', "<URL>", tagged) # URL
    tagged = re.sub(r'(?:(?:\d+,?)+(?:\.?\d+)?)', "<NUMBER>", tagged) # numbers
    regex_str = [
        r'<[A-Z]*>', #tags
        r'<[^>]+', # HTML tags
        r'(?:@[\w_]+)', # @-mentions
        r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
        r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|![*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
        r'&amp;', # &amp; tags
        r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
        r"(?:[a-z][a-z'\- ]+[a-z]])", # words with - and '
        r'(?:[\w_]+)', # other words
        r'(?:\S)' #anything else
    ]
    tokens_re = re.compile(r'('+'|'.join(regex_str)+ ')', re.VERBOSE | re.IGNORECASE)
    stop_word = set(stopwords.words('english'))
    word_tokens = tokens_re.findall(tagged)


    # remove stopwords and punctuation
    # return [w.lower() for w in word_tokens if (w.isalpha() or w[0]=="<") and len(w) > 1 and not w.lower() in stop_word and w != "RT"]
    return word_tokens

# https://radimrehurek.com/gensim/models/word2vec.html

# load pretrained word embeddings for GloVe
# glove source: https://nlp.stanford.edu/projects/glove/
def loadGlove(dim=25):
    glove_dict = {}
    with open ("data/glove/glove.twitter.27B."+str(dim)+"d.txt",  encoding="utf8") as f:
        for line in f:
            word = line.split()[0]
            vec = [float(val) for val in line.split()[1:]]
            glove_dict[word] = vec
    return glove_dict


def embed_words(user_tweets, dim=25):
    print("Load embeddings for %i dimensions..." % (dim))
    # this might take a while for high dimensions
    glove_dict = loadGlove(dim)

    # list of all sequences
    embedded_words = []
    # iterate over all users 
    for tweets in user_tweets:
        # iterate over all tweets per user
        # one sequence contains all tweets of a user
        seq = [] 
        for tweet in tweets:
            tokens = tweet_tokenization(tweet)
            for token in tokens:
                try:
                    seq.append(glove_dict[token])
                except KeyError:
                    pass
        embedded_words.append(seq)
    
    return embedded_words

def zero_pad(embedded_words, labels):
    lengths = [len(seq) for seq in embedded_words]
    max_len = max(lengths)
    min_len = min(lengths)
    median = np.median(lengths)
    mean = np.mean(lengths)
    std = np.std(lengths)
    print("Stats for sequence lengths: min=%i, max=%i, mean=%i, median=%i, std=%i" % (min_len, max_len, mean, median, std))
    dim = len(embedded_words[0][0])
    padded = []
    for idx, seq in enumerate(embedded_words):
        if len(seq) < median - std  or len(seq) > median + std:
            del labels[idx]
        else:
            for i in range(int(median) + int(std)-len(seq)):
                seq.append(np.zeros(dim))
            padded.append(seq)
    return padded
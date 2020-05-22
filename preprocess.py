import re
import requests
import zipfile
import pathlib
import nltk
import os
import sys
import numpy as np
from nltk.corpus import stopwords
from tqdm import tqdm
# from gensim.models import Word2Vec



def download_glove():
    ''''
        Downloads and unzips the Glove Twitter word embeddings
    '''
    pathlib.Path("data/glove").mkdir(parents=True, exist_ok=True)
    url = 'http://nlp.stanford.edu/data/glove.twitter.27B.zip'
    file_name = 'data/glove.twitter.27B.zip'
    with open(file_name, "wb") as f:
        print("Downloading %s" % (file_name))
        response = requests.get(url, stream=True)
        total_length = response.headers.get('content-length')

        if total_length is None: # no content length header
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                done = int(50 * dl / total_length)
                sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50-done)) )    
                sys.stdout.flush()
    
    #unzip
    with zipfile.ZipFile('data/glove.twitter.27B.zip', 'r') as zip_ref:
        zip_ref.extractall('data/glove')
    os.remove("data/glove.twitter.27B.zip")


def tweet_tokenization(tweet, remove_stopwords = False):
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
    if remove_stopwords:
        return [w.lower() for w in word_tokens if (w.isalpha() or w[0]=="<") and len(w) > 1 and not w.lower() in stop_word and w != "RT"]
    else:
        return word_tokens

# https://radimrehurek.com/gensim/models/word2vec.html

# load pretrained word embeddings for GloVe
# glove source: https://nlp.stanford.edu/projects/glove/
def loadGlove(dim=25):
    filename = "data/glove/glove.twitter.27B."+str(dim)+"d.txt"
    # download glove file if they don't exist
    if not pathlib.Path(filename).is_file():
        download_glove()
    glove_dict = {}
    with open (filename,  encoding="utf8") as f:
        for line in f:
            word = line.split()[0]
            vec = [float(val) for val in line.split()[1:]]
            glove_dict[word] = vec
    return glove_dict


def embed_words(user_tweets, labels, dim=25):
    print("Load embeddings for %i dimensions..." % (dim))
    # this might take a while for high dimensions
    glove_dict = loadGlove(dim)

    # list of all sequences
    embedded_words = []
    y = [] 
    # iterate over all users 
    for idx, tweets in enumerate(user_tweets):
        # iterate over all tweets per user
        # one sequence contains all tweets of a user 
        for tweet in tweets:
            seq = []
            tokens = tweet_tokenization(tweet)
            for token in tokens:
                try:
                    seq.append(glove_dict[token])
                except KeyError:
                    pass
            embedded_words.append(seq)
            y.append(labels[idx]) 
    
    return embedded_words

def zero_pad(embedded_words, labels):
    lengths = [len(seq) for seq in embedded_words]
    max_len = max(lengths)
    min_len = min(lengths)
    median = np.median(lengths)
    mean = np.mean(lengths)
    std = np.std(lengths)
    print("Stats for sequence lengths: min=%i, max=%i, mean=%i, median=%i, std=%i" % (min_len, max_len, mean, median, std))
    try:
        dim = len(embedded_words[0][0])
    except TypeError:
        dim = 1
    padded = []
    labels_new = []
    for idx, seq in enumerate(embedded_words):
        if not( len(seq) < median - std  or len(seq) > median + std):
            for i in range(int(median) + int(std)-len(seq)):
                if dim ==1:
                    seq.append(0)
                else:
                    seq.append(np.zeros(dim))
            padded.append(seq)
            labels_new.append(labels[idx])
    print("Original length:%d, trimmed length:%d" % (len(embedded_words), len(padded)))
    return padded, labels_new

def get_word_to_ix(user_tweets, word_to_ix={}):
    for tweets in user_tweets:
        for tweet in tweets:
            tokens = tweet_tokenization(tweet, True)
            for token in tokens:
                if token not in word_to_ix:
                    word_to_ix[token] = len(word_to_ix)+1
    return word_to_ix

def prepare_sequence(user_tweets, labels, word_to_ix={},):
    # list of all sequences
    sequences = []
    y =[] 
    # iterate over all users 
    print("Prepare sequences")
    for idx, tweets in tqdm(enumerate(user_tweets)):
        # iterate over all tweets per user
        # one sequence contains all tweets of a user
        for tweet in tweets:
            seq = [] 
            tokens = tweet_tokenization(tweet, True)
            for token in tokens:
                if token not in word_to_ix:
                    word_to_ix[token] = len(word_to_ix)+1
                seq.append(word_to_ix[token])
            sequences.append(seq)
            y.append(labels[idx]) 
    return sequences, y, word_to_ix

def preprocess(x, y, dimensions, glove = True):
    if glove:
        # embedd words using glove matrix
        x= embed_words(x,dimensions)
    else:
        # TODO add words from test set
        word_to_ix = get_word_to_ix(x)
        x = prepare_sequence(x, word_to_ix)
    
    # remove too short and too long sequences, padd with zeros 
    x,y = zero_pad(x,y)
    return x,y

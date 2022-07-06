import pandas as pd
import json
import random

import nltk
import re
import string
from nltk.corpus import stopwords
nltk.download('stopwords')


# Do not forget to change to place where the data is stored
with open('C:/Users/Gina/Downloads/Semester 2/WS/Bots Assignment/Project Implementation/Twibot-20/test.json') as file:
    json_data = json.loads(file.read())

random.seed(23)

ID, location, label, tweets = [], [], [], []

for obj in json_data:
    ID.append(obj['profile']['id'])
    location.append(obj['profile']['location'])
    label.append(obj['label'])

    if isinstance(obj['tweet'], list):
        if len(obj['tweet']) > 20:
            tweets_all = obj['tweet']
            tweets_sampled = random.sample(tweets_all, 20)  # does seed work for it?
            tweets_sampled_cleaned = nlp_preprocessing(tweets_sampled)
            tweets.append(tweets_sampled_cleaned)
        else:
            tweets_cleaned = nlp_preprocessing(obj['tweet'])
            tweets.append(tweets_cleaned)  # if less 20 do we actually want it?
    else:
        label.append("NaN")


df = pd.DataFrame(list(zip(ID, location, label, tweets)), columns=['ID', 'location', 'label', 'tweets'])
print(df)

##PREPROCESSING CODE (leave it in a new .py document?)

punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@'

def rmv_links(tweet):
    """Takes a string and removes http, bitly links from it"""
    tweet = re.sub(r'http\S+', '', tweet)   # remove http links
    tweet = re.sub(r'bit.ly/\S+', '', tweet)  # remove bitly links
    tweet = tweet.strip('[link]')   # remove [links]
    tweet = re.sub(r'pic.twitter\S+','', tweet)
    return tweet

def rmv_audiovisual(tweet):
    """Takes a string and removes videos and audio tags or labels"""
    tweet = re.sub('VIDEO:', '', tweet)  # remove 'VIDEO:' from start of tweet
    tweet = re.sub('AUDIO:', '', tweet)  # remove 'AUDIO:' from start of tweet
    return tweet

def lemm_function(token):
    """Returns lemma of a token"""
    return WordNetLemmatizer().lemmatize(token, pos='v')

def tok(tweet):
    """Returns tokenized representation of words in lemma and removes stopwords"""
    result = []
    for token in gensim.utils.simple_preprocess(tweet):
        if token not in gensim.parsing.preprocessing.STOPWORDS \
                and len(token) > 2:  # drops words with less than 3 characters
            result.append(lemm_function(token))
    return result

def nlp_preprocess(tweet):
    """NLP function to clean tweets, stripping noisy characters, removes user mentions
     and tokenization and lemmatization"""

    tweet = tweet.lower()  # tweet to lower case
    tweet = rmv_links(tweet)
    tweet = rmv_audiovisual(tweet)
    tweet = re.sub('(@[A-Za-z0-9_]+)', ' ', tweet) #removes mention to other users
    tweet = re.sub('[' + punctuation + ']+', ' ', tweet)  # strip punctuation
    tweet = re.sub('\n', '', tweet)
    tweet = re.sub('\s+', ' ', tweet)  # remove double spacing
    tweet = re.sub('([0-9]+)', '', tweet)  # remove numbers
    tweet = re.sub(r'([A-   Za-z])\1{2,}', r'\1', tweet)  # character normalization --> todaaaaay = today
    tweet = re.sub('\w*\d\w*', '', tweet)  # remove words containing numbers
    tweet_token_list = tok(tweet)  # apply lemmatization and tokenization
    tweet = ' '.join(tweet_token_list)
    return tweet

def tokenize_tweets(df):
    """Function that reads and return cleaned and preprocessed tweets dataframe."""

    start_time = time.time() #to monitor time the function runs

    df['tokens'] = df.tweet.apply(nlp_preprocess)
    num_tweets = len(df)
    print('Complete. Number of Tweets that have been cleaned and tokenized : {}'.format(num_tweets))

    #main()
    print("--- %s seconds ---" % (time.time() - start_time))
    return df

df = tokenize_tweets(df)
df

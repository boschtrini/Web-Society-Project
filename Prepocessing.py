import pandas as pd
import json
import random

import nltk
import re
import string
from nltk.corpus import stopwords
nltk.download('stopwords')


def clean_tweet(tweet: str) -> str:  # functions in own script
    # https://www.kaggle.com/code/parulpandey/eda-and-preprocessing-for-bert/notebook
    # https://gist.github.com/bicachu/09cc71bb4b0e3711eaf1556b12fa7ad7

    # Look into what we still need, maybe think about writing multiple functions
    # Still need to delete emojis and smiles
    # https://github.com/Deffro/text-preprocessing-techniques
    # https://github.com/Deffro/text-preprocessing-techniques/blob/master/techniques.py
    tweet = re.sub(r'https?://[^ ]+', '', tweet)  # remove https links
    tweet = re.sub(r'@[^ ]+', '', tweet)  # remove usernames (tweeted at)
    tweet = re.sub(r'(RT\s@[^ ]+)', '', tweet)  # remove usernames (re-tweet)
    tweet = re.sub(r'#[^ ]+', '', tweet)  # remove hashtags
    tweet = re.sub(r'VIDEO:', '', tweet)  # remove 'VIDEO:' from start of tweet
    tweet = re.sub(r'AUDIO:', '', tweet)  # remove 'AUDIO:' from start of tweet

    tweet = tweet.lower()  # tweet to lower
    tweet = re.sub('[%s]' % re.escape(string.punctuation), '', tweet)  # remove punctuation
    tweet = re.sub('\n', '', tweet)
    tweet = re.sub(r'([A-Za-z])\1{2,}', r'\1', tweet)  # character normalization --> todaaaaay = today
    tweet = re.sub('\w*\d\w*', '', tweet)  # remove words containing numbers
    tweet = re.sub('([0-9]+)', '', tweet)  # remove numbers
    tweet = re.sub('\s+', ' ', tweet)  # remove double spacing

    return tweet


def nlp_preprocessing(tweets: list) -> list: # functions in own script
    stopword = stopwords.words('english')
    tweets_cleaned = []

    for tweet in tweets:
        tweet = clean_tweet(tweet)
        tweet = nltk.word_tokenize(tweet)
        tweet = [token for token in tweet if token not in stopword]
        tweet = ' '.join(tweet)
        tweets_cleaned.append(tweet)

    return tweets_cleaned


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

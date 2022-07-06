import pandas as pd
import json
import random

# Do not forget to change to place where the data is stored
with open('Twibot-20/train.json') as file:
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
            #tweets_sampled_cleaned = nlp_preprocessing(tweets_sampled)
            tweets.append(tweets_sampled)
        else:
            #tweets_cleaned = nlp_preprocessing(obj['tweet'])
            tweets.append(tweets_sampled)  # if less 20 do we actually want it?
    else:
        label.append("NaN")


df = pd.DataFrame(list(zip(ID, location, label, tweets)), columns=['ID', 'location', 'label', 'tweet']
df1 = df.explode('tweet')
df.head()

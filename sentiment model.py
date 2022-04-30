#!/usr/bin/env python
# coding: utf-8

# In[77]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textblob.classifiers import NaiveBayesClassifier
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import tweepy as tw
from tweepy import OAuthHandler
import time
import re
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import nltk
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from collections import Counter


# In[2]:


data = pd.read_csv('bfml.csv')
data


# In[3]:


data.isnull().sum()


# In[4]:


data.label.value_counts()


# In[6]:


data.label = data.label.replace({'Positive': 'positive',
                                'Postive': 'positive',
                                'postive': 'positive',
                                'positve': 'positive',
                                'poitive': 'positive',
                                'poistive': 'positive',
                                
                                'Negative': 'negative',
                                 
                                'Neutral': 'neutral'})


# In[7]:


data.label.value_counts()


# In[8]:


data = data[(data['label'] == 'positive') | (data['label'] == 'negative') | (data['label'] == 'neutral')]
data = data[['tweet_text', 'label']]
# data


# In[9]:


data['tweet_text'] = data['tweet_text'].astype(str)
data['label'] = data['label'].astype(str)


# In[10]:


data['data_col'] = data[['tweet_text', 'label']].apply(tuple, axis=1)


# In[11]:


# data


# In[12]:


traindf = data[:1100]
# traindf


# In[13]:


testdf = data[1100:]
# testdf


# In[14]:


train = traindf['data_col']
train = train.values.tolist()
testdf = testdf['data_col']
testdf = testdf.values.tolist()


# In[15]:


cl = NaiveBayesClassifier(train)


# In[16]:


cl.classify('RT @AirgoneFr: Congratulations and thank you to@CreatendCollect who just bought /" Overflow /" 🤩(black friday sale) Now owner of 7 pieces f…')


# In[17]:


cl.accuracy(testdf)


# ### Get recent tweets

# In[22]:


api_key = ['Yjc72iLaViJKBgUkgQTlbxinB']
api_secret_key = ['e94PdXwuFOuwU8Vh0nveP0MiWlu2NenKH8sdGMiLIoIkn1v2IJ']
access_token = ['898536142852718592-JrOTEmiLUWaJTzKVw1kVZbk27YqBtnK']
access_token_secret =['uWV4eQf0rDqzA8idPs1nXRo1l1NiAnegUZcaPcCjAZoDZ']


# In[23]:


auth = tw.OAuthHandler('Yjc72iLaViJKBgUkgQTlbxinB','e94PdXwuFOuwU8Vh0nveP0MiWlu2NenKH8sdGMiLIoIkn1v2IJ')
auth.set_access_token('898536142852718592-JrOTEmiLUWaJTzKVw1kVZbk27YqBtnK','uWV4eQf0rDqzA8idPs1nXRo1l1NiAnegUZcaPcCjAZoDZ')
api = tw.API(auth, wait_on_rate_limit=True)


# In[25]:


tweets = []
def tweets_query(api,query,count,max_requests):
    tweets = tw.Cursor(api.search_tweets, q=query,lang="en",since_id='2021-30-11').items(count)
    #tweets
    tweets_list  = [[tweet.text, tweet.id, tweet.source, tweet.coordinates, tweet.retweet_count, tweet.favorite_count,
                tweet.user._json['name'],tweet.user._json['screen_name'], tweet.user._json['location'], tweet.user._json['friends_count'],
                tweet.user._json['verified'], tweet.user._json['description'], tweet.user._json['followers_count']] for tweet in tweets]
    tweets_df= pd.DataFrame(tweets_list, columns = ['tweet_text','tweet_id', 'tweet_source','coordinates','retweet_count','likes_count','Username', 'screen_name','location', 'friends_count','verification_status','description','followers_count'])
    return tweets_df


# In[26]:


query = 'black friday'
count = 1500
max_requests = 3
tweets= tweets_query(api,query,count,max_requests)


# In[32]:


tweetdf = (tweets
 .sort_values(['followers_count'], ascending=False)
 [['tweet_text', 'retweet_count', 'followers_count']])


# In[33]:


tweetdf['tweet_text'] = tweetdf['tweet_text'].astype(str)
#tweetdf['tweetdf_col'] = tweetdf[['tweet_text']].apply(tuple, axis=1)


# In[35]:


inference = tweetdf['tweet_text']
inference = inference.values.tolist()


# In[42]:


sentiments = []
for tweet in inference:
    sentiment = cl.classify(tweet)
    sentiments.append(sentiment) 


# In[51]:


tweetdf['sentiments'] = np.asarray(sentiments)


# In[62]:


tweetdfdict  = tweetdf['sentiments'].value_counts().to_dict()


# In[68]:


plt.bar(list(tweetdfdict.keys()), list(tweetdfdict.values()))

plt.xlabel('Sentiments', fontsize=12)
plt.ylabel('Total Number', fontsize=12)
plt.title('Sentiment Analysis on Black Friday tweets', fontsize=14)
plt.show()


# ### Wordcloud

# In[70]:


tweetdf


# In[73]:


pos_review = tweetdf[tweetdf['sentiments'] == 'positive']


# In[74]:


text = " ".join(tweet for tweet in pos_review.tweet_text.astype(str))


# In[75]:


#Generating the word cloud image
stopwords=set(STOPWORDS)
wordcloud_pos = WordCloud(stopwords=stopwords, background_color='white').generate(text)

plt.figure(figsize=[20,10])
plt.imshow(wordcloud_pos, interpolation='bilinear')
plt.axis("off")
plt.show()


# ### Tabular word frequency

# In[93]:


stop = stopwords.words('english')
newStopWords = ['rt', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
               '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30']
stop.extend(newStopWords)


# In[94]:


pos_review["tweet_text_stop"] = pos_review["tweet_text"].str.replace("[^\w\s]", "").str.lower()


# In[95]:


pos_review["tweet_text_stop"] = pos_review["tweet_text_stop"].apply(lambda x: ' '.join([item for item in x.split() if item not in stop]))
# a = pos_review["tweet_text_stop"].str.split(expand=True).stack().value_counts()


# In[96]:


Counter(" ".join(pos_review["tweet_text_stop"].str.lower()).split()).most_common(100)


# In[ ]:





# In[ ]:




